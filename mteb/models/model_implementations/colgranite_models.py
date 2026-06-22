from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction
from safetensors.torch import load_file

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

from .colpali_models import (
    COLPALI_CITATION,
    COLPALI_TRAINING_DATA,
    ColPaliEngineWrapper,
)
from colpali_engine.models import ColGranite4Vision, ColGranite4VisionProcessor, ColGranite4VisionDistill

logger = logging.getLogger(__name__)


class ColGranite4Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColGranite4 model.

    Supports both a full fine-tune dir (``model.safetensors``) and a LoRA dir
    (``adapter_config.json``). LoRA dirs are loaded explicitly via
    ``peft.PeftModel.from_pretrained`` rather than through transformers'
    ``from_pretrained`` adapter auto-load path, which is brittle across
    transformers/peft version skews.
    """

    def __init__(
        self,
        model_name: str = "/proj/dmfexp/yangy/vlm_distill/colpali/colgranite-multidataset-5e-5-0.05",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        
        is_lora = os.path.exists(os.path.join(model_name, "adapter_config.json"))
        if not is_lora:
            super().__init__(
                model_name=model_name,
                model_class=ColGranite4Vision,
                processor_class=ColGranite4VisionProcessor,
                revision=revision,
                device=device,
                **kwargs,
            )
            return

        #requires_image_dependencies()
        #requires_package(
        #    self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        #)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(model_name, "adapter_config.json")) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        model = ColGranite4Vision.from_pretrained(
            base_model_name,
            device_map=self.device,
            **kwargs,
        )

        model = PeftModel.from_pretrained(model, model_name, revision=revision)
        print('peft lora merged')
        try:
            model = model.merge_and_unload()
        except Exception as e: 
            logger.warning("merge_and_unload failed (%s); running unmerged.", e)

        self.mdl = model.to(self.device)
        self.mdl.eval()

        self.processor = ColGranite4VisionProcessor.from_pretrained(base_model_name)


class ColGranite4DistillWrapper(AbsEncoder):
    """Wrapper for `ColGranite4VisionDistill` (summary-token) models.

    Unlike the plain ColGranite4 model, the distill model:
      * appends K learned summary tokens on the DOC (image) side and returns ONLY
        those K projected vectors -- this happens in ``forward`` when
        ``is_query=False`` and ``pixel_values`` are present;
      * uses standard per-token vectors on the QUERY (text) side, where the
        ``forward`` must be called with ``is_query=True``.

    The learned ``summary_tokens`` / cross-attn matcher are custom params absent
    from the base checkpoint, so the loader must call ``init_distill_modules()``
    after ``from_pretrained`` and then load the trained weights into them.

    ``checkpoint_path`` may be either:
      * a full fine-tune dir containing ``model.safetensors`` (loaded by
        load_state_dict on top of the base model + initialized distill modules), or
      * a LoRA dir containing ``adapter_config.json`` (loaded via
        ``peft.PeftModel.from_pretrained``).
    The flavor is auto-detected from the directory contents.
    """

    def __init__(
        self,
        name: str,
        base_model_name: str = "ibm-granite/granite-vision-4.1-4b",
        num_summary_tokens: int = 16,
        num_retrieval_tokens: int = 0,
        summary_distill_heads: int = 8,
        doc_use_summary_tokens_only: bool = True,
        query_pooling: str = "none",
        processor_name: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):

        checkpoint_path = name

        #requires_image_dependencies()
        #requires_package(
        #    self, "colpali_engine", checkpoint_path, "pip install mteb[colpali_engine]"
        #)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_summary_tokens = num_summary_tokens

        is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

        # 1) Build the base distill model from the ORIGINAL granite backbone with matching summary-token config, then materialize the custom params.
        model = ColGranite4VisionDistill.from_pretrained(
            base_model_name,
            num_summary_tokens=num_summary_tokens,
            num_retrieval_tokens=num_retrieval_tokens,
            summary_distill_heads=summary_distill_heads,
            doc_use_summary_tokens_only=doc_use_summary_tokens_only,
            query_pooling=query_pooling,
            device_map=self.device,
            **kwargs,
        )
        model.init_distill_modules()

        # 2) Load the trained weights.
        if is_lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, checkpoint_path, revision=revision
            )
            try:
                model = model.merge_and_unload()
            except Exception as e:  # noqa: BLE001
                logger.warning("merge_and_unload failed (%s); running unmerged.", e)
        else:
            state = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            state = {k.removeprefix("model."): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            # summary_tokens / custom_text_proj / distill matcher must be present.
            crit = [k for k in missing if "summary_tokens" in k or "custom_text_proj" in k]
            if crit:
                raise RuntimeError(
                    f"Trained summary/proj weights missing from checkpoint: {crit}"
                )

        self.mdl = model.to(self.device)
        self.mdl.eval()

        self.processor = ColGranite4VisionProcessor.from_pretrained(
            processor_name or base_model_name
        )

    # --- routing: queries vs docs need different forward flags ---
    def encode_input(self, inputs, is_query: bool):
        return self.mdl(**inputs, is_query=is_query)

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs,
    ):
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            raise ValueError(
                "Distill eval expects text (query) and image (doc) in separate "
                "passes; got both in one batch."
            )
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in inputs.")

    def get_image_embeddings(self, images, batch_size: int = 32, **kwargs):
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images (docs)"):
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                proc = self.processor.process_images(imgs)
                proc = {k: v.to(self.device) for k, v in proc.items()}
                # is_query=False + pixel_values present -> doc path: append the K
                # summary tokens and return ONLY those projected vectors.
                outs = self.encode_input(proc, is_query=False)
                all_embeds.extend(outs.cpu().to(torch.float32))

        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def get_text_embeddings(self, texts, batch_size: int = 32, **kwargs):
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts (queries)"):
                batch = [  # noqa: PLW2901
                    self.processor.query_prefix
                    + t.replace("<image>", "")
                    + self.processor.query_augmentation_token * 10
                    for t in batch["text"]
                ]
                proc = self.processor.process_texts(batch)
                proc = {k: v.to(self.device) for k, v in proc.items()}
                outs = self.encode_input(proc, is_query=True)
                all_embeds.extend(outs.cpu().to(torch.float32))

        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def similarity(self, a, b):
        return self.processor.score(a, b, device=self.device)

colgranite_model = ModelMeta(
    loader=ColGranite4Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="/proj/dmfexp/yangy/vlm_distill/colpali/colgranite-multidataset-5e-5-0.05",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="4ad8f151e39bce3adcf88e0bdd72e724c7606638",
    release_date="2026-03-15",
    modalities=["image", "text"],
    n_parameters=4_600_000_000,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8660,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "ColPali", "safetensors"],
    reference="https://huggingface.co/ibm-granite/granite-vision-4.1-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLPALI_TRAINING_DATA,
)


colgranite_distill_model = ModelMeta(
    loader=ColGranite4DistillWrapper,
    loader_kwargs=dict(
        base_model_name="ibm-granite/granite-vision-4.1-4b",
        num_summary_tokens=16,
        summary_distill_heads=8,
        doc_use_summary_tokens_only=True,
        torch_dtype=torch.bfloat16,
    ),
    name="/proj/dmfexp/yangy/vlm_distill/colpali/colgranite-distill-1e-5-0.5",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="1",
    release_date="2026-06-04",
    modalities=["image", "text"],
    n_parameters=4_600_000_000,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8660,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "ColPali", "safetensors"],
    reference="https://huggingface.co/ibm-granite/granite-vision-4.1-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLPALI_TRAINING_DATA,
)


# Same distill model, but the QUERY side is mean-pooled to a single vector
# (doc side stays K summary tokens). Scoring is 1 query vector x K doc vectors via MaxSim. 
colgranite_distill_meanq_model = ModelMeta(
    loader=ColGranite4DistillWrapper,
    loader_kwargs=dict(
        base_model_name="ibm-granite/granite-vision-4.1-4b",
        num_summary_tokens=16,
        summary_distill_heads=8,
        doc_use_summary_tokens_only=True,
        query_pooling="mean",
        torch_dtype=torch.bfloat16,
    ),
    name="/proj/dmfexp/yangy/vlm_distill_granite/colpali/colgranite-peft-1-colpali-4e-5",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="1",
    release_date="2026-06-05",
    modalities=["image", "text"],
    n_parameters=4_600_000_000,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8660,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "ColPali", "safetensors"],
    reference="https://huggingface.co/ibm-granite/granite-vision-4.1-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLPALI_TRAINING_DATA,
)
