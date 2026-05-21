from __future__ import annotations

import logging
from functools import partial

import torch
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.models import ModelMeta
from mteb.models.model_implementations.colpali_models import ColPaliEngineWrapper
from mteb._requires_package import (
    requires_package,
)
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)


class ColModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColModernVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/colvbert-modernbert_base-vidore",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColModernVBert,
            processor_class=ColModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )
        
        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])


class BiModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for BiVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/bivbert-slbert_210",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        # Keep pooling_strategy in kwargs so from_pretrained passes it to __init__,
        # which creates the necessary modules (position_token_scorer, expert_heads, etc.)
        pooling_strategy = kwargs.get("pooling_strategy", None)
        eval_scoring = kwargs.pop("eval_scoring", None)
        use_query_proj = kwargs.get("use_query_proj", None)
        query_proj_mode = kwargs.pop("query_proj_mode", None)
        self._matryoshka_dim = kwargs.pop("matryoshka_dim", None)
        print('pooling strategy',pooling_strategy,eval_scoring,use_query_proj)
        # Check if this is a PEFT checkpoint (has adapter_config.json).
        # If so, load base model first, then apply PEFT adapter properly
        # so that modules_to_save (e.g. pos_projection) are loaded correctly.
        import os, json
        adapter_config_path = os.path.join(model_name, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            from peft import PeftModel
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg["base_model_name_or_path"]
            print(f"[BiModernVBertWrapper] PEFT checkpoint detected, loading base from {base_model_path}")

            from mteb._requires_package import requires_image_dependencies
            requires_image_dependencies()

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            base_model = BiModernVBert.from_pretrained(
                base_model_path,
                device_map=self.device,
                **kwargs,
            )
            if hasattr(base_model, '_apply_cross_attn_wrapping') and base_model.mid_layer_cross_attn:
                base_model._apply_cross_attn_wrapping()
            self.mdl = PeftModel.from_pretrained(base_model, model_name)
            peft_state_dict = get_peft_model_state_dict(self.mdl)

            # Print the keys
            for key in peft_state_dict.keys():
                print(key)
            self.mdl = self.mdl.merge_and_unload()
            self.mdl.eval()
            print(self.mdl)
            self.processor = BiModernVBertProcessor.from_pretrained(model_name)
            # Move cross-attn modules to the correct device after merge_and_unload
            if hasattr(self.mdl, '_cross_attn_config') and self.mdl._cross_attn_config:
                for layer_idx in self.mdl._cross_attn_config["after_layers"]:
                    self.mdl.model.text_model.layers[layer_idx].cross_attn.to(self.device)
            print(f"[BiModernVBertWrapper] PEFT adapter merged successfully")
            self._cross_attn_loaded = True
        else:
            super().__init__(
                model_name=model_name,
                model_class=BiModernVBert,
                processor_class=BiModernVBertProcessor,
                revision=revision,
                device=device,
                **kwargs,
            )
        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])

        if hasattr(self.mdl, '_apply_cross_attn_wrapping') and self.mdl.mid_layer_cross_attn and not getattr(self, '_cross_attn_loaded', False):
            self.mdl._apply_cross_attn_wrapping()
            import safetensors.torch
            import glob
            shard_files = sorted(glob.glob(os.path.join(model_name, "model*.safetensors")))
            if not shard_files:
                shard_files = sorted(glob.glob(os.path.join(model_name, "*.safetensors")))
            cross_attn_keys = {}
            for shard_path in shard_files:
                shard_state = safetensors.torch.load_file(shard_path, device="cpu")
                for k, v in shard_state.items():
                    if "original_layer" not in k and "cross_attn" not in k:
                        continue
                    clean_key = k
                    for prefix in ("base_model.model.", "model."):
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    cross_attn_keys[clean_key] = v
            if cross_attn_keys:
                missing, unexpected = self.mdl.load_state_dict(cross_attn_keys, strict=False)
                print(f"[BiModernVBertWrapper] Loaded {len(cross_attn_keys)} cross-attn/wrapped-layer keys (missing={len(missing)}, unexpected={len(unexpected)})")
            model_device = next(self.mdl.model.text_model.layers[0].parameters()).device
            for layer_idx in self.mdl._cross_attn_config["after_layers"]:
                self.mdl.model.text_model.layers[layer_idx].cross_attn.to(model_device)

        if pooling_strategy is not None:
            print(f"[INFO] pooling_strategy: {pooling_strategy}")
            self.mdl.pooling_strategy = pooling_strategy

        if eval_scoring:
            self.processor.eval_scoring = eval_scoring
            print(f"[INFO] eval_scoring: {eval_scoring}")

        self._is_query = False
        self._query_image_embeds = None
        self._query_proj_mode = query_proj_mode

    def get_text_embeddings(self, texts, prompt_type=None, **kwargs):
        from mteb.types import PromptType
        self._is_query = prompt_type == PromptType.query
        text_embeds = []
        image_embeds = []

        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                if self._is_query:
                    batch_texts = [
                        self.processor.query_prefix
                        + t.replace("<image>", "")
                        + self.processor.query_augmentation_token * 10
                        for t in batch["text"]
                    ]
                else:
                    batch_texts = [t.replace("<image>", "") for t in batch["text"]]
                inputs = self.processor.process_texts(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self.encode_input(inputs)

                text_embeds.extend(outs.cpu().to(torch.float32))

        self._is_query = False

        padded = torch.nn.utils.rnn.pad_sequence(
            text_embeds, batch_first=True, padding_value=0
        )
        if image_embeds:
            self._query_image_embeds = torch.nn.utils.rnn.pad_sequence(
                image_embeds, batch_first=True, padding_value=0
            )
        else:
            self._query_image_embeds = None

        return padded

    def get_image_embeddings(self, images, **kwargs):
        self._is_query = False
        return super().get_image_embeddings(images, **kwargs)

    def _truncate(self, embeddings):
        if self._matryoshka_dim is None:
            return embeddings
        truncated = embeddings[..., :self._matryoshka_dim]
        return torch.nn.functional.normalize(truncated, p=2, dim=-1)

    def encode_input(self, inputs):
        if self._is_query and self._query_proj_mode is not None:
            return self._truncate(self.mdl(is_query=True, query_proj_mode=self._query_proj_mode, **inputs))
        return self._truncate(self.mdl(is_query=self._is_query, **inputs))

    def similarity(self, a, b):
        if self._query_image_embeds is not None:
            p_sample = b[0] if isinstance(b, list) else b[0]
            if p_sample.dim() == 2:
                return self.processor.score(self._query_image_embeds, b, device=self.device)
        return self.processor.score(a, b, device=self.device)

# colvbert_modernvbert_base = ModelMeta(
#     loader=partial(
#         ColModernVBertWrapper,
#         model_name="SmolVEncoder/colvbert-modernbert_base-vidore",
#         torch_dtype=torch.float16,
#         attn_implementation="flash_attention_2"
#         if is_flash_attn_2_available()
#         else None,
#     ),
#     name="SmolVEncoder/colvbert-modernbert_base-vidore",
#     languages=["eng-Latn"],
#     revision="c71ee9a431b74e87c138460f38be01248984d2f4",
#     release_date="2025-06-01",
#     modalities=["image", "text"],
#     n_parameters=252_000_000,
#     memory_usage_mb=480,
#     max_tokens=8192,
#     embed_dim=128,
#     license="apache-2.0",
#     open_weights=True,
#     public_training_code="https://github.com/illuin-tech/colpali",
#     public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
#     framework=["ColPali"],
#     reference="https://huggingface.co/SmolVEncoder/colvbert-modernbert_base-vidore",
#     similarity_fn_name="max_sim",
#     use_instructions=True,
#     training_datasets=COLPALI_TRAINING_DATA,
# )