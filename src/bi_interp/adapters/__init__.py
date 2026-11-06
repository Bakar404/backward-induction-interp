"""PyTorch and HuggingFace Transformers Adapter Hooks."""

from bi_interp.adapters.hf_hooks import (
    extract_mlp_down_projection_numpy,
    apply_surgical_mlp_edit_to_hf_model,
)

__all__ = [
    "extract_mlp_down_projection_numpy",
    "apply_surgical_mlp_edit_to_hf_model",
]
