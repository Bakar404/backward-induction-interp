"""
Adapters for Inspecting and Surgically Editing Real PyTorch / HuggingFace Models
(such as Qwen/Qwen3.5-0.8B, google/gemma-3-270m, and meta-llama architectures).

Operates without gradient descent (`torch.no_grad()`), bridging `bi_interp` NumPy
backward-induction solvers directly to HuggingFace `PreTrainedModel` weight tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence
import numpy as np

from bi_interp.editing.surgical_editor import project_out_mlp_loop_direction


def _get_transformer_layers(model: Any) -> Any:
    """Resolve the layer ModuleList across Qwen, Gemma, Llama, and GPT-NeoX architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Could not locate transformer layer list on model (expected model.model.layers or model.transformer.h)."
    )


def _get_mlp_down_proj_module(layer: Any) -> Any:
    """Resolve the MLP down-projection Linear module writing to the residual stream."""
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for attr in ("down_proj", "c_proj", "dense_4h_to_h", "fc2"):
            if hasattr(mlp, attr):
                return getattr(mlp, attr)
    raise AttributeError("Could not locate MLP down-projection module on transformer layer.")


def extract_mlp_down_projection_numpy(model: Any, layer_idx: int) -> np.ndarray:
    """Extract a layer's MLP down-projection matrix W_down as a float64 NumPy array."""
    layers = _get_transformer_layers(model)
    down_mod = _get_mlp_down_proj_module(layers[layer_idx])
    weight = down_mod.weight
    if hasattr(weight, "detach"):
        return weight.detach().cpu().float().numpy().astype(np.float64)
    return np.asarray(weight, dtype=np.float64)


def apply_surgical_mlp_edit_to_hf_model(
    model: Any,
    target_layers: Sequence[int],
    loop_directions: Dict[int, np.ndarray],
    top_k_columns: int = 64,
    scale: float = 1.0,
) -> Dict[int, Dict[str, float]]:
    """
    Apply WSP Experiment 8's surgical rank-1 MLP loop-direction removal directly
    to a HuggingFace model (e.g., Layers 15 and 19 of Qwen3.5-0.8B) with zero gradients.
    """
    layers = _get_transformer_layers(model)
    summary: Dict[int, Dict[str, float]] = {}

    for ell in target_layers:
        w_np = extract_mlp_down_projection_numpy(model, ell)
        edit_res = project_out_mlp_loop_direction(
            w_down=w_np,
            loop_direction=loop_directions[ell],
            top_k_columns=top_k_columns,
            scale=scale,
        )
        down_mod = _get_mlp_down_proj_module(layers[ell])
        weight = down_mod.weight
        if hasattr(weight, "detach") and hasattr(weight, "copy_"):
            import torch  # Imported lazily so pure NumPy environments need not install torch

            with torch.no_grad():
                edited_tensor = torch.from_numpy(edit_res.edited_weight).to(
                    device=weight.device, dtype=weight.dtype
                )
                weight.copy_(edited_tensor)
        else:
            down_mod.weight = edit_res.edited_weight

        summary[ell] = {
            "num_weights_modified": float(edit_res.num_weights_modified),
            "fraction_modified": float(edit_res.fraction_modified),
            "before_loop_alignment": float(edit_res.before_loop_alignment),
            "after_loop_alignment": float(edit_res.after_loop_alignment),
        }

    return summary
