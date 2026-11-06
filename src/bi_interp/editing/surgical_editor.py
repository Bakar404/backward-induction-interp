"""
Surgical, Gradient-Free Weight-Space Editing Guided by Backward Induction.

Implements the three core weight-editing primitives from Takehi et al.
(MIT CSAIL, Nov 2026, Sparks of Weight-Space Programming):
  1. Sparse Rank-1 MLP Output-Column Projection (WSP Exp 8: fixing Qwen3.5-0.8B
     infinite reasoning loops by subtracting the looping direction from top-k
     MLP down-projection columns in target layers, touching <0.02% of weights).
  2. Dormant-Neuron Subgame Pathway Grafting (WSP Exp 4: fixing a PPO policy
     trapped in a ramp local minimum by programming dormant hidden units to
     reverse and build momentum before switching off).
  3. Sycophantic Head OV/QK Suppression (WSP Exp 7 follow-up: neutralizing the
     attention head that routes user suggestions over factual tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


@dataclass
class RankOneProjectionEdit:
    """Summary of a sparse rank-1 weight-space edit on an MLP down-projection matrix."""

    edited_weight: np.ndarray
    edited_column_indices: np.ndarray
    num_weights_modified: int
    total_matrix_weights: int
    fraction_modified: float
    before_loop_alignment: float
    after_loop_alignment: float


def project_out_mlp_loop_direction(
    w_down: np.ndarray,
    loop_direction: np.ndarray,
    top_k_columns: int = 64,
    scale: float = 1.0,
) -> RankOneProjectionEdit:
    """
    Subtract a problematic residual direction `u` (e.g., an infinite CoT repetition
    loop direction in Layers 15 & 19 of Qwen3.5-0.8B, WSP Exp 8) from ONLY the
    `top_k_columns` of `w_down` (shape d_model x d_ff) that align most strongly with `u`.
    """
    w = np.array(w_down, dtype=np.float64, copy=True)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix (d_model, d_ff), got {w.shape}.")

    u = np.asarray(loop_direction, dtype=np.float64).reshape(-1)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        raise ValueError("loop_direction must have non-zero norm.")
    u = u / u_norm

    d_model, d_ff = w.shape
    if u.shape[0] != d_model:
        raise ValueError(f"loop_direction length {u.shape[0]} != d_model {d_model}.")

    k = min(max(int(top_k_columns), 1), d_ff)
    projections = u @ w  # shape (d_ff,)
    before_align = float(np.linalg.norm(projections, ord=2))

    # Select top-k columns by absolute alignment |u^T w_{:, j}|
    top_idx = np.argsort(np.abs(projections))[::-1][:k]

    # Subtract rank-1 component on selected columns only: w_{:, j} <- w_{:, j} - scale * (u^T w_{:, j}) u
    w[:, top_idx] -= scale * np.outer(u, projections[top_idx])

    after_projections = u @ w
    after_align = float(np.linalg.norm(after_projections, ord=2))
    num_modified = d_model * k
    total_weights = d_model * d_ff

    return RankOneProjectionEdit(
        edited_weight=w,
        edited_column_indices=top_idx,
        num_weights_modified=num_modified,
        total_matrix_weights=total_weights,
        fraction_modified=num_modified / float(total_weights),
        before_loop_alignment=before_align,
        after_loop_alignment=after_align,
    )


def graft_subgame_backup_pathway(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    dormant_neuron_indices: Sequence[int],
    pos_feature_idx: int = 0,
    vel_feature_idx: int = 1,
    backup_threshold_pos: float = -0.5,
    min_runway_speed: float = 0.8,
    reverse_action_idx: int = 0,
    forward_action_idx: int = 1,
    graft_gain: float = 6.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Graft a backward-inducted multi-step subgame pathway into dormant hidden units
    of a policy network (inspired by WSP Experiment 4: Kinetix Car-on-Ramp).

    Problem:
      A PPO-trained policy drives forward immediately (`forward_action_idx`) and stalls on
      a steep ramp because it lacks speed (`v < min_runway_speed`). To succeed, the agent
      must first reverse (`reverse_action_idx`) until `pos <= backup_threshold_pos`, then
      accelerate forward at full speed.

    Surgical Weight Edit:
      Uses a small set of dormant hidden units identified by Stage 1's Backward Stopping Game
      to detect when the agent is in front of the backup point (`pos > backup_threshold_pos`)
      without sufficient runway speed (`vel < min_runway_speed`), overriding the output logits
      to reverse until behind the line, where the ReLU unit switches off (0) and the original
      trained forward policy drives over the ramp.

    Returns:
        (edited_weights, edited_biases, fraction_of_weights_changed)
    """
    if len(weights) < 2:
        raise ValueError("Policy network must have at least 1 hidden layer (2 weight matrices).")

    w_list = [np.array(w, dtype=np.float64, copy=True) for w in weights]
    b_list = [np.array(b, dtype=np.float64, copy=True) for b in biases]

    total_params = sum(w.size for w in w_list) + sum(b.size for b in b_list)
    modified_params = 0

    # Use up to 2 dormant units in each hidden layer to carry the backup gate signal
    u0 = int(dormant_neuron_indices[0])

    # Layer 0 (input -> hidden 0):
    # Activate unit u0 via ReLU when: (pos - backup_threshold_pos) > 0 AND (min_runway_speed - vel) > 0
    w_list[0][u0, :] = 0.0
    w_list[0][u0, pos_feature_idx] = 2.5
    w_list[0][u0, vel_feature_idx] = -2.5
    b_list[0][u0] = -2.5 * backup_threshold_pos + 0.25 * min_runway_speed
    modified_params += w_list[0].shape[1] + 1

    # Propagate signal cleanly through intermediate hidden layers (if any) via unit u0
    for ell in range(1, len(w_list) - 1):
        w_list[ell][u0, :] = 0.0
        w_list[ell][u0, u0] = 1.0
        b_list[ell][u0] = 0.0
        modified_params += w_list[ell].shape[1] + 1

    # Output layer: when unit u0 is active (> 0), boost reverse_action and suppress forward_action
    w_list[-1][reverse_action_idx, u0] = graft_gain
    w_list[-1][forward_action_idx, u0] = -graft_gain
    modified_params += 2

    return w_list, b_list, modified_params / float(total_params)


def suppress_sycophantic_head_weights(
    w_o: np.ndarray,
    suppression_factor: float = 0.0,
) -> np.ndarray:
    """
    Neutralize or attenuate the output projection W_O of a sycophantic attention head
    identified by Stage 2's attention subgame resolver (e.g. Head 10.2 in WSP Exp 7).
    """
    return np.array(w_o, dtype=np.float64, copy=True) * float(suppression_factor)
