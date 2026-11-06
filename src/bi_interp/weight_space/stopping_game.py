"""
Backward Stopping Game on Neural Network Weight Graphs (Zero Forward Passes).

Implements the game-theoretic backward attribution formulation building on:
  - Gaubert & Vlassopoulos (2025): ReLU Net Stopping Games
  - Zimmermann et al. (Oct 2026, arXiv:2605.06212): Playing the network backward
  - Takehi et al. (Nov 2026, WSP Exp 1 & 2): Zero-pass game/shape reading from weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class StoppingGameResult:
    """Result of solving the two-player backward stopping game on network weights."""

    # Positive (Player I / excitatory) backward value at input: shape (d_out, d_in)
    v_pos_input: np.ndarray
    # Negative (Player II / inhibitory) backward value at input: shape (d_out, d_in)
    v_neg_input: np.ndarray
    # Net backward subgame value Phi = v_pos_input - v_neg_input: shape (d_out, d_in)
    net_subgame_value: np.ndarray
    # Per-layer neuron game importance (L1 norm of backward value passing through each neuron)
    layer_neuron_importance: List[np.ndarray]
    # Indices of dormant / low-value neurons per hidden layer (candidates for surgical grafting)
    dormant_neurons_per_layer: List[np.ndarray]


def solve_backward_stopping_game(
    weights: Sequence[np.ndarray],
    activation_gates: Optional[Sequence[np.ndarray]] = None,
    dormant_quantile: float = 0.15,
) -> StoppingGameResult:
    """
    Induct backward from output layer L down to input layer 0 using the
    signed two-player Stopping Decomposition (Zimmermann et al., 2026).

    Args:
        weights: Sequence of weight matrices [W^(1), ..., W^(L)], where
            W^(l) has shape (d_l, d_{l-1}).
        activation_gates: Optional per-layer survival/gate diagonals gamma^(l)
            of shape (d_l,). If None, runs in pure zero-pass static weight mode
            (gamma^(l) = 1).
        dormant_quantile: Quantile below which hidden units are flagged as dormant
            (useful for Stage 3 surgical pathway grafting).

    Returns:
        StoppingGameResult containing signed trajectory matrices and neuron importances.
    """
    if not weights:
        raise ValueError("weights must contain at least one layer matrix.")

    num_layers = len(weights)
    d_out = weights[-1].shape[0]

    # Terminal payoff at output layer L: identity matrix (each row k is unit payoff e_k)
    v_pos = np.eye(d_out, dtype=np.float64)
    v_neg = np.zeros((d_out, d_out), dtype=np.float64)

    layer_importance_rev: List[np.ndarray] = []
    dormant_rev: List[np.ndarray] = []

    # Backward induction from l = L down to l = 1
    for ell in range(num_layers - 1, -1, -1):
        w = np.asarray(weights[ell], dtype=np.float64)
        if w.ndim != 2:
            raise ValueError(f"Weight matrix at layer {ell} must be 2D, got shape {w.shape}.")

        if activation_gates is not None and activation_gates[ell] is not None:
            gate = np.asarray(activation_gates[ell], dtype=np.float64).reshape(1, -1)
            v_pos_gated = v_pos * gate
            v_neg_gated = v_neg * gate
        else:
            v_pos_gated = v_pos
            v_neg_gated = v_neg

        # Split weight matrix into cooperative (+) and turn-switching (-) components
        w_pos = np.maximum(w, 0.0)
        w_neg = np.maximum(-w, 0.0)

        # Stopping game recurrence:
        # Positive paths come from (+ after +) or (- after -)
        # Negative paths come from (- after +) or (+ after -)
        next_v_pos = v_pos_gated @ w_pos + v_neg_gated @ w_neg
        next_v_neg = v_pos_gated @ w_neg + v_neg_gated @ w_pos

        v_pos, v_neg = next_v_pos, next_v_neg

        # If ell > 0, the columns of W^(ell) correspond to hidden neurons of layer ell - 1
        if ell > 0:
            # Total game flow magnitude through each hidden neuron j in layer ell - 1
            flow_mag = np.sum(np.abs(v_pos) + np.abs(v_neg), axis=0)
            # Also weigh by outgoing weight norm in layer ell so completely disconnected units score 0
            outgoing_norm = np.linalg.norm(w, ord=1, axis=0)
            importance = 0.5 * (flow_mag + outgoing_norm)
            layer_importance_rev.append(importance)
            threshold = float(np.quantile(importance, dormant_quantile))
            dormant_idx = np.where(importance <= threshold)[0]
            dormant_rev.append(dormant_idx)

    layer_importance = list(reversed(layer_importance_rev))
    dormant_neurons = list(reversed(dormant_rev))
    net_value = v_pos - v_neg

    return StoppingGameResult(
        v_pos_input=v_pos,
        v_neg_input=v_neg,
        net_subgame_value=net_value,
        layer_neuron_importance=layer_importance,
        dormant_neurons_per_layer=dormant_neurons,
    )


def infer_grid_game_from_weights(
    weights: Sequence[np.ndarray],
    grid_shape: Tuple[int, int, int] = (5, 10, 10),
    action_names: Tuple[str, ...] = ("noop", "up", "down", "left", "right"),
) -> Dict[str, object]:
    """
    Zero-Pass Game Identification from MLP Weights (inspired by Takehi et al., 2026, Exp 1).

    Given an MLP policy mapping a 5-channel (agent, moving_obj, target, obstacle, prev_frame)
    H x W grid to 5 actions (noop, up, down, left, right), solves the backward stopping game
    and extracts spatial-action invariants to identify the game with 0 forward passes:
      - Breakout: Agent paddle restricted to bottom row; strong left/right action separation,
        minimal up/down separation.
      - Pong: Agent paddle restricted to vertical side column (left or right border); strong
        up/down action separation, minimal left/right separation.
      - Freeway: Agent moves strictly vertically (up/down) across horizontal traffic lanes
        from bottom to top goal row, minimal left/right control.
      - Pac-Man: Full 2D spatial coverage across all rows/cols with balanced 4-way movement
        (up, down, left, right) and strong obstacle/target channel coupling.
    """
    res = solve_backward_stopping_game(weights)
    c, h, w = grid_shape
    expected_in = c * h * w
    if res.net_subgame_value.shape[1] != expected_in:
        raise ValueError(
            f"Expected input dimension {expected_in} for grid {grid_shape}, "
            f"got {res.net_subgame_value.shape[1]}."
        )

    action_idx = {name: i for i, name in enumerate(action_names)}
    phi = res.net_subgame_value.reshape(len(action_names), c, h, w)

    # Channel 0: controlled agent position; Channel 1: moving objects; Channel 2: targets; Channel 3: obstacles
    horiz_diff = np.abs(phi[action_idx["left"]] - phi[action_idx["right"]])
    vert_diff = np.abs(phi[action_idx["up"]] - phi[action_idx["down"]])

    horiz_energy = float(np.sum(horiz_diff))
    vert_energy = float(np.sum(vert_diff))
    total_dir_energy = horiz_energy + vert_energy + 1e-12
    horiz_ratio = horiz_energy / total_dir_energy
    vert_ratio = vert_energy / total_dir_energy

    # Spatial concentration of agent channel (channel 0) + moving object (channel 1)
    agent_spatial = np.sum(np.abs(phi[:, 0, :, :]), axis=0)  # shape (H, W)
    total_agent = float(np.sum(agent_spatial)) + 1e-12

    bottom_row_share = float(np.sum(agent_spatial[-2:, :])) / total_agent
    side_col_share = float(np.sum(agent_spatial[:, :2]) + np.sum(agent_spatial[:, -2:])) / total_agent
    obstacle_share = float(np.sum(np.abs(phi[:, 3, :, :]))) / (float(np.sum(np.abs(phi))) + 1e-12)

    scores: Dict[str, float] = {
        "Breakout": 0.55 * horiz_ratio + 0.45 * bottom_row_share,
        "Pong": 0.55 * vert_ratio + 0.45 * side_col_share,
        "Freeway": 0.65 * vert_ratio + 0.35 * (1.0 - side_col_share) * (1.0 - horiz_ratio),
        "Pac-Man": (
            0.45 * (1.0 - abs(horiz_ratio - vert_ratio))
            + 0.30 * (1.0 - max(bottom_row_share, side_col_share))
            + 0.25 * min(1.0, obstacle_share * 4.0)
        ),
    }

    # Normalize into confidence distribution via softmax
    names = list(scores.keys())
    raw_vals = np.array([scores[k] for k in names], dtype=np.float64)
    exp_vals = np.exp((raw_vals - np.max(raw_vals)) * 6.0)
    probs = exp_vals / np.sum(exp_vals)
    confidence_map = {k: float(p) for k, p in zip(names, probs)}
    predicted_game = max(confidence_map, key=confidence_map.get)

    return {
        "predicted_game": predicted_game,
        "confidence": confidence_map[predicted_game],
        "probabilities": confidence_map,
        "diagnostics": {
            "horizontal_action_ratio": horiz_ratio,
            "vertical_action_ratio": vert_ratio,
            "bottom_row_agent_concentration": bottom_row_share,
            "side_col_agent_concentration": side_col_share,
            "obstacle_channel_share": obstacle_share,
        },
    }
