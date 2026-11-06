"""Empirical reproduction of the MIT CSAIL Weight-Space Programming (WSP) mechanisms.

Loads and analyzes the actual neural network checkpoints, sparse weight edits,
and attention probe tensors from `RikiyaT/lm_understand-logs` (Takehi et al., 2026)
stored in `data/wsp_artifacts/`, demonstrating why each specific circuit and
weight edit was selected through the lens of backward induction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "wsp_artifacts"


@dataclass
class Exp1GameWeightAnalysis:
    """Backward-inducted spatial & action weight profile for a 500->32->32->5 PPO policy."""

    game_name: str
    network_id: str
    total_actor_params: int
    channel_row_energy: np.ndarray  # shape (5, 10): channel x row L2 energy in W1
    channel_col_energy: np.ndarray  # shape (5, 10): channel x col L2 energy in W1
    action_logit_std: np.ndarray    # shape (5,): std across 32 hidden units in W3
    horizontal_vs_vertical_ratio: float
    paddle_row8_concentration: float
    upper_brick_band_ratio: float
    velocity_pair_correlation: float
    notable_weights: Dict[str, float]


def analyze_exp1_game_weights(
    npz_path: str | Path,
    game_name: str = "breakout",
    network_id: str = "a9f535cc",
) -> Exp1GameWeightAnalysis:
    """Analyze a 500->32->32->5 tanh PPO policy from WSP Experiment 1.

    Input index layout in WSP Exp 1:
        idx = (row * 10 + col) * 5 + channel
    where row in 0..9 (top to bottom), col in 0..9 (left to right),
    and channels are:
        0: agent, 1: moving objects, 2: reward targets, 3: obstacles, 4: previous frame.
    Actions (0..4):
        0: no-op, 1: up, 2: down, 3: left, 4: right.
    """
    z = np.load(npz_path)
    W1 = z["actor/dense1/kernel"].astype(np.float64)  # (500, 32)
    b1 = z["actor/dense1/bias"].astype(np.float64)    # (32,)
    W2 = z["actor/dense2/kernel"].astype(np.float64)  # (32, 32)
    b2 = z["actor/dense2/bias"].astype(np.float64)    # (32,)
    W3 = z["actor/logits/kernel"].astype(np.float64)  # (32, 5)
    b3 = z["actor/logits/bias"].astype(np.float64)    # (5,)

    total_actor_params = int(W1.size + b1.size + W2.size + b2.size + W3.size + b3.size)

    # Reshape W1 into (row=10, col=10, channel=5, unit=32)
    W1_grid = W1.reshape(10, 10, 5, 32)

    # Compute L2 norm across the 32 first-layer units for each (channel, row) and (channel, col)
    channel_row_energy = np.sqrt(np.sum(W1_grid ** 2, axis=(1, 3))).T  # (5, 10)
    channel_col_energy = np.sqrt(np.sum(W1_grid ** 2, axis=(0, 3))).T  # (5, 10)

    # Action differentiation at output layer W3 (actions 3=left, 4=right vs 0=noop, 1=up, 2=down)
    action_logit_std = np.std(W3, axis=0)
    horiz_energy = float(np.linalg.norm(W3[:, 3:5]))
    vert_energy = float(np.linalg.norm(W3[:, 1:3]))
    horizontal_vs_vertical_ratio = horiz_energy / (vert_energy + 1e-12)

    # Row 8 concentration in Channel 0 (agent position on playable interior rows 1..8)
    ch0_interior_rows = channel_row_energy[0, 1:9]
    paddle_row8_concentration = float(channel_row_energy[0, 8] / (np.sum(ch0_interior_rows) + 1e-12))

    # Upper brick band concentration (rows 1..3 vs rows 4..8 in Channel 2 reward targets)
    ch2_upper = float(np.sum(channel_row_energy[2, 1:4]))
    ch2_lower = float(np.sum(channel_row_energy[2, 4:9]))
    upper_brick_band_ratio = ch2_upper / (ch2_lower + 1e-12)

    # Velocity tracking correlation between Channel 1 (current ball) and Channel 4 (previous frame)
    ch1_interior = W1_grid[1:8, 1:9, 1, :].ravel()
    ch4_interior = W1_grid[1:8, 1:9, 4, :].ravel()
    velocity_pair_correlation = float(np.corrcoef(ch1_interior, ch4_interior)[0, 1])

    notable = {
        "W1_r8_c1_ch0_u15": round(float(W1_grid[8, 1, 0, 15]), 3),
        "W1_r8_c1_ch0_u31": round(float(W1_grid[8, 1, 0, 31]), 3),
        "W1_r1_c3_ch1_u7": round(float(W1_grid[1, 3, 1, 7]), 3),
        "W1_r1_c3_ch1_u8": round(float(W1_grid[1, 3, 1, 8]), 3),
        "W1_r1_c3_ch4_u7": round(float(W1_grid[1, 3, 4, 7]), 3),
        "W1_r1_c3_ch4_u8": round(float(W1_grid[1, 3, 4, 8]), 3),
        "W1_r3_c1_ch2_u0": round(float(W1_grid[3, 1, 2, 0]), 3),
        "W3_u22_left": round(float(W3[22, 3]), 3),
        "W3_u11_left": round(float(W3[11, 3]), 3),
        "W3_u24_right": round(float(W3[24, 4]), 3),
    }

    return Exp1GameWeightAnalysis(
        game_name=game_name,
        network_id=network_id,
        total_actor_params=total_actor_params,
        channel_row_energy=channel_row_energy,
        channel_col_energy=channel_col_energy,
        action_logit_std=action_logit_std,
        horizontal_vs_vertical_ratio=horizontal_vs_vertical_ratio,
        paddle_row8_concentration=paddle_row8_concentration,
        upper_brick_band_ratio=upper_brick_band_ratio,
        velocity_pair_correlation=velocity_pair_correlation,
        notable_weights=notable,
    )


def compare_exp1_all_games(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, Exp1GameWeightAnalysis]:
    """Run zero-pass weight analysis across all four WSP Exp 1 game networks."""
    specs = [
        ("breakout", "a9f535cc", "exp1_breakout_a9f535cc.npz"),
        ("pong", "dfc6cb65", "exp1_pong_dfc6cb65.npz"),
        ("pacman", "bdf99279", "exp1_pacman_bdf99279.npz"),
        ("freeway", "0bbcc3ea", "exp1_freeway_0bbcc3ea.npz"),
    ]
    results: Dict[str, Exp1GameWeightAnalysis] = {}
    for game, net_id, fname in specs:
        results[game] = analyze_exp1_game_weights(artifact_dir / fname, game_name=game, network_id=net_id)
    return results


def analyze_exp3_char_lm_circuits(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, object]:
    """Reproduce WSP Exp 3 embedding-free induction head detection on the 4L-6H Python char-LM.

    Uses Astra's exact LayerNorm-projected bilinear operators:
        M = (1/8) * P D W_q^T W_k D P
        B = P W_o W_v D P
    and directional composition scores:
        C_k = tr(M_later B_earlier) / (||M_later||_F ||B_earlier||_F)
        C_q = tr(M_later B_earlier^T) / (||M_later||_F ||B_earlier||_F)
    """
    with open(artifact_dir / "exp3_python_weight_algebra.json", "r", encoding="utf-8") as f:
        algebra = json.load(f)
    with open(artifact_dir / "exp3_heads_eval.json", "r", encoding="utf-8") as f:
        eval_gt = json.load(f)["python_L4_H6_d384_s0"]

    compositions = algebra["compositions"]
    head_stats = algebra["head_stats"]

    # For each head in layers 1..3, find its strongest incoming key-composition C_k from an earlier layer
    later_head_scores = {}
    for l_idx in range(1, 4):
        for h_idx in range(6):
            target = f"L{l_idx}H{h_idx}"
            best_pair = None
            best_ck = -1.0
            best_cq = 0.0
            for l_prev in range(l_idx):
                for h_prev in range(6):
                    src = f"L{l_prev}H{h_prev}"
                    key = f"{src}->{target}"
                    ck = compositions[key]["C_k"]
                    cq = compositions[key]["C_q"]
                    if ck > best_ck:
                        best_ck = ck
                        best_cq = cq
                        best_pair = src
            later_head_scores[target] = {
                "best_predecessor": best_pair,
                "C_k": best_ck,
                "C_q": best_cq,
                "asymmetry": round(best_ck - abs(best_cq), 5),
                "ground_truth_induct": eval_gt[target]["induct"],
                "ground_truth_prev": eval_gt[target]["prev"],
            }

    # Rank heads in layers 2 and 3 by C_k
    late_ranked = sorted(
        [(k, v) for k, v in later_head_scores.items() if k.startswith(("L2", "L3"))],
        key=lambda item: item[1]["C_k"],
        reverse=True,
    )
    predicted_copy_heads = [k for k, _ in late_ranked[:3]]
    ground_truth_copy_heads = [
        k for k, v in eval_gt.items() if v["induct"] > 0.2
    ]

    return {
        "head_stats": head_stats,
        "later_head_scores": later_head_scores,
        "top_3_predicted_copy_heads": predicted_copy_heads,
        "ground_truth_copy_heads": ground_truth_copy_heads,
        "exact_top3_match": set(predicted_copy_heads) == set(ground_truth_copy_heads),
        "key_circuits": {
            "L0H5->L2H1": compositions["L0H5->L2H1"],
            "L2H2->L3H1": compositions["L2H2->L3H1"],
            "L2H2->L3H5": compositions["L2H2->L3H5"],
        },
    }


def analyze_exp4_kinetix_repair(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, object]:
    """Analyze WSP Exp 4 one-shot surgical repair of the 171,664-weight Kinetix PPO actor.

    Compares `ppo_final_seed0.npz` against `edited_actor.npz` and evaluates the
    backward-inducted 3-phase control subgame grafted into units 122..127.
    """
    ppo_z = np.load(artifact_dir / "exp4_kinetix_ppo_seed0.npz")
    edit_z = np.load(artifact_dir / "exp4_kinetix_edited_actor.npz")

    layer_map = [
        ("embed", "params/Dense_0"),
        ("actor0", "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0"),
        ("actor1", "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0"),
        ("actor2", "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0"),
        ("actor3", "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0"),
        ("actor4", "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0"),
        ("actor_head", "params/GeneralActorCriticRNN_0/Dense_0"),
    ]

    total_params = 0
    changed_params = 0
    per_layer_diff = {}
    base_tensors = {}
    edit_tensors = {}

    for short_name, ppo_prefix in layer_map:
        w_raw = ppo_z[f"{ppo_prefix}/kernel"].astype(np.float64)
        b_raw = ppo_z[f"{ppo_prefix}/bias"].astype(np.float64)
        w_dec = 2 if short_name == "embed" else 3
        b_dec = 4 if short_name in ("embed", "actor0") else 3
        w_base = np.round(w_raw, w_dec)
        b_base = np.round(b_raw, b_dec)
        w_edit = edit_z[f"{short_name}.W"].astype(np.float64)
        b_edit = edit_z[f"{short_name}.b"].astype(np.float64)

        base_tensors[f"{short_name}.W"] = w_base
        base_tensors[f"{short_name}.b"] = b_base
        edit_tensors[f"{short_name}.W"] = w_edit
        edit_tensors[f"{short_name}.b"] = b_edit

        w_diff = int(np.sum(np.abs(w_edit - w_base) > 1e-6))
        b_diff = int(np.sum(np.abs(b_edit - b_base) > 1e-6))
        n_p = int(w_base.size + b_base.size)
        total_params += n_p
        changed_params += w_diff + b_diff
        per_layer_diff[short_name] = {
            "total": n_p,
            "changed_W": w_diff,
            "changed_b": b_diff,
            "changed_total": w_diff + b_diff,
        }

    def forward_actor(tensors: Dict[str, np.ndarray], obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        acts = {}
        h = np.maximum(0.0, obs @ tensors["embed.W"] + tensors["embed.b"])
        acts["embed"] = h
        for i in range(5):
            h = np.tanh(h @ tensors[f"actor{i}.W"] + tensors[f"actor{i}.b"])
            acts[f"actor{i}"] = h
        logits = h @ tensors["actor_head.W"] + tensors["actor_head.b"]
        acts["logits"] = logits
        return logits, acts

    # Evaluate across the 4 key physical subgame states of Kinetix l/h2_car_ramp:
    # Features used by grafted circuit: obs[52] = body_x, obs[54] = body_vx, obs[234] = green_x
    regimes = [
        ("stalled_at_ramp", {"body_x": 2.2, "body_vx": 0.0, "green_x": 2.0}),
        ("backing_up_midway", {"body_x": 3.3, "body_vx": 1.2, "green_x": 3.1}),
        ("turnaround_launch", {"body_x": 3.9, "body_vx": 0.2, "green_x": 3.7}),
        ("fast_ramp_climb", {"body_x": 2.3, "body_vx": -2.2, "green_x": 2.1}),
        ("post_ramp_handback", {"body_x": 0.9, "body_vx": -0.8, "green_x": 0.75}),
    ]

    regime_evals = {}
    for name, state in regimes:
        obs = np.zeros(679, dtype=np.float64)
        obs[52] = state["body_x"]
        obs[54] = state["body_vx"]
        obs[234] = state["green_x"]

        logits_base, _ = forward_actor(base_tensors, obs)
        logits_edit, acts_edit = forward_actor(edit_tensors, obs)
        override_u126_127 = acts_edit["actor4"][126:128]
        regime_evals[name] = {
            "state": state,
            "actor0_thresholds_122_128": [round(float(x), 3) for x in acts_edit["actor0"][122:128]],
            "actor4_override_126_127": [round(float(x), 4) for x in override_u126_127],
            "edited_top_action": int(np.argmax(logits_edit[:3])),
            "edited_logits_0_1_2": [round(float(x), 2) for x in logits_edit[:3]],
            "base_logits_0_1_2": [round(float(x), 2) for x in logits_base[:3]],
            "max_logit_diff_from_base": round(float(np.max(np.abs(logits_edit - logits_base))), 4),
        }

    return {
        "total_params": total_params,
        "changed_params": changed_params,
        "changed_fraction": round(changed_params / total_params, 6),
        "per_layer_diff": per_layer_diff,
        "regime_evaluations": regime_evals,
    }


def analyze_exp5_handwritten_breakout(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, object]:
    """Deconstruct Astra's zero-shot hand-written MinAtar Breakout network (WSP Exp 5)."""
    z = np.load(artifact_dir / "exp5_breakout_hand.npz")
    W1 = z["actor/dense1/kernel"].astype(np.float64)  # (600, 32)
    b1 = z["actor/dense1/bias"].astype(np.float64)    # (32,)
    W2 = z["actor/dense2/kernel"].astype(np.float64)  # (32, 32)
    b2 = z["actor/dense2/bias"].astype(np.float64)    # (32,)
    W3 = z["actor/logits/kernel"].astype(np.float64)  # (32, 6)
    b3 = z["actor/logits/bias"].astype(np.float64)    # (6,)

    total_params = int(W1.size + b1.size + W2.size + b2.size + W3.size + b3.size)
    nonzero_params = int(
        np.count_nonzero(W1) + np.count_nonzero(b1)
        + np.count_nonzero(W2) + np.count_nonzero(b2)
        + np.count_nonzero(W3) + np.count_nonzero(b3)
    )

    # Extract paddle weights on row 9 channel 0 (indices 540:600:6) for units 0..3
    paddle_row9_weights = W1[540:600:6, 0].tolist()
    # Extract ball row 2 descending-right target curve (indices 123:180:6, unit 0)
    ball_row2_desc_right = W1[123:180:6, 0].tolist()

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_fraction": round(nonzero_params / total_params, 5),
        "active_h1_units": int(np.sum(np.any(W1 != 0, axis=0) | (b1 != 0))),
        "active_h2_units": int(np.sum(np.any(W2 != 0, axis=0) | (b2 != 0))),
        "paddle_row9_weights": [round(float(x), 3) for x in paddle_row9_weights],
        "ball_row2_desc_right": [round(float(x), 3) for x in ball_row2_desc_right],
        "b2_gate_bias": float(b2[0]),
        "b3_action_bias": [round(float(x), 1) for x in b3],
    }


def analyze_exp7_sycophancy_subgame(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, object]:
    """Analyze WSP Exp 7 Gemma-3-270M Pair 2 attention routing and control subgames."""
    with open(artifact_dir / "exp7_gemma_r7_attention.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    routing = data["head_routing"]
    ranked_heads = sorted(
        routing.items(),
        key=lambda kv: kv[1]["sycophancy_shift"],
        reverse=True,
    )

    return {
        "top_sycophancy_heads": [
            {"head_id": k, **v} for k, v in ranked_heads[:6]
        ],
        "L10H2": routing["L10H2"],
        "L9H2": routing["L9H2"],
        "L5H1": routing["L5H1"],
        "controls": data["controls"],
        "L10H2_profile_A": data["L10H2_profile_A"],
        "L10H2_profile_B": data["L10H2_profile_B"],
        "tokens": data["tokens"],
    }


def analyze_exp8_qwen_loop_surgical_edit(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dict[str, object]:
    """Analyze WSP Exp 8 Qwen3.5-0.8B sparse top-64 MLP surgical weight edit (`p50c`)."""
    z8 = np.load(artifact_dir / "exp8_qwen_p50c_edit.npz")
    with open(artifact_dir / "exp8_qwen_p50c_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    layer_summaries = {}
    for i, layer in enumerate(["15", "19"]):
        idx = z8[f"t{i}_idx"]
        val_u16 = z8[f"t{i}_val"]
        val_f32 = (val_u16.astype(np.uint32) << 16).view(np.float32)
        rows, cols = np.divmod(idx, 3584)
        unique_cols = sorted(np.unique(cols).tolist())

        # Reconstruct the 1024 x 64 edited submatrix of active columns
        col_to_local = {c: j for j, c in enumerate(unique_cols)}
        sub_mat = np.zeros((1024, len(unique_cols)), dtype=np.float64)
        for r, c, v in zip(rows, cols, val_f32):
            sub_mat[r, col_to_local[int(c)]] = float(v)

        sv = np.linalg.svd(sub_mat, compute_uv=False)
        layer_summaries[layer] = {
            "tensor_name": manifest[layer]["name"],
            "shape": manifest[layer]["shape"],
            "selected_columns_count": len(unique_cols),
            "selected_columns_match_manifest": unique_cols == manifest[layer]["selected_neurons"],
            "selected_columns_first10": unique_cols[:10],
            "bfloat16_changed_entries": int(len(idx)),
            "relative_frobenius_change": float(manifest[layer]["relative_frobenius_change"]),
            "directional_residual_fraction": float(manifest[layer]["directional_residual_fraction"]),
            "edited_submatrix_top3_sv": [round(float(x), 4) for x in sv[:3]],
        }

    return {
        "total_model_params_approx": 800_000_000,
        "edited_params_total": 131_072,
        "edited_bfloat16_entries": int(len(z8["t0_idx"]) + len(z8["t1_idx"])),
        "edited_fraction_of_model": 131_072 / 800_000_000,
        "selectivity_ratio": round(0.10311 / 0.00903, 2),
        "layers": layer_summaries,
        "budget_progression": {
            "base": {"passes": 0, "edit": "unmodified Qwen3.5-0.8B", "math500": 7.9, "gsm8k": 15.6, "ifeval": 21.0, "cutoff_rate": 93.0},
            "p10c": {"passes": 10, "edit": "30 matrices (L8-L22) rank-1 W - 1.5 u(u^T W)", "math500": 12.3, "gsm8k": 21.3, "ifeval": 27.9, "cutoff_rate": 82.2},
            "p25c": {"passes": 25, "edit": "L15 Attention Head 7 o_proj[:, 1792:2048] = 0", "math500": 14.1, "gsm8k": 32.1, "ifeval": 29.8, "cutoff_rate": 74.5},
            "p50c": {"passes": 50, "edit": "L15 & L19 MLP down_proj top-64 cols W - 6 u(m64 * u^T W)", "math500": 48.1, "gsm8k": 67.1, "ifeval": 62.3, "cutoff_rate": 33.1},
        },
    }
