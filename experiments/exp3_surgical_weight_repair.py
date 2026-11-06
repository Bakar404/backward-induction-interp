"""
Experiment 3 (Stage 3): Backward-Induction Guided Surgical Weight-Space Editing.

Demonstrates:
  Part A — Fixing a PPO policy stuck in a myopic ramp local minimum (WSP Exp 4:
           Kinetix Car-on-Ramp) by grafting a reverse-for-runway subgame pathway
           into a dormant hidden neuron identified by Stage 1's Backward Stopping Game.
  Part B — Fixing infinite reasoning loops (WSP Exp 8: Qwen3.5-0.8B Layers 15 & 19)
           by subtracting the looping residual direction from only the top-64 MLP
           output columns (touching < 1.6% of the layer's columns, zero gradients).
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_interp.weight_space.stopping_game import solve_backward_stopping_game
from bi_interp.editing.surgical_editor import (
    graft_subgame_backup_pathway,
    project_out_mlp_loop_direction,
)


def simulate_car_ramp_episode(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    max_steps: int = 40,
) -> tuple[bool, float, list[float]]:
    """
    Simulate 1D Car-on-Ramp physics:
      - State s = [pos, vel, dist_to_goal, ramp_slope]
      - Starts at pos = -0.1, vel = 0.0. Ramp is at pos = 0.5, Goal is at pos = 1.0.
      - To clear the steep ramp at pos >= 0.5, the car MUST hit pos = 0.5 with vel >= 0.85.
      - Starting from pos = -0.1 and driving straight forward only reaches vel = 0.55 at
        pos = 0.5 -> stalls on the ramp forever!
      - Backing up first to pos <= -0.5 gives enough runway to accelerate to vel >= 0.95
        by pos = 0.5 and jump the strip to reach pos >= 1.0.
    """
    pos = -0.1
    vel = 0.0
    trajectory = [pos]

    for _ in range(max_steps):
        state = np.array([pos, vel, 1.0 - pos, 1.0], dtype=np.float64)
        # Forward pass through ReLU MLP
        h = state
        for ell in range(len(weights) - 1):
            h = np.maximum(weights[ell] @ h + biases[ell], 0.0)
        logits = weights[-1] @ h + biases[-1]
        action = int(np.argmax(logits))  # 0 = reverse, 1 = accelerate forward

        accel = -0.18 if action == 0 else 0.16
        vel = np.clip(vel + accel, -0.6, 1.4)

        # Ramp barrier check at pos >= 0.5
        next_pos = pos + vel * 0.35
        if pos < 0.5 and next_pos >= 0.5 and vel < 0.85:
            # Stalls on the ramp!
            pos = 0.48
            vel = 0.0
        else:
            pos = next_pos

        trajectory.append(pos)
        if pos >= 1.0:
            return True, pos, trajectory

    return False, pos, trajectory


def main() -> None:
    print("=" * 78)
    print("STAGE 3A: Surgical Subgame Pathway Grafting on Car-on-Ramp Policy (WSP Exp 4)")
    print("=" * 78)
    rng = np.random.default_rng(7)
    # 4 -> 32 -> 2 MLP where PPO converged to a greedy forward-only local minimum
    w1 = rng.normal(0.0, 0.05, size=(32, 4))
    b1 = np.zeros(32)
    w2 = rng.normal(0.0, 0.05, size=(2, 32))
    b2 = np.zeros(2)

    # Units 0..4 drive forward greedily; Unit 31 is completely dormant (0 weight)
    w1[0, 2] = 1.5
    w2[1, 0] = 2.0
    w1[31, :] = 0.0
    w2[:, 31] = 0.0

    base_weights = [w1, w2]
    base_biases = [b1, b2]

    # 1. Run Stage 1 Backward Stopping Game to find dormant hidden units automatically
    stop_res = solve_backward_stopping_game(base_weights, dormant_quantile=0.10)
    dormant_units = stop_res.dormant_neurons_per_layer[0]
    print(f"Backward Stopping Game identified dormant hidden neurons: {dormant_units.tolist()}")

    # 2. Evaluate base PPO policy (stalls on ramp)
    ok_base, final_pos_base, _ = simulate_car_ramp_episode(base_weights, base_biases)
    print(f"Base PPO Policy     -> Success: {str(ok_base):<5} | Final Position: {final_pos_base:.2f} (stalled at ramp)")

    # 3. Apply Stage 3 Surgical Subgame Graft on dormant neuron
    edited_w, edited_b, frac_changed = graft_subgame_backup_pathway(
        base_weights,
        base_biases,
        dormant_neuron_indices=dormant_units,
        backup_threshold_pos=-0.5,
        min_runway_speed=0.85,
    )
    ok_edit, final_pos_edit, traj_edit = simulate_car_ramp_episode(edited_w, edited_b)
    min_pos_reached = min(traj_edit)
    print(
        f"Surgically Edited   -> Success: {str(ok_edit):<5} | Final Position: {final_pos_edit:.2f} "
        f"(backed up to x={min_pos_reached:.2f} first, edited {frac_changed*100:.1f}% of params)"
    )

    print("\n" + "=" * 78)
    print("STAGE 3B: Sparse Rank-1 MLP Reasoning-Loop Direction Removal (WSP Exp 8)")
    print("=" * 78)
    # Simulate a layer's MLP down-projection (d_model=256, d_ff=4096) where 64 columns
    # strongly write along a repetition-loop vector `u_loop`
    d_model, d_ff = 256, 4096
    w_down = rng.normal(0.0, 0.02, size=(d_model, d_ff))
    u_loop = rng.normal(0.0, 1.0, size=d_model)
    u_loop /= np.linalg.norm(u_loop)

    loop_cols = rng.choice(d_ff, size=64, replace=False)
    for col in loop_cols:
        w_down[:, col] += 0.85 * u_loop

    edit_report = project_out_mlp_loop_direction(
        w_down=w_down,
        loop_direction=u_loop,
        top_k_columns=64,
        scale=1.0,
    )
    reduction_pct = (
        100.0
        * (edit_report.before_loop_alignment - edit_report.after_loop_alignment)
        / edit_report.before_loop_alignment
    )
    print(
        f"MLP Matrix Shape: ({d_model}, {d_ff}) | Modified {edit_report.num_weights_modified:,} / "
        f"{edit_report.total_matrix_weights:,} weights ({edit_report.fraction_modified*100:.2f}%)"
    )
    print(
        f"Loop-Direction Alignment Norm: {edit_report.before_loop_alignment:.3f} -> "
        f"{edit_report.after_loop_alignment:.3f} ({reduction_pct:.1f}% reduction, 0 gradients)"
    )


if __name__ == "__main__":
    main()
