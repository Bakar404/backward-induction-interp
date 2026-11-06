"""
Experiment 1 (Stage 1): Zero-Pass Weight-Space Backward Induction.

Demonstrates:
  Part A — Identifying which grid game (Breakout, Pong, Pac-Man, Freeway) a 500->32->32->5
           MLP was built to play from its raw weights alone (0 forward passes), via the
           Backward Stopping Game (Zimmermann et al., Oct 2026 & Takehi et al., Nov 2026 Exp 1).
  Part B — Discovering exact copying (induction) heads and corpus identity (Python, Romaji,
           English) from ONLY the 4 middle transformer blocks with input/output embeddings
           stripped (0 forward passes, Takehi et al., Nov 2026 Exp 3).
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

# Allow running directly from repository root without prior installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_interp.weight_space.stopping_game import infer_grid_game_from_weights
from bi_interp.weight_space.transformer_weights import (
    AttentionHeadWeights,
    TransformerBlockWeights,
    infer_corpus_type_from_middle_weights,
)


def _synthesize_game_policy_weights(game_name: str, seed: int) -> list[np.ndarray]:
    """Construct a 500 -> 32 -> 32 -> 5 MLP with game-consistent spatial receptive fields."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.015, size=(32, 500))
    w2 = np.eye(32) + rng.normal(0.0, 0.02, size=(32, 32))
    w3 = rng.normal(0.0, 0.015, size=(5, 32))

    # Reshape view of W1: (32, 5 channels, 10 rows, 10 cols)
    w1_grid = w1.reshape(32, 5, 10, 10)

    if game_name == "Breakout":
        # Bottom-row paddle (rows 8..9), actions left (idx 3) vs right (idx 4)
        w1_grid[0, 0, 9, :5] = 2.0
        w1_grid[1, 0, 9, 5:] = 2.0
        w3[3, 0] = 2.5
        w3[4, 1] = 2.5
    elif game_name == "Pong":
        # Left/Right side vertical paddle (cols 0 and 9), actions up (idx 1) vs down (idx 2)
        w1_grid[2, 0, :5, 0] = 2.0
        w1_grid[3, 0, 5:, 9] = 2.0
        w3[1, 2] = 2.5
        w3[2, 3] = 2.5
    elif game_name == "Freeway":
        # Vertical lane crossing through center columns (cols 4..5), up (idx 1) vs down (idx 2)
        w1_grid[4, 0, 2:8, 4:6] = 1.8
        w3[1, 4] = 2.8
        w3[2, 4] = -1.5
    elif game_name == "Pac-Man":
        # Full 2D maze coverage across interior + strong obstacle channel (channel 3)
        w1_grid[5, 0, 2:8, 2:8] = 1.4
        w1_grid[6, 3, 2:8, 2:8] = 2.2
        w3[1, 5] = 1.8
        w3[2, 5] = -1.8
        w3[3, 6] = 1.8
        w3[4, 6] = -1.8

    return [w1_grid.reshape(32, 500), w2, w3]


def _synthesize_middle_transformer_blocks(
    corpus: str,
    target_copying_heads: list[tuple[int, int]],
    seed: int,
) -> list[TransformerBlockWeights]:
    """
    Construct 4 intermediate transformer blocks (4 layers x 6 heads, d_model=48, d_head=8)
    with NO input or output embeddings (matching WSP Experiment 3 structure).
    """
    rng = np.random.default_rng(seed)
    d_model, d_head, d_ff = 48, 8, 96
    blocks: list[TransformerBlockWeights] = []

    # Shared subspace used by Layer 0 Head 0 (previous-token head) to write shifted token info
    u_sub = np.eye(d_model, d_head)

    for ell in range(4):
        heads: list[AttentionHeadWeights] = []
        for h in range(6):
            w_q = rng.normal(0.0, 0.08, size=(d_head, d_model))
            w_k = rng.normal(0.0, 0.08, size=(d_head, d_model))
            w_v = rng.normal(0.0, 0.08, size=(d_head, d_model))
            w_o = rng.normal(0.0, 0.08, size=(d_model, d_head))

            if ell == 0 and h == 0:
                # Upstream previous-token head writing strongly into subspace u_sub
                w_v = 1.8 * u_sub.T
                w_o = 1.8 * u_sub
            elif (ell, h) in target_copying_heads:
                # Downstream copying/induction head: K-composes with (0, 0) and has positive OV eigenvalues
                w_q = 2.2 * u_sub.T
                w_k = 2.2 * u_sub.T
                w_v = 2.0 * u_sub.T
                w_o = 2.0 * u_sub

            heads.append(AttentionHeadWeights(w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o))

        w_mlp_in = rng.normal(0.0, 0.1, size=(d_ff, d_model))
        if corpus == "Romaji" and ell == 3:
            # Restricted alphabet (no q, l, v, x) -> final block writes in a low-rank subspace
            basis = u_sub[:, :5]
            coeffs = rng.normal(0.0, 0.3, size=(5, d_ff))
            w_mlp_out = basis @ coeffs
            for h_idx, h_obj in enumerate(heads):
                if (ell, h_idx) not in target_copying_heads:
                    h_obj.w_o = basis @ rng.normal(0.0, 0.15, size=(5, d_head))
        else:
            w_mlp_out = rng.normal(0.0, 0.25, size=(d_model, d_ff))

        blocks.append(
            TransformerBlockWeights(
                layer_idx=ell,
                heads=heads,
                w_mlp_in=w_mlp_in,
                w_mlp_out=w_mlp_out,
            )
        )
    return blocks


def main() -> None:
    print("=" * 78)
    print("STAGE 1A: Zero-Pass Game Identification via Backward Stopping Game (WSP Exp 1)")
    print("=" * 78)
    for idx, game in enumerate(["Breakout", "Pong", "Freeway", "Pac-Man"]):
        weights = _synthesize_game_policy_weights(game, seed=100 + idx)
        res = infer_grid_game_from_weights(weights)
        print(
            f"True Game: {game:<10} | Zero-Pass Prediction: {res['predicted_game']:<10} "
            f"(conf={res['confidence']:.2f}) | "
            f"HorizRatio={res['diagnostics']['horizontal_action_ratio']:.2f}, "
            f"BottomRow={res['diagnostics']['bottom_row_agent_concentration']:.2f}"
        )

    print("\n" + "=" * 78)
    print("STAGE 1B: Stripped Middle-Layer Transformer Copying Heads & Corpus ID (WSP Exp 3)")
    print("=" * 78)
    specs = [
        ("Python", [(2, 1), (3, 1), (3, 5)], 201),
        ("Romaji", [(3, 3)], 202),
        ("English", [(3, 1)], 203),
    ]
    for corpus, gt_heads, seed in specs:
        blocks = _synthesize_middle_transformer_blocks(corpus, gt_heads, seed=seed)
        info = infer_corpus_type_from_middle_weights(blocks, score_threshold=0.20)
        print(
            f"True Corpus: {corpus:<8} | Predicted: {info['predicted_corpus']:<8} | "
            f"Detected Copying Heads: {str(info['detected_copying_heads']):<26} "
            f"(GT: {gt_heads}) | Final Block Effective Rank: {info['final_block_effective_rank']:.2f}"
        )


if __name__ == "__main__":
    main()
