"""Unit tests for Stage 1: Zero-Pass Weight-Space Backward Induction."""

from __future__ import annotations

import numpy as np
import pytest

from bi_interp.weight_space.stopping_game import (
    infer_grid_game_from_weights,
    solve_backward_stopping_game,
)
from bi_interp.weight_space.transformer_weights import (
    AttentionHeadWeights,
    TransformerBlockWeights,
    compute_effective_rank,
    detect_copying_heads_zero_pass,
    infer_corpus_type_from_middle_weights,
)


def test_backward_stopping_game_decomposition_conservation() -> None:
    """Verify that V_pos - V_neg equals the exact multilinear product W^(L) ... W^(1)."""
    rng = np.random.default_rng(123)
    w1 = rng.normal(0.0, 0.5, size=(16, 10))
    w2 = rng.normal(0.0, 0.5, size=(8, 16))
    w3 = rng.normal(0.0, 0.5, size=(4, 8))

    res = solve_backward_stopping_game([w1, w2, w3])
    expected_linear = w3 @ w2 @ w1

    np.testing.assert_allclose(res.net_subgame_value, expected_linear, atol=1e-10)
    assert len(res.layer_neuron_importance) == 2
    assert res.layer_neuron_importance[0].shape == (16,)
    assert res.layer_neuron_importance[1].shape == (8,)


def test_zero_pass_grid_game_identification() -> None:
    """Verify that Breakout and Pong weight structures are distinguished with 0 forward passes."""
    from experiments.exp1_zero_pass_weight_reading import _synthesize_game_policy_weights

    for idx, game in enumerate(["Breakout", "Pong", "Freeway", "Pac-Man"]):
        weights = _synthesize_game_policy_weights(game, seed=50 + idx)
        pred = infer_grid_game_from_weights(weights)
        assert pred["predicted_game"] == game
        assert pred["confidence"] > 0.30


def test_transformer_middle_weights_copying_heads_and_rank() -> None:
    """Verify zero-pass induction head discovery and corpus classification (WSP Exp 3)."""
    from experiments.exp1_zero_pass_weight_reading import _synthesize_middle_transformer_blocks

    py_blocks = _synthesize_middle_transformer_blocks("Python", [(2, 1), (3, 1), (3, 5)], seed=1)
    py_res = infer_corpus_type_from_middle_weights(py_blocks, score_threshold=0.20)
    assert py_res["predicted_corpus"] == "Python"
    assert set(py_res["detected_copying_heads"]) == {(2, 1), (3, 1), (3, 5)}

    rom_blocks = _synthesize_middle_transformer_blocks("Romaji", [(3, 3)], seed=2)
    rom_res = infer_corpus_type_from_middle_weights(rom_blocks, score_threshold=0.20)
    assert rom_res["predicted_corpus"] == "Romaji"
    assert rom_res["detected_copying_heads"] == [(3, 3)]

    eng_blocks = _synthesize_middle_transformer_blocks("English", [(3, 1)], seed=3)
    eng_res = infer_corpus_type_from_middle_weights(eng_blocks, score_threshold=0.20)
    assert eng_res["predicted_corpus"] == "English"
    assert eng_res["detected_copying_heads"] == [(3, 1)]
    assert eng_res["final_block_effective_rank"] > rom_res["final_block_effective_rank"]
