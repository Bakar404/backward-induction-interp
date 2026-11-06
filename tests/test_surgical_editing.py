"""Unit tests for Stage 3: Surgical Weight-Space Editing & HuggingFace Hooks."""

from __future__ import annotations

import numpy as np

from bi_interp.editing.surgical_editor import (
    graft_subgame_backup_pathway,
    project_out_mlp_loop_direction,
    suppress_sycophantic_head_weights,
)
from bi_interp.adapters.hf_hooks import apply_surgical_mlp_edit_to_hf_model
from experiments.exp3_surgical_weight_repair import simulate_car_ramp_episode


def test_sparse_rank_one_mlp_loop_projection() -> None:
    """Verify that projecting out a loop direction on top-k columns sharply reduces alignment."""
    rng = np.random.default_rng(11)
    d_model, d_ff = 64, 512
    w_down = rng.normal(0.0, 0.02, size=(d_model, d_ff))
    u_loop = rng.normal(0.0, 1.0, size=d_model)
    u_loop /= np.linalg.norm(u_loop)

    target_cols = np.arange(16)
    w_down[:, target_cols] += 1.2 * u_loop[:, None]

    res = project_out_mlp_loop_direction(w_down, u_loop, top_k_columns=16)
    assert res.num_weights_modified == 64 * 16
    assert np.isclose(res.fraction_modified, 16.0 / 512.0)
    assert res.after_loop_alignment < 0.25 * res.before_loop_alignment


def test_subgame_backup_pathway_grafting_fixes_ramp() -> None:
    """Verify that grafting a backup pathway into a dormant unit solves the ramp task."""
    rng = np.random.default_rng(7)
    w1 = rng.normal(0.0, 0.05, size=(16, 4))
    b1 = np.zeros(16)
    w2 = rng.normal(0.0, 0.05, size=(2, 16))
    b2 = np.zeros(2)
    w1[0, 2] = 1.5
    w2[1, 0] = 2.0
    w1[15, :] = 0.0
    w2[:, 15] = 0.0

    ok_before, _, _ = simulate_car_ramp_episode([w1, w2], [b1, b2])
    assert ok_before is False

    edited_w, edited_b, frac = graft_subgame_backup_pathway(
        [w1, w2], [b1, b2], dormant_neuron_indices=[15]
    )
    ok_after, final_pos, traj = simulate_car_ramp_episode(edited_w, edited_b)
    assert ok_after is True
    assert final_pos >= 1.0
    assert min(traj) <= -0.5
    assert frac < 0.10


def test_hf_adapter_mock_model_edit() -> None:
    """Verify HuggingFace adapter edits mock Qwen/Llama-style model.model.layers[l].mlp.down_proj."""

    class _MockLinear:
        def __init__(self, weight: np.ndarray) -> None:
            self.weight = weight

    class _MockMLP:
        def __init__(self, weight: np.ndarray) -> None:
            self.down_proj = _MockLinear(weight)

    class _MockLayer:
        def __init__(self, weight: np.ndarray) -> None:
            self.mlp = _MockMLP(weight)

    class _MockInnerModel:
        def __init__(self, layers: list[_MockLayer]) -> None:
            self.layers = layers

    class _MockHFModel:
        def __init__(self, layers: list[_MockLayer]) -> None:
            self.model = _MockInnerModel(layers)

    rng = np.random.default_rng(99)
    u = rng.normal(0.0, 1.0, size=32)
    u /= np.linalg.norm(u)
    w = rng.normal(0.0, 0.01, size=(32, 128))
    w[:, :8] += 1.0 * u[:, None]

    mock_model = _MockHFModel([_MockLayer(w.copy()), _MockLayer(w.copy())])
    summary = apply_surgical_mlp_edit_to_hf_model(
        mock_model,
        target_layers=[0, 1],
        loop_directions={0: u, 1: u},
        top_k_columns=8,
    )
    assert summary[0]["after_loop_alignment"] < 0.25 * summary[0]["before_loop_alignment"]
    assert np.allclose(suppress_sycophantic_head_weights(np.ones((4, 4)), 0.0), 0.0)
