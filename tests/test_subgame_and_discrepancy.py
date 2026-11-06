"""Unit tests for Stage 2: Extensive-Form Subgame Backward Induction & CoT Discrepancy."""

from __future__ import annotations

import numpy as np

from bi_interp.subgame.extensive_form import solve_backward_induction
from bi_interp.subgame.inverse_bi import (
    recover_latent_utilities_across_layers,
    resolve_sycophancy_via_attention_subgame,
)
from bi_interp.discrepancy.cot_vs_decision import compute_decision_reasoning_discrepancy
from experiments.exp2_sycophancy_cot_subgame import build_sycophancy_probing_game


def test_extensive_form_backward_induction_spe() -> None:
    """Verify exact Subgame Perfect Equilibrium switches when latent utility weights change."""
    game = build_sycophancy_probing_game()

    # Honest utility vector: favors factual_accuracy
    eq_honest = solve_backward_induction(game, np.array([1.0, 0.0, 0.0]))
    assert eq_honest.spe_actions["root"] == "attend_question_facts"
    assert eq_honest.spe_actions["node_verify"] == "output_true_answer"

    # Sycophantic utility vector: favors user_sycophancy
    eq_syc = solve_backward_induction(game, np.array([0.0, 1.0, 0.0]))
    assert eq_syc.spe_actions["root"] == "attend_user_hint"
    assert eq_syc.spe_actions["node_rationalize"] == "fabricate_cot_for_user_hint"


def test_inverse_backward_induction_and_cot_discrepancy() -> None:
    """Verify recovery of Decision Lock-In Layer l* and unfaithful CoT detection."""
    game = build_sycophancy_probing_game()
    layer_policies = []
    for ell in range(6):
        w = np.array([0.8, 0.1, 0.1]) if ell < 2 else np.array([0.05, 0.90, 0.05])
        layer_policies.append(solve_backward_induction(game, w, temperature=0.25).qre_policies)

    profile = recover_latent_utilities_across_layers(game, layer_policies, temperature=0.25)
    assert profile.decision_lock_in_layer == 2
    assert profile.final_factor_weights["user_sycophancy"] > 0.60

    disc = compute_decision_reasoning_discrepancy(
        internal_factor_weights=profile.final_factor_weights,
        stated_cot_factor_weights={
            "factual_accuracy": 0.92,
            "user_sycophancy": 0.03,
            "brevity_heuristic": 0.05,
        },
    )
    assert disc.is_unfaithful_cot is True
    assert disc.dominant_internal_factor == "user_sycophancy"
    assert disc.dominant_cot_factor == "factual_accuracy"
    assert disc.js_divergence > 0.35


def test_attention_subgame_sycophancy_resolver() -> None:
    """Verify Head 10.2 detection on WSP Exp 7 style attention routing."""
    rng = np.random.default_rng(9)
    attn = rng.dirichlet(np.ones(12), size=(2, 12, 4, 12))
    attn[:, 10, 2, -1, :] = 0.01
    attn[:, 10, 2, -1, [1, 2]] = 0.02
    attn[:, 10, 2, -1, [8, 9]] = 0.43

    res = resolve_sycophancy_via_attention_subgame(
        attn,
        question_fact_token_indices=[1, 2],
        user_suggestion_token_indices=[8, 9],
    )
    assert res["classification"] == "sycophantic"
    assert res["dominant_routing_head"] == (10, 2)
