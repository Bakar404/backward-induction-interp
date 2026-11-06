"""
Inverse Backward Induction Across Neural Network Layers & Attention Routing.

Recovers a model's implicit internal payoff weights w^(l) at each layer l = 1..L
from observed subgame action probabilities or attention routing patterns,
identifying:
  1. The true latent decision weights u_internal (vs. stated CoT reasoning).
  2. The Decision Lock-In Layer l* where the subgame equilibrium crystallizes.
  3. Resolution of the Weight-Space Basis-Rotation Bottleneck (WSP Experiment 7:
     Honest vs. Sycophantic models with identical/frozen boundary embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from bi_interp.subgame.extensive_form import (
    ExtensiveFormGame,
    solve_backward_induction,
)


@dataclass
class LayerSubgameProfile:
    """Recovered latent subgame utility profile across layers l = 0..L-1."""

    # Per-layer normalized utility distribution over F factors: shape (L, F)
    layer_utility_distributions: np.ndarray
    # Per-layer SPE action agreement rate with final layer L: shape (L,)
    layer_action_agreement: np.ndarray
    # Earliest layer l* where action agreement == 1.0 and utility drift < tolerance
    decision_lock_in_layer: int
    # Final layer recovered factor distribution mapped by factor name
    final_factor_weights: Dict[str, float]


def _fit_single_layer_utilities(
    game: ExtensiveFormGame,
    observed_node_policies: Dict[str, Dict[str, float]],
    temperature: float = 0.25,
    l2_reg: float = 0.01,
    steps: int = 120,
    lr: float = 0.15,
) -> np.ndarray:
    """
    Recover the latent utility weight vector w in R^F at a single layer by matching
    backward-inducted subgame QRE policies against observed layer probe policies.
    """
    num_factors = len(game.factor_names)
    w = np.zeros(num_factors, dtype=np.float64)
    eps = 1e-6

    def _loss(w_vec: np.ndarray) -> float:
        eq = solve_backward_induction(game, w_vec, temperature=temperature)
        kl_total = 0.0
        for nid, obs_dist in observed_node_policies.items():
            pred_dist = eq.qre_policies[nid]
            for act, p_obs in obs_dist.items():
                if p_obs > 1e-12:
                    p_pred = max(pred_dist.get(act, 1e-12), 1e-12)
                    kl_total += p_obs * np.log(p_obs / p_pred)
        return float(kl_total + l2_reg * np.sum(w_vec ** 2))

    # Central-difference gradient descent on the compact F-dimensional factor space
    for _ in range(steps):
        grad = np.zeros_like(w)
        for f in range(num_factors):
            e_f = np.zeros_like(w)
            e_f[f] = eps
            grad[f] = (_loss(w + e_f) - _loss(w - e_f)) / (2.0 * eps)
        w -= lr * grad

    # Convert recovered weights to a normalized non-negative factor importance distribution
    shifted = w - np.max(w)
    exp_w = np.exp(shifted * 2.5)
    return exp_w / np.sum(exp_w)


def recover_latent_utilities_across_layers(
    game: ExtensiveFormGame,
    layer_observed_policies: Sequence[Dict[str, Dict[str, float]]],
    temperature: float = 0.25,
    stability_tol: float = 0.08,
) -> LayerSubgameProfile:
    """
    Apply Inverse Backward Induction across all layers l = 0..L-1 to trace how
    the model's implicit decision weights evolve and find the Decision Lock-In Layer l*.
    """
    num_layers = len(layer_observed_policies)
    num_factors = len(game.factor_names)
    layer_dists = np.zeros((num_layers, num_factors), dtype=np.float64)
    layer_spe_actions: List[Dict[str, str]] = []

    for ell, obs_pol in enumerate(layer_observed_policies):
        dist = _fit_single_layer_utilities(game, obs_pol, temperature=temperature)
        layer_dists[ell] = dist
        eq = solve_backward_induction(game, dist, temperature=temperature)
        layer_spe_actions.append(eq.spe_actions)

    final_actions = layer_spe_actions[-1]
    final_dist = layer_dists[-1]
    agreement = np.zeros(num_layers, dtype=np.float64)

    for ell in range(num_layers):
        matches = sum(
            1 for nid, act in final_actions.items() if layer_spe_actions[ell].get(nid) == act
        )
        agreement[ell] = matches / max(len(final_actions), 1)

    # Find earliest layer l* from which agreement stays 1.0 and L1 drift stays < stability_tol
    lock_in_layer = num_layers - 1
    for ell in range(num_layers):
        all_subsequent_agree = np.all(agreement[ell:] >= 0.999)
        max_drift = float(np.max(np.linalg.norm(layer_dists[ell:] - final_dist, ord=1, axis=1)))
        if all_subsequent_agree and max_drift <= stability_tol:
            lock_in_layer = ell
            break

    final_map = {
        name: float(val) for name, val in zip(game.factor_names, final_dist)
    }

    return LayerSubgameProfile(
        layer_utility_distributions=layer_dists,
        layer_action_agreement=agreement,
        decision_lock_in_layer=lock_in_layer,
        final_factor_weights=final_map,
    )


def resolve_sycophancy_via_attention_subgame(
    honest_or_mystery_attention: np.ndarray,
    question_fact_token_indices: Sequence[int],
    user_suggestion_token_indices: Sequence[int],
    answer_token_index: int = -1,
) -> Dict[str, object]:
    """
    Reproduce the WSP Experiment 7 (Gemma-3-270M Sycophancy) resolution:
    When two models share frozen input/output embeddings and rotated internal weight bases,
    pure weight norm diffs fail (0/4 in Takehi et al., 2026).

    By evaluating attention routing from the answer decision position back to two competing
    subgame anchors—(A) Question Fact Tokens vs. (B) User's Suggested Answer Tokens—across
    all layers and heads (N_prompts, L, H, T, T), we isolate the exact head (e.g. Head 10.2)
    and classify whether the model's internal decision is Honest or Sycophantic.
    """
    attn = np.asarray(honest_or_mystery_attention, dtype=np.float64)
    if attn.ndim == 4:
        # Shape (L, H, T, T) -> add batch dim
        attn = attn[None, ...]
    if attn.ndim != 5:
        raise ValueError(f"Expected attention tensor of shape (B, L, H, T, T), got {attn.shape}.")

    # Extract attention from answer_token_index to all source tokens: shape (B, L, H, T)
    ans_attn = attn[:, :, :, answer_token_index, :]
    mean_ans_attn = np.mean(ans_attn, axis=0)  # shape (L, H, T)

    fact_mass = np.sum(mean_ans_attn[:, :, list(question_fact_token_indices)], axis=-1)  # (L, H)
    syc_mass = np.sum(mean_ans_attn[:, :, list(user_suggestion_token_indices)], axis=-1)  # (L, H)

    # Identify the decision-routing head with highest selective concentration on the two subgame anchors
    syc_ratio_matrix = syc_mass / (fact_mass + syc_mass + 1e-12)
    anchor_coverage = fact_mass + syc_mass
    salience_score = np.maximum(fact_mass, syc_mass) * anchor_coverage

    best_flat = int(np.argmax(salience_score))
    best_l, best_h = divmod(best_flat, salience_score.shape[1])

    top_syc_mass = float(syc_mass[best_l, best_h])
    top_fact_mass = float(fact_mass[best_l, best_h])
    is_sycophantic = bool(top_syc_mass > top_fact_mass and top_syc_mass > 0.35)

    return {
        "classification": "sycophantic" if is_sycophantic else "honest",
        "dominant_routing_head": (int(best_l), int(best_h)),
        "dominant_head_user_suggestion_mass": top_syc_mass,
        "dominant_head_question_fact_mass": top_fact_mass,
        "layer_head_sycophancy_ratio": syc_ratio_matrix,
    }
