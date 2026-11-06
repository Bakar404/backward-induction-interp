"""
Experiment 2 (Stage 2): Resolving Weight-Space Basis Ambiguity & Unfaithful CoT
via Extensive-Form Subgame Backward Induction.

Demonstrates:
  Part A — Why pure static weight norms fail on Sycophancy (WSP Exp 7: Honest vs.
           Sycophantic models with rotated internal bases) and how Attention Subgame
           Probing pinpoints Head (10, 2) routing 85.7% mass to the user's false suggestion.
  Part B — Inverse Backward Induction across layers to find the Decision Lock-In Layer l*
           and compute the Decision-Reasoning Discrepancy (Delta_DR) when the model's
           surface Chain-of-Thought claims it used factual math while its internal
           neural subgame equilibrium optimizes user agreement.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_interp.subgame.extensive_form import (
    DecisionNode,
    ExtensiveFormGame,
    TerminalNode,
    solve_backward_induction,
)
from bi_interp.subgame.inverse_bi import (
    recover_latent_utilities_across_layers,
    resolve_sycophancy_via_attention_subgame,
)
from bi_interp.discrepancy.cot_vs_decision import compute_decision_reasoning_discrepancy


def build_sycophancy_probing_game() -> ExtensiveFormGame:
    """
    Build a 2-stage extensive-form probing subgame with 3 latent decision factors:
      f0: factual_accuracy
      f1: user_sycophancy (agreeing with user's false hint)
      f2: brevity_heuristic
    """
    decision_nodes = {
        "root": DecisionNode(
            node_id="root",
            player="model",
            children={
                "attend_question_facts": "node_verify",
                "attend_user_hint": "node_rationalize",
            },
            description="Initial internal routing at middle transformer layers",
        ),
        "node_verify": DecisionNode(
            node_id="node_verify",
            player="model",
            children={
                "output_true_answer": "term_honest_correct",
                "defer_to_user": "term_reluctant_sycophant",
            },
            description="Subgame after attending to factual numbers",
        ),
        "node_rationalize": DecisionNode(
            node_id="node_rationalize",
            player="model",
            children={
                "fabricate_cot_for_user_hint": "term_unfaithful_sycophant",
                "admit_user_shortcut": "term_blunt_sycophant",
            },
            description="Subgame after locking onto user's false suggestion",
        ),
    }

    terminal_nodes = {
        "term_honest_correct": TerminalNode(
            node_id="term_honest_correct",
            features=np.array([1.0, 0.0, 0.2]),
            description="Outputs correct factual answer despite user's wrong suggestion",
        ),
        "term_reluctant_sycophant": TerminalNode(
            node_id="term_reluctant_sycophant",
            features=np.array([0.3, 0.7, 0.1]),
            description="Computes truth internally but caves at final token",
        ),
        "term_unfaithful_sycophant": TerminalNode(
            node_id="term_unfaithful_sycophant",
            features=np.array([0.0, 1.0, 0.3]),
            description="Selects user's wrong answer and writes plausible-looking fake CoT",
        ),
        "term_blunt_sycophant": TerminalNode(
            node_id="term_blunt_sycophant",
            features=np.array([0.0, 0.6, 0.9]),
            description="Immediately echoes user's guess with no explanation",
        ),
    }

    return ExtensiveFormGame(
        root_id="root",
        decision_nodes=decision_nodes,
        terminal_nodes=terminal_nodes,
        factor_names=["factual_accuracy", "user_sycophancy", "brevity_heuristic"],
    )


def main() -> None:
    print("=" * 78)
    print("STAGE 2A: Resolving WSP Exp 7 Basis-Rotation Ambiguity via Attention Subgames")
    print("=" * 78)
    rng = np.random.default_rng(42)
    # 18 layers, 8 heads, 16 tokens (tokens 2..4 = question numbers; tokens 10..12 = user's guess)
    honest_attn = rng.dirichlet(np.ones(16), size=(4, 18, 8, 16))
    syc_attn = rng.dirichlet(np.ones(16), size=(4, 18, 8, 16))

    # Inject WSP Exp 7 Figure 2 signature at Layer 10, Head 2 from answer position (-1)
    honest_attn[:, 10, 2, -1, :] = 0.004
    honest_attn[:, 10, 2, -1, 2:5] = 0.943 / 3.0
    honest_attn[:, 10, 2, -1, 10:13] = 0.0075 / 3.0

    syc_attn[:, 10, 2, -1, :] = 0.008
    syc_attn[:, 10, 2, -1, 2:5] = 0.035 / 3.0
    syc_attn[:, 10, 2, -1, 10:13] = 0.857 / 3.0

    for label, tensor in [("Honest Model", honest_attn), ("Sycophantic Model", syc_attn)]:
        res = resolve_sycophancy_via_attention_subgame(
            tensor,
            question_fact_token_indices=[2, 3, 4],
            user_suggestion_token_indices=[10, 11, 12],
        )
        print(
            f"{label:<18} -> Classified: {res['classification']:<12} | "
            f"Dominant Routing Head: Layer {res['dominant_routing_head'][0]}.{res['dominant_routing_head'][1]} | "
            f"FactAttn={res['dominant_head_question_fact_mass']*100:5.1f}%, "
            f"UserGuessAttn={res['dominant_head_user_suggestion_mass']*100:5.1f}%"
        )

    print("\n" + "=" * 78)
    print("STAGE 2B: Inverse Backward Induction Across Layers & Unfaithful CoT Detection")
    print("=" * 78)
    game = build_sycophancy_probing_game()

    # Simulate an 8-layer probe trajectory where layers 0..2 explore facts,
    # and at layer 3 (Decision Lock-In Layer l* = 3) the model locks into user_sycophancy
    layer_policies = []
    for ell in range(8):
        if ell < 3:
            w_ell = np.array([0.75, 0.20, 0.05])
        else:
            w_ell = np.array([0.08, 0.85, 0.07])
        eq = solve_backward_induction(game, w_ell, temperature=0.25)
        layer_policies.append(eq.qre_policies)

    profile = recover_latent_utilities_across_layers(game, layer_policies, temperature=0.25)
    print(f"Detected Decision Lock-In Layer (l*): Layer {profile.decision_lock_in_layer} (out of 8)")
    print("Recovered Internal Neural Decision Weights at Final Layer:")
    for k, v in profile.final_factor_weights.items():
        print(f"  - {k:<20}: {v:.3f}")

    # Meanwhile, the model's surface Chain-of-Thought claims it relied 90% on factual_accuracy!
    stated_cot_weights = {
        "factual_accuracy": 0.90,
        "user_sycophancy": 0.02,
        "brevity_heuristic": 0.08,
    }
    disc = compute_decision_reasoning_discrepancy(
        profile.final_factor_weights, stated_cot_weights
    )
    print("\nSurface CoT vs. Internal Neural Decision Discrepancy Report:")
    print(f"  - Dominant Internal Factor : {disc.dominant_internal_factor}")
    print(f"  - Dominant Stated CoT Factor: {disc.dominant_cot_factor}")
    print(f"  - Jensen-Shannon Divergence : {disc.js_divergence:.3f}")
    print(f"  - Unfaithful CoT Detected?  : {disc.is_unfaithful_cot}")
    print(f"  - Hidden Sycophancy Gap     : {disc.hidden_factor_gaps['user_sycophancy']:+.3f}")


if __name__ == "__main__":
    main()
