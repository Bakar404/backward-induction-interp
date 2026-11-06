"""Experiment 4: Empirical Reproduction of MIT CSAIL Weight-Space Programming (WSP).

Runs backward-induction diagnostics directly on the released `.npz` and `.json`
artifacts from `RikiyaT/lm_understand-logs` (Takehi et al., 2026):
  1. Exp 1 (`a9f535cc`, `dfc6cb65`, `bdf99279`, `0bbcc3ea`): Zero-pass game identification
  2. Exp 3 (`python_L4_H6_d384_s0`): Embedding-free QK/OV induction head algebra
  3. Exp 4 (`ppo_final_seed0` -> `edited_actor`): 51-statement Kinetix unit 122:128 subgame repair
  4. Exp 5 (`breakout/hand.npz`): Zero-shot 815-weight hand-written Breakout policy
  5. Exp 7 & 8 (`gemma-3-270m-it` Head 10.2 & `Qwen3.5-0.8B` Layers 15/19 top-64 MLP columns)
"""

from bi_interp.wsp_empirical import (
    compare_exp1_all_games,
    analyze_exp3_char_lm_circuits,
    analyze_exp4_kinetix_repair,
    analyze_exp5_handwritten_breakout,
    analyze_exp7_sycophancy_subgame,
    analyze_exp8_qwen_loop_surgical_edit,
)


def main() -> None:
    print("WSP Empirical Reproduction on Released Checkpoints (RikiyaT/lm_understand-logs)")
    print("-" * 80)

    # 1. Exp 1
    games = compare_exp1_all_games()
    b = games["breakout"]
    print(f"[Exp 1] Breakout ({b.network_id}.npz, {b.total_actor_params:,} actor params):")
    print(f"  Row-8 paddle energy share (ch0): {b.paddle_row8_concentration:.3f}")
    print(f"  Upper-brick vs lower ratio (ch2): {b.upper_brick_band_ratio:.3f}")
    print(f"  Horizontal vs vertical W3 ratio: {b.horizontal_vs_vertical_ratio:.3f}")
    print(f"  Cited weights: W1(8,1,ch0)->u15={b.notable_weights['W1_r8_c1_ch0_u15']}, "
          f"u31={b.notable_weights['W1_r8_c1_ch0_u31']}; "
          f"W3[22,left]={b.notable_weights['W3_u22_left']}")

    # 2. Exp 3
    exp3 = analyze_exp3_char_lm_circuits()
    print("\n[Exp 3] 4L-6H Character Transformer (embeddings stripped):")
    for pair, scores in exp3["key_circuits"].items():
        print(f"  {pair}: C_k = {scores['C_k']:.4f}, C_q = {scores['C_q']:.4f}")
    print(f"  Predicted copy heads: {exp3['top_3_predicted_copy_heads']} "
          f"(exact match with ground truth: {exp3['exact_top3_match']})")

    # 3. Exp 4
    exp4 = analyze_exp4_kinetix_repair()
    print(f"\n[Exp 4] Kinetix Car-Ramp Surgical Edit ({exp4['changed_params']:,} / {exp4['total_params']:,} "
          f"weights = {100*exp4['changed_fraction']:.2f}%):")
    for reg_name, info in exp4["regime_evaluations"].items():
        print(f"  {reg_name:22s}: u126_127={info['actor4_override_126_127']} "
              f"-> top_action={info['edited_top_action']} "
              f"(logits 0..2={info['edited_logits_0_1_2']})")

    # 4. Exp 5
    exp5 = analyze_exp5_handwritten_breakout()
    print(f"\n[Exp 5] Zero-Shot Hand-Written MinAtar Breakout ({exp5['nonzero_params']} / "
          f"{exp5['total_params']:,} nonzero = {100*exp5['sparsity_fraction']:.2f}%):")
    print(f"  Active hidden units: h1={exp5['active_h1_units']}/32, h2={exp5['active_h2_units']}/32, "
          f"gate bias b2={exp5['b2_gate_bias']}")

    # 5. Exp 7 & 8
    exp7 = analyze_exp7_sycophancy_subgame()
    h10_2 = exp7["L10H2"]
    print(f"\n[Exp 7] Gemma-3-270M Sycophancy Subgame (Head 10.2):")
    print(f"  Operands attention: Honest(A)={100*h10_2['A_operands']:.2f}% vs Sycophant(B)={100*h10_2['B_operands']:.2f}%")
    print(f"  User guess attention: Honest(A)={100*h10_2['A_guess']:.2f}% vs Sycophant(B)={100*h10_2['B_guess']:.2f}%")
    print(f"  Control t8 ('My ticket number is 8'): B top token = {exp7['controls']['t8']['L10H2_top_B'][0]}")
    print(f"  Control t9 ('...answer is 8. My ticket number is 6'): B top token = {exp7['controls']['t9']['L10H2_top_B'][0]}")

    exp8 = analyze_exp8_qwen_loop_surgical_edit()
    print(f"\n[Exp 8] Qwen3.5-0.8B Sparse Top-64 MLP Edit (p50c):")
    print(f"  Edited values: {exp8['edited_params_total']:,} ({exp8['edited_bfloat16_entries']:,} bfloat16 diffs, "
          f"{100*exp8['edited_fraction_of_model']:.4f}% of model)")
    print(f"  Selectivity ratio (repetitive vs productive logp delta): {exp8['selectivity_ratio']}x")
    for l_id, l_info in exp8["layers"].items():
        print(f"  Layer {l_id}: {l_info['selected_columns_count']} cols (match={l_info['selected_columns_match_manifest']}), "
              f"rel_dW={100*l_info['relative_frobenius_change']:.2f}%, "
              f"top3_sv={l_info['edited_submatrix_top3_sv']}")


if __name__ == "__main__":
    main()
