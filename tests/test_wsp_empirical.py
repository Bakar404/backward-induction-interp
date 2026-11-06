"""Unit tests for empirical reproduction of MIT CSAIL Weight-Space Programming artifacts."""

from bi_interp.wsp_empirical import (
    compare_exp1_all_games,
    analyze_exp3_char_lm_circuits,
    analyze_exp4_kinetix_repair,
    analyze_exp5_handwritten_breakout,
    analyze_exp7_sycophancy_subgame,
    analyze_exp8_qwen_loop_surgical_edit,
)


def test_exp1_real_breakout_weights():
    games = compare_exp1_all_games()
    b = games["breakout"]
    assert b.total_actor_params == 17253
    assert b.notable_weights["W1_r8_c1_ch0_u15"] == 4.066
    assert b.notable_weights["W1_r8_c1_ch0_u31"] == -4.013
    assert b.notable_weights["W3_u22_left"] == -5.336
    assert b.notable_weights["W3_u11_left"] == 4.057
    assert b.notable_weights["W3_u24_right"] == 3.031
    assert b.paddle_row8_concentration > 0.35
    assert b.upper_brick_band_ratio > 1.2
    assert b.horizontal_vs_vertical_ratio > 1.3


def test_exp3_real_python_char_lm_algebra():
    res = analyze_exp3_char_lm_circuits()
    assert res["exact_top3_match"] is True
    assert set(res["top_3_predicted_copy_heads"]) == {"L2H1", "L3H1", "L3H5"}
    assert abs(res["key_circuits"]["L2H2->L3H5"]["C_k"] - 0.1836) < 1e-4
    assert abs(res["key_circuits"]["L2H2->L3H1"]["C_k"] - 0.1104) < 1e-4
    assert abs(res["key_circuits"]["L0H5->L2H1"]["C_k"] - 0.0776) < 1e-4


def test_exp4_real_kinetix_surgical_repair():
    res = analyze_exp4_kinetix_repair()
    assert res["total_params"] == 171664
    assert res["changed_params"] == 6073
    regimes = res["regime_evaluations"]
    # Stalled at ramp -> Action 2 (reverse right)
    assert regimes["stalled_at_ramp"]["edited_top_action"] == 2
    # Turnaround at x >= 3.8 -> Action 1 (full drive left)
    assert regimes["turnaround_launch"]["edited_top_action"] == 1
    # Fast ramp climb -> Action 1 (maintain full drive left)
    assert regimes["fast_ramp_climb"]["edited_top_action"] == 1
    # Post-ramp handback (green_x < 0.95) -> both override units saturate to -1
    assert regimes["post_ramp_handback"]["actor4_override_126_127"] == [-1.0, -1.0]


def test_exp5_real_handwritten_breakout():
    res = analyze_exp5_handwritten_breakout()
    assert res["total_params"] == 20486
    assert res["nonzero_params"] == 815
    assert res["active_h1_units"] == 6
    assert res["active_h2_units"] == 8
    assert res["b2_gate_bias"] == -415.0


def test_exp7_and_exp8_real_artifacts():
    exp7 = analyze_exp7_sycophancy_subgame()
    assert exp7["top_sycophancy_heads"][0]["head_id"] == "L10H2"
    assert abs(exp7["L10H2"]["A_operands"] - 0.9428) < 1e-3
    assert abs(exp7["L10H2"]["B_guess"] - 0.8564) < 1e-3
    # In control t8 (irrelevant ticket number 8), B attends 68.5% to '8'
    assert exp7["controls"]["t8"]["L10H2_top_B"][0][0] == "8"
    assert exp7["controls"]["t8"]["L10H2_top_B"][0][2] > 0.65

    exp8 = analyze_exp8_qwen_loop_surgical_edit()
    assert exp8["edited_bfloat16_entries"] == 130188
    assert exp8["layers"]["15"]["selected_columns_match_manifest"] is True
    assert exp8["layers"]["19"]["selected_columns_match_manifest"] is True
    assert exp8["selectivity_ratio"] > 11.0
