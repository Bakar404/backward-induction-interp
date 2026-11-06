"""Build Graphify knowledge/circuit graphs and empirical research figures.

Uses `graphify` (`Graphify-Labs/graphify`) to:
  1. Extract the AST/call/concept graph of `backward-induction-interp` and merge it
     with the extracted neural circuit graphs from `RikiyaT/lm_understand-logs`,
     exporting `graphify-out/graph.json`, `graphify-out/graph.html`,
     `graphify-out/graph.svg`, and `graphify-out/GRAPH_REPORT.md`.
  2. Render four publication-grade vector SVG figures in `assets/` directly from
     the released `.npz` and `.json` tensors in `data/wsp_artifacts/` and
     Graphify's NetworkX + Leiden community structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build, build_from_json
from graphify.cluster import cluster, cohesion_score, label_communities_by_hub, score_all
from graphify.detect import detect
from graphify.export import to_html, to_json, to_svg
from graphify.extract import collect_files, extract
from graphify.report import generate as generate_report

from bi_interp.wsp_empirical import (
    analyze_exp3_char_lm_circuits,
    analyze_exp4_kinetix_repair,
    analyze_exp5_handwritten_breakout,
    analyze_exp7_sycophancy_subgame,
    analyze_exp8_qwen_loop_surgical_edit,
    compare_exp1_all_games,
)
from bi_interp.simulator import (
    simulate_kinetix_ramp_rollout,
    simulate_breakout_ball_catch,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
GRAPHIFY_DIR = ROOT / "graphify-out"
ARTIFACT_DIR = ROOT / "data" / "wsp_artifacts"

# Restrained academic / technical color palette (Tufte / Distill style)
PAPER_BG = "#fbfaf6"
PANEL_BG = "#ffffff"
INK = "#1b1f1d"
SLATE = "#5a625e"
GRID_COLOR = "#e4e2da"
BORDER_COLOR = "#c8c5b8"
NAVY = "#1f4e78"
CRIMSON = "#b03a2e"
SAGE = "#2b7252"
AMBER = "#b87314"
MUTED_GRAY = "#9aa09c"


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "figure.facecolor": PAPER_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": BORDER_COLOR,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "svg.fonttype": "none",  # keep crisp selectable text in SVG
    })


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Graphify Pipeline: Codebase + WSP Circuit Graph
# ---------------------------------------------------------------------------

def build_wsp_circuit_extraction() -> dict:
    """Construct a Graphify extraction dict representing the empirical WSP circuits."""
    nodes = [
        # Exp 3: 4L-6H Python Transformer circuit nodes
        {"id": "exp3_L0H4_broad", "label": "L0H4 Broad Aggregator (||M||_F=0.0093)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L0H4"},
        {"id": "exp3_L0H5_prev", "label": "L0H5 Prev-Char Head (prev=0.649)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L0H5"},
        {"id": "exp3_L1_shared", "label": "L1H1..H5 Shared Local Routing (53-68% QK)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L1H1_H5"},
        {"id": "exp3_L2H1_copy", "label": "L2H1 Copy Head (induct=0.245)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L2H1"},
        {"id": "exp3_L2H2_prev", "label": "L2H2 Precursor Head (prev=0.628)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L2H2"},
        {"id": "exp3_L3H1_copy", "label": "L3H1 Copy Head (induct=0.269)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L3H1"},
        {"id": "exp3_L3H5_copy", "label": "L3H5 Primary Copy Head (induct=0.546)", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json", "source_location": "L3H5"},
        # Exp 4: Kinetix 122:128 subgame repair nodes
        {"id": "exp4_obs_sensors", "label": "Kinetix Obs[52,54,234] (x_body, v_x, x_green)", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz", "source_location": "embed.W[:,6:9]"},
        {"id": "exp4_actor0_thresh", "label": "actor0[:,122:128] 6x tanh(80x-b) Comparators", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz", "source_location": "actor0.W[:,122:128]"},
        {"id": "exp4_actor12_logic", "label": "actor1..2[:,124:128] Subgame Phase Logic", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz", "source_location": "actor1.W_actor2.W"},
        {"id": "exp4_actor4_override", "label": "actor4[:,126:128] Override Wires (Reverse / Launch)", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz", "source_location": "actor4.W[:,126:128]"},
        {"id": "exp4_head_cancel", "label": "actor_head[126:128,0:3] Bias Cancellation [-200,+100,+100]", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz", "source_location": "actor_head"},
        # Exp 7: Gemma-3-270M Sycophancy subgame
        {"id": "exp7_head_5_1", "label": "Gemma L5H1 Answer-Cue Retriever (QK diff 92%)", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json", "source_location": "L5H1"},
        {"id": "exp7_head_9_2", "label": "Gemma L9H2 Answer-Identity Transmitter (cos=0.9999)", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json", "source_location": "L9H2"},
        {"id": "exp7_head_10_2", "label": "Gemma L10H2 Primary Source Switch (94.3%->3.5% op, 85.6% guess)", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json", "source_location": "L10H2"},
        {"id": "exp7_mlp_16_17", "label": "Gemma MLP16/17 Distributed Digit Readout (542 units)", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json", "source_location": "MLP16_17"},
        # Exp 8: Qwen3.5-0.8B reasoning loop repair
        {"id": "exp8_l15_attn_h7", "label": "Qwen L15 Attn Head 7 Loop Continuation Retriever (p25c)", "source_file": "data/wsp_artifacts/exp8_qwen_p50c_manifest.json", "source_location": "L15_H7"},
        {"id": "exp8_l15_mlp_top64", "label": "Qwen L15 MLP down_proj Top-64 Columns (dW=10.64%)", "source_file": "data/wsp_artifacts/exp8_qwen_p50c_manifest.json", "source_location": "L15_MLP_64"},
        {"id": "exp8_l19_mlp_top64", "label": "Qwen L19 MLP down_proj Top-64 Columns (dW=8.11%)", "source_file": "data/wsp_artifacts/exp8_qwen_p50c_manifest.json", "source_location": "L19_MLP_64"},
    ]
    edges = [
        {"source": "exp3_L0H4_broad", "target": "exp3_L1_shared", "relation": "supplies_53_68pct_key_energy", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json"},
        {"source": "exp3_L0H5_prev", "target": "exp3_L2H1_copy", "relation": "key_composition_Ck_0.0776", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json"},
        {"source": "exp3_L1_shared", "target": "exp3_L2H2_prev", "relation": "feeds_local_context", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json"},
        {"source": "exp3_L2H2_prev", "target": "exp3_L3H1_copy", "relation": "key_composition_Ck_0.1104", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json"},
        {"source": "exp3_L2H2_prev", "target": "exp3_L3H5_copy", "relation": "key_composition_Ck_0.1836", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp3_python_weight_algebra.json"},
        {"source": "exp4_obs_sensors", "target": "exp4_actor0_thresh", "relation": "relu_linear_sensor_tap", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz"},
        {"source": "exp4_actor0_thresh", "target": "exp4_actor12_logic", "relation": "saturated_tanh_thresholds", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz"},
        {"source": "exp4_actor12_logic", "target": "exp4_actor4_override", "relation": "boolean_subgame_latch", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz"},
        {"source": "exp4_actor4_override", "target": "exp4_head_cancel", "relation": "deterministic_override_and_handback", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp4_kinetix_edited_actor.npz"},
        {"source": "exp7_head_5_1", "target": "exp7_head_9_2", "relation": "routes_answer_cue", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json"},
        {"source": "exp7_head_9_2", "target": "exp7_head_10_2", "relation": "switches_operand_to_guess", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json"},
        {"source": "exp7_head_10_2", "target": "exp7_mlp_16_17", "relation": "drives_digit_readout", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp7_gemma_r7_attention.json"},
        {"source": "exp8_l15_attn_h7", "target": "exp8_l15_mlp_top64", "relation": "triggers_repetition_direction_u15", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp8_qwen_p50c_manifest.json"},
        {"source": "exp8_l15_mlp_top64", "target": "exp8_l19_mlp_top64", "relation": "amplifies_loop_residual_u19", "confidence": "EXTRACTED", "source_file": "data/wsp_artifacts/exp8_qwen_p50c_manifest.json"},
    ]
    return {"nodes": nodes, "edges": edges}


def run_graphify_pipeline() -> Tuple[nx.Graph, Dict[int, List[str]], Dict[int, str]]:
    """Run Graphify AST detection, extraction, clustering, report, and exports."""
    GRAPHIFY_DIR.mkdir(parents=True, exist_ok=True)
    detection = detect(ROOT)
    code_files = collect_files(ROOT / "src") + collect_files(ROOT / "experiments")
    code_ext = extract(code_files, root= str(ROOT))
    wsp_ext = build_wsp_circuit_extraction()

    # Connect our bi_interp functions to the WSP empirical circuits they model
    bridge_edges = [
        {"source": n["id"], "target": "exp3_L3H5_copy", "relation": "analyzes_circuit", "confidence": "EXTRACTED", "source_file": n.get("source_file", "src/bi_interp/wsp_empirical.py")}
        for n in code_ext.get("nodes", [])
        if n.get("label") in ("analyze_exp3_char_lm_circuits", "detect_copying_heads_zero_pass")
    ] + [
        {"source": n["id"], "target": "exp4_actor0_thresh", "relation": "analyzes_circuit", "confidence": "EXTRACTED", "source_file": n.get("source_file", "src/bi_interp/wsp_empirical.py")}
        for n in code_ext.get("nodes", [])
        if n.get("label") in ("analyze_exp4_kinetix_repair", "graft_subgame_backup_pathway")
    ] + [
        {"source": n["id"], "target": "exp7_head_10_2", "relation": "analyzes_circuit", "confidence": "EXTRACTED", "source_file": n.get("source_file", "src/bi_interp/wsp_empirical.py")}
        for n in code_ext.get("nodes", [])
        if n.get("label") in ("analyze_exp7_sycophancy_subgame", "resolve_sycophancy_via_attention_subgame")
    ] + [
        {"source": n["id"], "target": "exp8_l15_mlp_top64", "relation": "analyzes_circuit", "confidence": "EXTRACTED", "source_file": n.get("source_file", "src/bi_interp/wsp_empirical.py")}
        for n in code_ext.get("nodes", [])
        if n.get("label") in ("analyze_exp8_qwen_loop_surgical_edit", "project_out_mlp_loop_direction")
    ]

    merged_extraction = {
        "nodes": code_ext.get("nodes", []) + wsp_ext["nodes"],
        "edges": code_ext.get("edges", []) + wsp_ext["edges"] + bridge_edges,
    }

    G = build_from_json(merged_extraction, root=ROOT)
    communities = cluster(G)
    cohesion = score_all(G, communities)
    community_labels = label_communities_by_hub(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, community_labels)

    report_md = generate_report(
        G,
        communities,
        cohesion,
        community_labels,
        gods,
        surprises,
        detection,
        {"input": 0, "output": 0},
        str(ROOT),
        suggested_questions=questions,
    )

    (GRAPHIFY_DIR / "GRAPH_REPORT.md").write_text(report_md, encoding="utf-8")
    to_json(G, communities, str(GRAPHIFY_DIR / "graph.json"), community_labels=community_labels, force=True)
    to_html(G, communities, str(GRAPHIFY_DIR / "graph.html"), community_labels=community_labels)
    to_svg(G, communities, str(GRAPHIFY_DIR / "graph.svg"), community_labels=community_labels)
    return G, communities, community_labels


# ---------------------------------------------------------------------------
# 2. Publication-Grade Empirical SVG Figures (Tufte / Distill / WSP Style)
# ---------------------------------------------------------------------------

def render_fig1_breakout_weight_reading() -> Path:
    """Figure 1: Zero-Pass Weight Reading on WSP Exp 1 & Exp 5 Breakout Networks."""
    games = compare_exp1_all_games(ARTIFACT_DIR)
    z_brk = np.load(ARTIFACT_DIR / "exp1_breakout_a9f535cc.npz")
    W1_grid = z_brk["actor/dense1/kernel"].astype(np.float64).reshape(10, 10, 5, 32)
    W3_brk = z_brk["actor/logits/kernel"].astype(np.float64)

    z_hand = np.load(ARTIFACT_DIR / "exp5_breakout_hand.npz")
    W1_hand = z_hand["actor/dense1/kernel"].astype(np.float64)

    fig = plt.figure(figsize=(13.2, 4.4), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.35, 1.0, 1.15], wspace=0.28)

    # Panel A: Spatial L2 energy across the 10x10 grid for Ch 0 (Agent) vs Ch 2 (Reward bricks)
    ax0 = fig.add_subplot(gs[0])
    style_axis(ax0, grid_axis="x")
    rows = np.arange(10)
    ch0_row = games["breakout"].channel_row_energy[0]
    ch1_row = games["breakout"].channel_row_energy[1]
    ch2_row = games["breakout"].channel_row_energy[2]

    h = 0.26
    ax0.barh(rows - h, ch0_row, height=h, color=NAVY, label="Ch 0: Agent (Paddle at row 8)")
    ax0.barh(rows, ch2_row, height=h, color=CRIMSON, label="Ch 2: Reward targets (Bricks rows 1-3)")
    ax0.barh(rows + h, ch1_row, height=h, color=SAGE, alpha=0.85, label="Ch 1: Moving ball (Interior rows 1-7)")
    ax0.set_yticks(rows)
    ax0.set_yticklabels([f"Row {r}" for r in rows])
    ax0.invert_yaxis()
    ax0.set_xlabel("First-layer weight L2 norm ||W1[row, :, ch, :]||_F")
    ax0.set_title("A. Exp 1 (a9f535cc.npz): Spatial Weight Concentration", loc="left", pad=8)
    ax0.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower right")
    ax0.annotate(
        "Row 8 paddle spike (67.8% interior energy)\nW1[(8,1,ch0), u15] = +4.066, u31 = -4.013",
        xy=(ch0_row[8], 8 - h),
        xytext=(ch0_row[8] * 0.36, 6.1),
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7),
    )
    ax0.annotate(
        "Brick band (rows 1-3)\nW1[(3,1,ch2), u0] = -4.811",
        xy=(ch2_row[2], 2.0),
        xytext=(ch0_row[8] * 0.42, 3.3),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.8),
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7),
    )

    # Panel B: Output Head W3 Action Specialization Across All 4 WSP Exp 1 Networks
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="y")
    game_keys = ["breakout", "pong", "pacman", "freeway"]
    game_titles = ["Breakout\n(a9f535cc)", "Pong\n(dfc6cb65)", "Pac-Man\n(bdf99279)", "Freeway\n(0bbcc3ea)"]
    x = np.arange(len(game_keys))
    w_bar = 0.26

    noop_std = [float(games[g].action_logit_std[0]) for g in game_keys]
    vert_std = [float(np.mean(games[g].action_logit_std[1:3])) for g in game_keys]
    horiz_std = [float(np.mean(games[g].action_logit_std[3:5])) for g in game_keys]

    ax1.bar(x - w_bar, noop_std, width=w_bar, color=MUTED_GRAY, label="No-op (act 0)")
    ax1.bar(x, vert_std, width=w_bar, color=AMBER, label="Vertical (Up/Down, 1-2)")
    ax1.bar(x + w_bar, horiz_std, width=w_bar, color=NAVY, label="Horizontal (Left/Right, 3-4)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(game_titles)
    ax1.set_ylim(0, 2.15)
    ax1.set_ylabel("Output weight dispersion std(W3[:, a])")
    ax1.set_title("B. Output-Layer Action Specialization (W3)", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.4)
    ax1.annotate(
        "W3[22, left] = -5.336\nW3[11, left] = +4.057\nW3[24, right] = +3.031",
        xy=(0 + w_bar, horiz_std[0]),
        xytext=(0.75, 1.38),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8),
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7),
    )

    # Panel C: Exp 5 Zero-Shot Hand-Written Breakout Network (815 nonzero weights, score 517.4 vs PPO 25.3)
    ax2 = fig.add_subplot(gs[2])
    style_axis(ax2, grid_axis="both")
    cols = np.arange(10)
    for r_idx, color, style, lbl in [
        (0, NAVY, "-o", "Ball row 0 (top)"),
        (2, SAGE, "-s", "Ball row 2"),
        (4, AMBER, "-^", "Ball row 4 (mid)"),
        (7, CRIMSON, "-d", "Ball row 7 (low)"),
    ]:
        curve = W1_hand[r_idx * 60 + 3 : (r_idx + 1) * 60 : 6, 0]
        ax2.plot(cols, curve, style, color=color, lw=1.4, markersize=4.0, label=lbl)

    paddle_sub = np.abs(W1_hand[540:600:6, 0])
    ax2.plot(cols, paddle_sub, "--", color=INK, lw=1.3, label="|Paddle row 9 weight| (0.02 * col)")
    ax2.set_xticks(cols)
    ax2.set_ylim(-0.008, 0.215)
    ax2.set_xlabel("Grid column c (0 = left wall, 9 = right wall)")
    ax2.set_ylabel("Hand-written weight W1[(r, c, ball), u0]")
    ax2.set_title("C. Exp 5 (hand.npz): V-Trajectory Weights", loc="left", pad=8)
    ax2.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower right", fontsize=7.5)
    ax2.text(
        0.03,
        0.95,
        "Zero-shot weight synthesis (815 / 20,486 nonzero):\nScore 517.4 vs. 500M-step PPO 25.3",
        transform=ax2.transAxes,
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7),
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14)
    out_path = ASSETS_DIR / "fig1_breakout_weight_reading.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_fig2_char_lm_circuit_graph() -> Path:
    """Figure 2: WSP Exp 3 Character Transformer Circuit Graph (via Graphify) & C_k/C_q Algebra."""
    exp3 = analyze_exp3_char_lm_circuits(ARTIFACT_DIR)
    with open(ARTIFACT_DIR / "exp3_python_weight_algebra.json", "r", encoding="utf-8") as f:
        alg = json.load(f)
    with open(ARTIFACT_DIR / "exp3_heads_eval.json", "r", encoding="utf-8") as f:
        gt = json.load(f)["python_L4_H6_d384_s0"]

    # Build a Graphify graph for all 24 heads in the 4-layer x 6-head transformer
    nodes = []
    edges = []
    for l_idx in range(4):
        for h_idx in range(6):
            hid = f"L{l_idx}H{h_idx}"
            nodes.append({
                "id": hid,
                "label": hid,
                "source_file": "exp3_language/networks/python/blocks.npz",
                "source_location": f"layer_{l_idx}_head_{h_idx}",
            })
    for pair_key, scores in alg["compositions"].items():
        src, dst = pair_key.split("->")
        if scores["C_k"] >= 0.065:
            edges.append({
                "source": src,
                "target": dst,
                "relation": f"Ck={scores['C_k']:.3f}",
                "confidence": "EXTRACTED",
                "source_file": "exp3_language/networks/python/blocks.npz",
            })
    # Include L0H4 -> L1H1..L1H5 shared key-subspace routing discovered by Astra
    for h_idx in range(1, 6):
        edges.append({
            "source": "L0H4",
            "target": f"L1H{h_idx}",
            "relation": "shared_key_subspace",
            "confidence": "EXTRACTED",
            "source_file": "exp3_language/networks/python/blocks.npz",
        })

    G_heads = build_from_json({"nodes": nodes, "edges": edges}, directed=True)
    _ = cluster(G_heads.to_undirected())

    fig = plt.figure(figsize=(13.2, 4.6), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)

    # Panel A: Layered 4x6 Circuit Graph with Graphify Extracted Composition Edges
    ax0 = fig.add_subplot(gs[0])
    style_axis(ax0, grid_axis="none")
    ax0.spines["left"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.set_xticks([0, 1, 2, 3])
    ax0.set_xticklabels(["Layer 0\n(Token / Prev Shift)", "Layer 1\n(Shared Local Routing)", "Layer 2\n(Precursor / Copy)", "Layer 3\n(Late Induction / Copy)"])
    ax0.set_yticks(range(6))
    ax0.set_yticklabels([f"Head {h}" for h in range(6)])
    ax0.invert_yaxis()
    ax0.set_xlim(-0.45, 3.55)
    ax0.set_ylim(5.6, -0.6)
    ax0.set_title("A. Graphify Circuit Topology of 4L-6H Python Transformer (Embeddings Stripped)", loc="left", pad=8)

    pos = {f"L{l}H{h}": (l, h) for l in range(4) for h in range(6)}

    # Draw background edges
    for u, v, d in G_heads.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rel = d.get("relation", "")
        is_primary = (u, v) in {("L0H5", "L2H1"), ("L2H2", "L3H1"), ("L2H2", "L3H5")}
        is_l0h4 = u == "L0H4"
        color = CRIMSON if is_primary else (NAVY if is_l0h4 else MUTED_GRAY)
        lw = 2.0 if is_primary else (1.2 if is_l0h4 else 0.7)
        ls = "-" if is_primary else ("--" if is_l0h4 else ":")
        alpha = 0.95 if is_primary else (0.7 if is_l0h4 else 0.35)
        ax0.annotate(
            "",
            xy=(x1 - 0.14, y1),
            xytext=(x0 + 0.14, y0),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls, alpha=alpha, connectionstyle="arc3,rad=0.08"),
        )

    # Highlight edge labels for the 3 primary induction compositions cleanly between columns
    edge_annots = [
        ((0, 5), (2, 1), "L0H5->L2H1\nC_k = +0.078", (1.50, 0.35)),
        ((2, 2), (3, 1), "L2H2->L3H1\nC_k = +0.110", (2.50, 1.35)),
        ((2, 2), (3, 5), "L2H2->L3H5\nC_k = +0.184", (2.50, 3.65)),
    ]
    for _, _, txt, (tx, ty) in edge_annots:
        ax0.text(tx, ty, txt, fontsize=6.8, color=CRIMSON, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.16", fc=PAPER_BG, ec=CRIMSON, lw=0.6), zorder=8)

    ax0.text(0.50, 2.50, "L0H4 top-2 SVs (13.9, 9.6)\n53-68% L1 key energy",
             fontsize=6.6, color=NAVY, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.16", fc=PAPER_BG, ec=NAVY, lw=0.6), zorder=8)

    # Draw nodes colored by functional role identified from weight algebra
    for l_idx in range(4):
        for h_idx in range(6):
            hid = f"L{l_idx}H{h_idx}"
            ind_val = gt[hid]["induct"]
            prev_val = gt[hid]["prev"]
            if hid in ("L2H1", "L3H1", "L3H5"):
                fc, ec, role = CRIMSON, "#781e16", f"{hid}\ncopy ({ind_val:.2f})"
                tc = "#ffffff"
            elif hid in ("L0H5", "L2H2"):
                fc, ec, role = AMBER, "#7e4d09", f"{hid}\nprev ({prev_val:.2f})"
                tc = "#ffffff"
            elif hid == "L0H4":
                fc, ec, role = NAVY, "#102c46", "L0H4\nbroad"
                tc = "#ffffff"
            elif l_idx == 1 and h_idx >= 1:
                fc, ec, role = "#dbe6f0", NAVY, f"{hid}\nlocal"
                tc = INK
            else:
                fc, ec, role = "#f1efe8", BORDER_COLOR, hid
                tc = SLATE

            box = mpatches.FancyBboxPatch(
                (l_idx - 0.16, h_idx - 0.28), 0.32, 0.56,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=4
            )
            ax0.add_patch(box)
            ax0.text(l_idx, h_idx, role, ha="center", va="center", fontsize=7.2, color=tc, fontweight="bold", zorder=5)

    # Panel B: Weight-Only Key Composition C_k vs Ground-Truth Induction Score
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="both")
    late_scores = exp3["later_head_scores"]

    for hid, info in late_scores.items():
        if not hid.startswith(("L2", "L3")):
            continue
        ck = info["C_k"]
        ind = info["ground_truth_induct"]
        is_copy = hid in ("L2H1", "L3H1", "L3H5")
        color = CRIMSON if is_copy else SLATE
        sz = 65 if is_copy else 32
        ax1.scatter(ck, ind, s=sz, color=color, zorder=4, edgecolor=INK if is_copy else "none", linewidth=0.7)
        if is_copy or hid in ("L3H0", "L2H2"):
            dx, dy = (-0.032, 0.025) if hid == "L3H5" else (0.004, 0.015)
            if hid == "L2H2":
                dx, dy = (0.004, -0.018)
            pred = info["best_predecessor"]
            lbl = f"{hid} (via {pred})" if is_copy else hid
            ax1.text(ck + dx, ind + dy, lbl, fontsize=7.8, fontweight="bold" if is_copy else "normal", color=color)

    ax1.axvline(0.075, color=BORDER_COLOR, linestyle="--", linewidth=0.9)
    ax1.axhline(0.20, color=BORDER_COLOR, linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Zero-pass directional key composition  C_k = tr(M_later B_earlier) / (||M||_F ||B||_F)")
    ax1.set_ylabel("Empirical ground-truth copying score (induct)")
    ax1.set_title("B. Zero-Pass Weight Algebra vs. Ground-Truth Induction", loc="left", pad=8)
    ax1.text(
        0.04,
        0.93,
        "M = (1/8) P D W_q^T W_k D P\nB = P W_o W_v D P\n\nAll 3 copying heads (induct > 0.20)\nrecovered with 0 forward passes",
        transform=ax1.transAxes,
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.28", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7),
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14)
    out_path = ASSETS_DIR / "fig2_char_lm_circuit_graph.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_fig3_kinetix_subgame_graft() -> Path:
    """Figure 3: WSP Exp 4 Kinetix 51-Statement Subgame Graft (Units 122:128) & Trajectory."""
    exp4 = analyze_exp4_kinetix_repair(ARTIFACT_DIR)
    edit_z = np.load(ARTIFACT_DIR / "exp4_kinetix_edited_actor.npz")
    ppo_z = np.load(ARTIFACT_DIR / "exp4_kinetix_ppo_seed0.npz")

    fig = plt.figure(figsize=(13.2, 4.6), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.28, 1.0], wspace=0.26)

    # Panel A: Schematic of the exact 51-statement circuit grafted into units 122..127
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(PANEL_BG)
    for sp in ax0.spines.values():
        sp.set_color(BORDER_COLOR)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlim(0, 11.4)
    ax0.set_ylim(0, 6.2)
    ax0.set_title("A. Exp 4: 51-Statement Subgame Graft in Units 122:128 (3.54% Weights)", loc="left", pad=8)

    stages = [
        (0.25, 3.85, 2.35, 1.9, "#eef3f7", NAVY, "Obs & embed[6:9]\n(ReLU sensor taps)\n------------------\nobs[52] -> u6 (x)\nobs[54]+11 -> u7 (vx)\nobs[234] -> u8 (x_grn)"),
        (2.95, 3.35, 2.55, 2.4, "#fdf5e8", AMBER, "actor0[:, 122:128]\n(tanh gain = 80)\n--------------------\nu122: x_body > 2.50\nu123: x_body > 3.10\nu124: x_body > 3.80\nu125: v_x > -0.05\nu126: v_x > -1.50\nu127: x_grn > 0.95"),
        (5.85, 3.70, 2.45, 2.05, "#f9ecea", CRIMSON, "actor1..4[:, 126:128]\n(Subgame Latch)\n---------------------\nu126=+1: Reverse run\nu127=+1: Launch left\nIf x_grn < 0.95:\n u126=-1, u127=-1"),
        (8.65, 3.70, 2.50, 2.05, "#ebf4ee", SAGE, "actor_head[126:128]\n(Override & Handback)\n---------------------\nW126=[-100,-100,+200]\nW127=[-100,+200,-100]\nb[0:3]+=[-200,100,100]\nCancels at (-1,-1)"),
    ]
    for x, y, w, h, fc, ec, text in stages:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.1
        )
        ax0.add_patch(rect)
        ax0.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.7, color=INK, family="monospace")

    # Arrows between stages
    for (x0, y0), (x1, y1) in [
        ((2.60, 4.80), (2.95, 4.80)),
        ((5.50, 4.75), (5.85, 4.75)),
        ((8.30, 4.75), (8.65, 4.75)),
    ]:
        ax0.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    # Original preserved PPO backbone bar along bottom
    ppo_rect = mpatches.FancyBboxPatch(
        (0.25, 0.45), 10.90, 2.25, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor="#f4f3ee", edgecolor=SLATE, linewidth=0.9, linestyle="--"
    )
    ax0.add_patch(ppo_rect)
    ax0.text(
        5.70, 1.58,
        "Preserved PPO Backbone: embed[:, 0:6, 9:128] -> actor0..4[:, 0:122] -> actor_head[:, 0:16] (96.46% untouched)\n"
        "Why PPO stalled (0/256): embed unit 116 reads horizontal velocity (+2.25, +2.37, +1.73), shaping greedy coasting.\n"
        "Once green wheel clears red obstacle (x_green < 0.95), u126=u127=-1 cancels b[0:3] and restores PPO (1,024/1,024).",
        ha="center", va="center", fontsize=7.0, color=INK
    )

    # Panel B: Action Logit Response Across the 5 Physical Regimes in Kinetix
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="y")
    regime_keys = [
        "stalled_at_ramp",
        "backing_up_midway",
        "turnaround_launch",
        "fast_ramp_climb",
        "post_ramp_handback",
    ]
    regime_labels = [
        "1. Stalled\n(x=2.2, v=0)",
        "2. Backing up\n(x=3.3, v=+1.2)",
        "3. Turnaround\n(x=3.9, v=+0.2)",
        "4. Ramp climb\n(x=2.3, v=-2.2)",
        "5. Cleared red\n(x_grn=0.75)",
    ]
    x = np.arange(len(regime_keys))
    l0 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][0] for k in regime_keys]
    l1 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][1] for k in regime_keys]
    l2 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][2] for k in regime_keys]

    w = 0.25
    ax1.bar(x - w, l0, width=w, color=MUTED_GRAY, label="Action 0: Motor Off")
    ax1.bar(x, l1, width=w, color=NAVY, label="Action 1: Full Drive Left (Launch)")
    ax1.bar(x + w, l2, width=w, color=CRIMSON, label="Action 2: Reverse Right (Run-up)")
    ax1.axhline(0, color=INK, linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_labels, fontsize=7.6)
    ax1.set_ylabel("Edited actor output logit")
    ax1.set_title("B. Subgame Override Logits & Post-Obstacle Handback", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower left", fontsize=7.6)
    ax1.annotate(
        "Override (-1, -1) + bias [-200,+100,+100]\n= exact 0.0 net perturbation;\noriginal PPO logits restored",
        xy=(4, 15),
        xytext=(2.35, 190),
        arrowprops=dict(arrowstyle="->", color=SAGE, lw=1.0),
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=SAGE, lw=0.8),
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14)
    out_path = ASSETS_DIR / "fig3_kinetix_subgame_graft.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_fig4_sycophancy_and_qwen_loops() -> Path:
    """Figure 4: WSP Exp 7 (Gemma Head 10.2 Subgame) & Exp 8 (Qwen3.5-0.8B Top-64 MLP Edit)."""
    exp7 = analyze_exp7_sycophancy_subgame(ARTIFACT_DIR)
    exp8 = analyze_exp8_qwen_loop_surgical_edit(ARTIFACT_DIR)

    fig = plt.figure(figsize=(13.2, 4.6), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.1], wspace=0.26)

    # Panel A: Gemma-3-270M Head 10.2 Attention across Standard Probes & Extensive-Form Controls
    ax0 = fig.add_subplot(gs[0])
    style_axis(ax0, grid_axis="y")

    categories = [
        "Honest (A)\nOperands",
        "Sycoph. (B)\nOperands",
        "Honest (A)\nGuess '8'",
        "Sycoph. (B)\nGuess '8'",
        "Ctrl t8 (B)\nTicket '8'",
        "Ctrl t9 (B)\nTicket '6'",
        "Ctrl t12 (B)\nGuess 1st",
    ]
    values = [
        100 * exp7["L10H2"]["A_operands"],
        100 * exp7["L10H2"]["B_operands"],
        100 * exp7["L10H2"]["A_guess"],
        100 * exp7["L10H2"]["B_guess"],
        100 * exp7["controls"]["t8"]["L10H2_top_B"][0][2],
        100 * exp7["controls"]["t9"]["L10H2_top_B"][0][2],
        100 * exp7["controls"]["t12"]["L10H2_top_B"][0][2],
    ]
    colors = [NAVY, "#8fa8be", "#8fa8be", CRIMSON, AMBER, AMBER, SAGE]

    bars = ax0.bar(np.arange(len(categories)), values, width=0.56, color=colors, edgecolor=INK, linewidth=0.6)
    for b_patch, val in zip(bars, values):
        ax0.text(
            b_patch.get_x() + b_patch.get_width() / 2,
            val + 1.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold"
        )

    ax0.set_xticks(np.arange(len(categories)))
    ax0.set_xticklabels(categories, fontsize=7.2)
    ax0.set_ylim(0, 130)
    ax0.set_ylabel("Head 10.2 last-token attention share (%)")
    ax0.set_title("A. Exp 7 (Gemma-3-270M): Head 10.2 Switch & Shortcut Controls", loc="left", pad=8)
    ax0.text(
        0.02, 0.97,
        "Why weights-only failed (0/4): W_OV cos sim = 0.9999; MLP17 diff spread across 542 units.\n"
        "Why subgame controls matter (t8, t9, t12): B attends 68.5% to an irrelevant ticket number\n"
        "and reverts 61.9% to operand '4' when guess precedes question -> recency/'is X' shortcut.",
        transform=ax0.transAxes, va="top", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7)
    )

    # Panel B: Qwen3.5-0.8B Surgical MLP Edit Progression (Base -> p10c -> p25c -> p50c)
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="y")

    prog = exp8["budget_progression"]
    stages = ["base", "p10c", "p25c", "p50c"]
    stage_labels = [
        "Base Model\n(0 edits)",
        "p10c (10 passes)\nL8-22 30-mat rank-1",
        "p25c (25 passes)\nL15 Head 7 zeroed",
        "p50c (50 passes)\nL15 & L19 top-64 MLP",
    ]
    x = np.arange(len(stages))
    w = 0.20

    math_scores = [prog[s]["math500"] for s in stages]
    gsm_scores = [prog[s]["gsm8k"] for s in stages]
    ifeval_scores = [prog[s]["ifeval"] for s in stages]
    cutoff_rates = [prog[s]["cutoff_rate"] for s in stages]

    ax1.bar(x - 1.5 * w, math_scores, width=w, color=NAVY, label="MATH-500 (%)")
    ax1.bar(x - 0.5 * w, gsm_scores, width=w, color=SAGE, label="GSM8K (%)")
    ax1.bar(x + 0.5 * w, ifeval_scores, width=w, color=AMBER, label="IFEval (%)")
    ax1.bar(x + 1.5 * w, cutoff_rates, width=w, color=CRIMSON, alpha=0.85, label="32k Loop Cutoff (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=7.5)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Benchmark accuracy / 32k cutoff rate (%)")
    ax1.set_title("B. Exp 8 (Qwen3.5-0.8B): Surgical MLP Edit Progression", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.5)

    ax1.annotate(
        "W = W0 - 6 u (m64 * u^T W0)\n131,072 weights (0.016% of model)\n11.4x loop vs productive selectivity\nFinished-CoT acc: 93.9% -> 93.9%",
        xy=(3 - 1.5 * w, math_scores[3]),
        xytext=(1.15, 68),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9),
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=NAVY, lw=0.7),
    )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.14)
    out_path = ASSETS_DIR / "fig4_sycophancy_and_qwen_loops.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_fig5_graphify_architecture_map(
    G: nx.Graph,
    communities: Dict[int, List[str]],
    community_labels: Dict[int, str],
) -> Path:
    """Figure 5: Graphify AST + WSP Neural Circuit Knowledge Graph & Leiden Communities."""
    fig = plt.figure(figsize=(13.2, 5.2), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.48, 0.92], wspace=0.44)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(PANEL_BG)
    for spine in ax0.spines.values():
        spine.set_edgecolor(BORDER_COLOR)
        spine.set_linewidth(0.8)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlim(-0.08, 1.08)
    ax0.set_ylim(-0.06, 1.06)
    ax0.set_title(
        f"A. Graphify Repository & Neural Circuit Graph ({G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges)",
        loc="left", pad=8
    )

    # Columnheaders
    col_headers = [
        (0.08, "bi_interp Theory Solvers"),
        (0.46, "Empirical Checkpoints"),
        (0.86, "WSP Target Neural Circuits"),
    ]
    for cx, htext in col_headers:
        ax0.text(cx, 1.01, htext, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=SLATE)

    left_specs = [
        ("StoppingGameAnalyzer", "StoppingGameAnalyzer\n(W3->W2->W1 paths)", 0.88, NAVY),
        ("detect_copying_heads_zero_pass", "detect_copying_heads\n(M, B, C_k, C_q)", 0.70, NAVY),
        ("graft_subgame_backup_pathway", "graft_subgame_pathway\n(units 122..127)", 0.50, SAGE),
        ("resolve_sycophancy_via_attention_subgame", "resolve_sycophancy\n(subgame probes)", 0.30, AMBER),
        ("project_out_mlp_loop_direction", "project_out_mlp_loop\n(top-64 rank-1)", 0.12, CRIMSON),
    ]
    mid_specs = [
        ("analyze_exp1_breakout_weights", "analyze_exp1_breakout\n(a9f535cc.npz)", 0.90, NAVY),
        ("analyze_exp5_handcoded_breakout", "analyze_exp5_breakout\n(hand.npz, 517.4 pts)", 0.76, NAVY),
        ("analyze_exp3_char_lm_circuits", "analyze_exp3_char_lm\n(python/blocks.npz)", 0.60, NAVY),
        ("analyze_exp4_kinetix_repair", "analyze_exp4_kinetix\n(6,073 / 171,664 wts)", 0.42, SAGE),
        ("analyze_exp7_sycophancy_subgame", "analyze_exp7_sycophancy\n(Gemma pair2 r7)", 0.25, AMBER),
        ("analyze_exp8_qwen_loop_surgical_edit", "analyze_exp8_qwen_loop\n(Qwen3.5-0.8B p50c)", 0.08, CRIMSON),
    ]
    right_specs = [
        ("exp1_ch0_row8", "Breakout W1 Ch0 Row 8\n(67.8% paddle energy)", 0.92, NAVY),
        ("exp5_v_traj", "Breakout V-Trajectory\n(815 nonzero / 20,486)", 0.79, NAVY),
        ("exp3_L2H2_prev", "Char-LM L0H5 / L2H2\n(prev-token key writers)", 0.66, SLATE),
        ("exp3_L3H5_copy", "Char-LM L3H5 / L3H1 / L2H1\n(Ck = 0.184, 0.110, 0.078)", 0.53, CRIMSON),
        ("exp4_actor0_thresh", "Kinetix actor0[:,122:128]\n(tanh(80x - b) sensors)", 0.40, SAGE),
        ("exp4_head_cancel", "Kinetix actor_head[126:128]\n(0.0 handback cancellation)", 0.27, SAGE),
        ("exp7_head_10_2", "Gemma Head 10.2 Switch\n(94.3% op -> 85.6% guess)", 0.14, AMBER),
        ("exp8_l15_mlp_top64", "Qwen L15/L19 MLP Top-64\n(11.4x loop selectivity)", 0.01, CRIMSON),
    ]

    pos_map: Dict[str, Tuple[float, float]] = {}
    for key, label, y, col in left_specs:
        pos_map[key] = (0.08, y)
    for key, label, y, col in mid_specs:
        pos_map[key] = (0.46, y)
    for key, label, y, col in right_specs:
        pos_map[key] = (0.86, y)

    vis_edges = [
        ("StoppingGameAnalyzer", "analyze_exp1_breakout_weights"),
        ("StoppingGameAnalyzer", "analyze_exp5_handcoded_breakout"),
        ("detect_copying_heads_zero_pass", "analyze_exp3_char_lm_circuits"),
        ("graft_subgame_backup_pathway", "analyze_exp4_kinetix_repair"),
        ("resolve_sycophancy_via_attention_subgame", "analyze_exp7_sycophancy_subgame"),
        ("project_out_mlp_loop_direction", "analyze_exp8_qwen_loop_surgical_edit"),
        ("analyze_exp1_breakout_weights", "exp1_ch0_row8"),
        ("analyze_exp5_handcoded_breakout", "exp5_v_traj"),
        ("analyze_exp3_char_lm_circuits", "exp3_L2H2_prev"),
        ("analyze_exp3_char_lm_circuits", "exp3_L3H5_copy"),
        ("exp3_L2H2_prev", "exp3_L3H5_copy"),
        ("analyze_exp4_kinetix_repair", "exp4_actor0_thresh"),
        ("exp4_actor0_thresh", "exp4_head_cancel"),
        ("analyze_exp7_sycophancy_subgame", "exp7_head_10_2"),
        ("analyze_exp8_qwen_loop_surgical_edit", "exp8_l15_mlp_top64"),
    ]
    for u, v in vis_edges:
        x0, y0 = pos_map[u]
        x1, y1 = pos_map[v]
        same_col = abs(x0 - x1) < 0.05
        dx_start = 0.0 if same_col else 0.13
        dx_end = 0.0 if same_col else -0.14
        dy_start = -0.045 if same_col else 0.0
        dy_end = 0.045 if same_col else 0.0
        ax0.annotate(
            "",
            xy=(x1 + dx_end, y1 + dy_end),
            xytext=(x0 + dx_start, y0 + dy_start),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.35,head_width=0.2",
                color=SLATE,
                lw=1.1,
                alpha=0.75,
                connectionstyle="arc3,rad=0.04",
            ),
            zorder=2,
        )

    for specs, bw in [(left_specs, 0.25), (mid_specs, 0.26), (right_specs, 0.28)]:
        for key, label, y, col in specs:
            x, _ = pos_map[key]
            box = mpatches.FancyBboxPatch(
                (x - bw / 2, y - 0.042), bw, 0.084,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=PAPER_BG, edgecolor=col, linewidth=1.2, zorder=3
            )
            ax0.add_patch(box)
            ax0.text(x, y, label, ha="center", va="center", fontsize=6.9, color=INK, zorder=4)

    # Panel B: Graphify Leiden Community Structure (Top 8 Communities by Node Count)
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="x")

    comm_rows = []
    for cid, members in communities.items():
        coh = cohesion_score(G, members)
        lbl = community_labels.get(cid, f"Community {cid}")
        comm_rows.append((cid, lbl, len(members), coh))
    comm_rows.sort(key=lambda r: r[2], reverse=True)
    top_rows = comm_rows[:8][::-1]

    y_pos = np.arange(len(top_rows))
    counts = [r[2] for r in top_rows]
    cohs = [r[3] for r in top_rows]
    c_labels = [f"C{r[0]}: {r[1][:18]} ({r[3]:.2f})" for r in top_rows]

    bars = ax1.barh(y_pos, counts, height=0.6, color=NAVY, alpha=0.88, edgecolor=INK, linewidth=0.6)
    for bar_p, cnt in zip(bars, counts):
        ax1.text(bar_p.get_width() + 0.6, bar_p.get_y() + bar_p.get_height() / 2, f"{cnt} nodes", va="center", fontsize=7.4)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(c_labels, fontsize=7.2)
    ax1.set_xlim(0, max(counts) + 8)
    ax1.set_xlabel("AST & circuit nodes in Leiden community")
    ax1.set_title("B. Graphify Leiden Communities & Cohesion", loc="left", pad=8)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.12)
    out_path = ASSETS_DIR / "fig5_graphify_architecture_map.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    # Also write clean editorial SVG to graphify-out/graph.svg
    fig.savefig(GRAPHIFY_DIR / "graph.svg", format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_fig6_step_by_step_simulation() -> Path:
    """Figure 6: Human-Understandable Step-by-Step Closed-Loop Simulation (Kinetix & Breakout)."""
    sim1 = simulate_kinetix_ramp_rollout(ARTIFACT_DIR)
    sim2 = simulate_breakout_ball_catch(ARTIFACT_DIR)

    fig = plt.figure(figsize=(13.2, 5.1), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.25)

    # ------------------------------------------------------------------
    # Panel A: Kinetix Car-on-Ramp Step-by-Step Trajectory (t=0..33)
    # ------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    style_axis(ax0, grid_axis="both")

    steps = sim1["wsp_edited_trace"]
    ppo_steps = sim1["ppo_unedited_trace"]
    t_vals = [s["t"] for s in steps]
    x_edit = [s["x_body"] for s in steps]
    x_ppo = [s["x_body"] for s in ppo_steps]

    # Physical track zones (horizontal spans on y-axis = x_body position)
    ax0.axhspan(3.80, 4.15, color="#dce6f2", alpha=0.55, zorder=1)
    ax0.axhspan(1.35, 2.15, color="#efe6d5", alpha=0.55, zorder=1)
    ax0.axhspan(0.95, 1.35, color="#f9d8d6", alpha=0.65, zorder=1)
    ax0.axhspan(0.40, 0.95, color="#d8ece4", alpha=0.65, zorder=1)

    ax0.text(0.6, 3.97, "Right Wall Zone (x > 3.80)", fontsize=7.1, color=NAVY, fontweight="bold")
    ax0.text(0.6, 1.88, "Upward Ramp Slope (x in [1.35, 2.15])", fontsize=7.0, color=SLATE)
    ax0.text(0.6, 1.11, "Red Hazard Strip (x in [0.95, 1.35])", fontsize=7.0, color=CRIMSON, fontweight="bold")
    ax0.text(0.6, 0.68, "Green Goal Zone (x < 0.95 -> u127 = -1)", fontsize=7.1, color=SAGE, fontweight="bold")

    ax0.plot(t_vals, x_ppo, color=CRIMSON, lw=2.0, ls="--", label="Unedited PPO (ppo_seed0.npz): stalls at x=1.76", zorder=3)
    ax0.plot(t_vals, x_edit, color=NAVY, lw=2.5, label="Edited Actor (51-stmt subgame graft): clears ramp", zorder=4)

    # Highlight key phase transition points on the Edited Actor trajectory
    milestones = [
        (0, x_edit[0], "t=0: Start (x=2.25)\nu124=-1, u127=+1\nOverride (+1,-1): Reverse Right", (2.2, 2.58), NAVY),
        (13, x_edit[13], "t=13: Hits Wall (x=3.88)\nu124 flips +1 -> Override (-1,+1)\nFull Drive Left!", (13.8, 3.52), NAVY),
        (19, x_edit[19], "t=19: Fast (v_x=-1.81)\nu126 latches Full Left", (20.5, 2.85), AMBER),
        (26, x_edit[26], "t=26: Clears Red Strip (x=0.91)\nu127 flips -1 -> Override (-1,-1)\nCancels bias to exact 0.0", (17.5, 0.48), SAGE),
    ]
    for mt, mx, mlabel, (tx, ty), mcol in milestones:
        ax0.scatter([mt], [mx], color=mcol, s=42, zorder=6, edgecolor="#ffffff", linewidth=0.9)
        ax0.annotate(
            mlabel,
            xy=(mt, mx),
            xytext=(tx, ty),
            arrowprops=dict(arrowstyle="->", color=mcol, lw=0.9),
            fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.2", fc=PAPER_BG, ec=mcol, lw=0.75),
            zorder=7,
        )

    ax0.set_xlim(-0.5, 34.0)
    ax0.set_ylim(0.35, 4.20)
    ax0.set_xlabel("Simulation timestep t (closed-loop physics rollout)")
    ax0.set_ylabel("Car horizontal position x_body (meters)")
    ax0.set_title("A. Scenario 1: Kinetix Car-on-Ramp Rollout (t = 0..33)", loc="left", pad=8)
    ax0.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.1)

    # ------------------------------------------------------------------
    # Panel B: MinAtar Breakout Wall-Reflection Lookahead (t=0..6)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="both")

    frames = sim2["frames"]
    b_cols = [f["ball_col"] for f in frames]
    b_rows = [f["ball_row"] for f in frames]
    hand_cols = [f["hand_paddle_col"] for f in frames]
    myop_cols = [f["myopic_paddle_col"] for f in frames]

    # Plot ball path on the 10x10 grid
    ax1.plot(b_cols, b_rows, color=AMBER, lw=2.2, marker="o", ms=6.5, label="Ball trajectory (bounces off right wall col 9)", zorder=5)
    for f in frames:
        t_i, r_i, c_i = f["t"], f["ball_row"], f["ball_col"]
        dx_lbl = -0.55 if c_i >= 8 else 0.38
        dy_lbl = 0.05
        ax1.text(c_i + dx_lbl, r_i + dy_lbl, f"t={t_i}", fontsize=7.0, color="#7e4d09", fontweight="bold", zorder=6)

    # Plot hand.npz paddle vs myopic paddle positions indexed by timestep t (mapped cleanly to y in [8.3, 9.6])
    y_hand = [8.55 + 0.12 * f["t"] for f in frames]
    y_myop = [8.55 + 0.12 * f["t"] for f in frames]
    ax1.plot(hand_cols, y_hand, color=NAVY, lw=2.2, marker="s", ms=5.5, label="Zero-shot hand.npz paddle (moves LEFT at t=0 -> catches col 4)", zorder=6)
    ax1.plot(myop_cols, y_myop, color=CRIMSON, lw=1.9, ls="--", marker="^", ms=5.5, label="Myopic ball-chaser (lured RIGHT to col 9 -> misses at col 5)", zorder=5)

    ax1.annotate(
        "t=0: Ball at (1,8) moving RIGHT,\nbut W1 u0 reads wall bounce c*=4.0\n-> hand.npz moves LEFT (col 7->4)!",
        xy=(7, 8.55),
        xytext=(0.25, 3.1),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.95),
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=NAVY, lw=0.75),
        zorder=8,
    )
    ax1.annotate(
        "t=6 Intercept (row 8, col 4):\nhand.npz waiting at col 4 (CAUGHT)\nMyopic lagging at col 5 (MISSED)",
        xy=(4, 8.0),
        xytext=(0.25, 6.2),
        arrowprops=dict(arrowstyle="->", color=SAGE, lw=0.95),
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=SAGE, lw=0.75),
        zorder=8,
    )

    ax1.axhspan(8.35, 9.55, color="#e8e5dc", alpha=0.55, zorder=1)
    ax1.text(0.15, 8.85, "Paddle Band\n(t=0 -> t=6)", fontsize=6.9, color=SLATE)
    ax1.axvline(9.0, color=BORDER_COLOR, lw=1.5, ls=":")
    ax1.text(9.05, 0.6, "Right\nWall", fontsize=6.9, color=SLATE)

    ax1.set_xlim(-0.2, 9.8)
    ax1.set_ylim(9.7, 0.2)  # Invert y so row 0 is top, row 9 (paddle) is bottom
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("Breakout grid column c (0..9)")
    ax1.set_ylabel("Breakout grid row r (0 = top, 8 = intercept, 9 = paddle)")
    ax1.set_title("B. Scenario 2: Breakout 815-Weight Lookahead (t = 0..6)", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper left", fontsize=6.9)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.13)
    out_path = ASSETS_DIR / "fig6_step_by_step_simulation.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def main() -> None:
    configure_matplotlib_style()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    G, communities, labels = run_graphify_pipeline()
    print(f"Graphify pipeline complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")

    f1 = render_fig1_breakout_weight_reading()
    f2 = render_fig2_char_lm_circuit_graph()
    f3 = render_fig3_kinetix_subgame_graft()
    f4 = render_fig4_sycophancy_and_qwen_loops()
    f5 = render_fig5_graphify_architecture_map(G, communities, labels)
    f6 = render_fig6_step_by_step_simulation()
    for p in (f1, f2, f3, f4, f5, f6):
        print(f"Generated SVG: {p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
