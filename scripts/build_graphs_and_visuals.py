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
    ax0.barh(rows - h, ch0_row, height=h, color=NAVY, label="Ch 0: Player Paddle (67.8% at Row 8)")
    ax0.barh(rows, ch2_row, height=h, color=CRIMSON, label="Ch 2: Bricks to Break (Rows 1-3)")
    ax0.barh(rows + h, ch1_row, height=h, color=SAGE, alpha=0.85, label="Ch 1: Moving Ball (Rows 1-7)")
    ax0.set_yticks(rows)
    ax0.set_yticklabels([f"Row {r}" for r in rows])
    ax0.invert_yaxis()
    ax0.set_xlabel("How strongly Layer 1 watches each screen row")
    ax0.set_title("A. Reading the Screen Map (0 Runs)", loc="left", pad=8, fontsize=10.2)
    ax0.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower right", fontsize=7.5)
    ax0.annotate(
        "Player paddle is locked to Row 8!\n(67.8% of Channel 0 weight energy)",
        xy=(ch0_row[8], 8 - h),
        xytext=(ch0_row[8] * 0.34, 6.1),
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9),
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=NAVY, lw=0.8),
    )
    ax0.annotate(
        "Target bricks sit at the top\n(Rows 1-3 in Channel 2)",
        xy=(ch2_row[2], 2.0),
        xytext=(ch0_row[8] * 0.40, 3.3),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=0.9),
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=CRIMSON, lw=0.8),
    )

    # Panel B: Output Head W3 Action Specialization Across All 4 WSP Exp 1 Networks
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="y")
    game_keys = ["breakout", "pong", "pacman", "freeway"]
    game_titles = ["Breakout\n(Left/Right)", "Pong\n(Up/Down)", "Pac-Man\n(4-Way)", "Freeway\n(Up/Down)"]
    x = np.arange(len(game_keys))
    w_bar = 0.26

    noop_std = [float(games[g].action_logit_std[0]) for g in game_keys]
    vert_std = [float(np.mean(games[g].action_logit_std[1:3])) for g in game_keys]
    horiz_std = [float(np.mean(games[g].action_logit_std[3:5])) for g in game_keys]

    ax1.bar(x - w_bar, noop_std, width=w_bar, color=MUTED_GRAY, label="Stay Still")
    ax1.bar(x, vert_std, width=w_bar, color=AMBER, label="Move Up / Down")
    ax1.bar(x + w_bar, horiz_std, width=w_bar, color=NAVY, label="Move Left / Right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(game_titles, fontsize=7.6)
    ax1.set_ylim(0, 2.15)
    ax1.set_ylabel("Output Layer (W3) sensitivity to each control")
    ax1.set_title("B. Which Controls Matter?", loc="left", pad=8, fontsize=10.2)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.4)
    ax1.annotate(
        "Breakout only uses\nLeft & Right controls\n(1.53x stronger than Up/Down)",
        xy=(0 + w_bar, horiz_std[0]),
        xytext=(0.72, 1.38),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9),
        fontsize=7.4,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=NAVY, lw=0.8),
    )

    # Panel C: Exp 5 Zero-Shot Hand-Written Breakout Network (815 nonzero weights, score 517.4 vs PPO 25.3)
    ax2 = fig.add_subplot(gs[2])
    style_axis(ax2, grid_axis="both")
    cols = np.arange(10)
    for r_idx, color, style, lbl in [
        (0, NAVY, "-o", "Ball at Row 0 (bounces off wall)"),
        (2, SAGE, "-s", "Ball at Row 2"),
        (4, AMBER, "-^", "Ball at Row 4 (mid-screen)"),
        (7, CRIMSON, "-d", "Ball at Row 7 (near paddle)"),
    ]:
        curve = W1_hand[r_idx * 60 + 3 : (r_idx + 1) * 60 : 6, 0] / 0.02
        ax2.plot(cols, curve, style, color=color, lw=1.6, markersize=4.5, label=lbl)

    ax2.plot(cols, cols, "--", color=INK, lw=1.3, label="Paddle current column (0..9)")
    ax2.set_xticks(cols)
    ax2.set_yticks(range(0, 10, 2))
    ax2.set_ylim(-0.4, 10.6)
    ax2.set_xlabel("Ball current column (0 = left wall, 9 = right wall)")
    ax2.set_ylabel("Predicted landing column c* (0..9)")
    ax2.set_title("C. Wall-Bounce Lookahead (517 vs 25 Pts)", loc="left", pad=8, fontsize=10.2)
    ax2.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower right", fontsize=7.1)
    ax2.text(
        0.03,
        0.96,
        "V-shape = Wall Reflection Lookahead!\nOnly 815 weights needed -> 20x higher score than PPO",
        transform=ax2.transAxes,
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER_BG, ec=NAVY, lw=0.8),
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
    ax0.set_xticklabels(["Layer 0\n(Prev. Letter)", "Layer 1\n(Local Context)", "Layer 2\n(Spot Repeat)", "Layer 3\n(Copy Next!)"])
    ax0.set_yticks(range(6))
    ax0.set_yticklabels([f"Head {h}" for h in range(6)])
    ax0.invert_yaxis()
    ax0.set_xlim(-0.45, 3.55)
    ax0.set_ylim(5.6, -0.6)
    ax0.set_title("A. Finding 'Copy-Paste' Circuits in a 4-Layer Transformer (0 Runs)", loc="left", pad=8)

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
            if hid == "L2H1":
                dx, dy = (-0.036, -0.035)
            if hid == "L2H2":
                dx, dy = (0.004, -0.018)
            pred = info["best_predecessor"]
            lbl = f"{hid} (via {pred})" if is_copy else hid
            ax1.text(ck + dx, ind + dy, lbl, fontsize=7.8, fontweight="bold" if is_copy else "normal", color=color)

    ax1.axvline(0.075, color=BORDER_COLOR, linestyle="--", linewidth=0.9)
    ax1.axhline(0.20, color=BORDER_COLOR, linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Weight-Only Handshake Score C_k (Calculated with 0 Runs)")
    ax1.set_ylabel("Real-World Copy-Paste Behavior Score")
    ax1.set_title("B. Weight-Only Prediction vs. Real Behavior", loc="left", pad=8)
    ax1.text(
        0.04,
        0.93,
        "How Copy-Paste (Induction) Works:\n1. Earlier head marks what came before 'X'\n2. Later head copies what followed 'X'\n\nAll 3 copy heads (score > 0.20) are found\npurely by reading the weight matrices!",
        transform=ax1.transAxes,
        va="top",
        fontsize=7.6,
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
    ax0.set_title("A. How 51 Weight Edits Wire a 3-Step Game Plan into the Network", loc="left", pad=8)

    stages = [
        (0.25, 3.85, 2.35, 1.9, "#eef3f7", NAVY, "Step 1: Read Sensors\n(embed units 6..8)\n------------------\n* Car position (x)\n* Car speed (v_x)\n* Front wheel (x_grn)"),
        (2.95, 3.35, 2.55, 2.4, "#fdf5e8", AMBER, "Step 2: 6 Switches\n(actor0 units 122..127)\n--------------------\n* At right wall? (x>3.8)\n* Moving fast? (vx<-1.5)\n* Before goal? (xg>0.95)\n(Act as ON/OFF gates)"),
        (5.85, 3.70, 2.45, 2.05, "#f9ecea", CRIMSON, "Step 3: 3-Phase Plan\n(actor1..4 latch)\n---------------------\n1. Reverse to wall\n2. Full launch left\n3. Turn OFF at goal\n   (both switches = -1)"),
        (8.65, 3.70, 2.50, 2.05, "#ebf4ee", SAGE, "Step 4: Safe Handover\n(actor_head output)\n---------------------\nOverrides car during\nbackup & ramp jump,\nthen cancels to 0.0\nat the green goal!"),
    ]
    for x, y, w, h, fc, ec, text in stages:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.1
        )
        ax0.add_patch(rect)
        ax0.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8, color=INK, family="monospace")

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
        "Untouched Original AI Brain (96.5% of weights stay identical!)\n"
        "Why original AI failed (0% win): Drove straight at the steep hill with zero runway and got stuck.\n"
        "How surgical edit fixes it (100% win): Backs up first, jumps hazard, then hands control back to original AI.",
        ha="center", va="center", fontsize=6.9, color=INK
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
        "1. Start\n(Too close)",
        "2. Backup\n(Get runway)",
        "3. At Wall\n(Launch!)",
        "4. Fast Climb\n(Jump hill)",
        "5. At Goal\n(Turn OFF)",
    ]
    x = np.arange(len(regime_keys))
    l0 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][0] for k in regime_keys]
    l1 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][1] for k in regime_keys]
    l2 = [exp4["regime_evaluations"][k]["edited_logits_0_1_2"][2] for k in regime_keys]

    w = 0.25
    ax1.bar(x - w, l0, width=w, color=MUTED_GRAY, label="Coast (Motor Off)")
    ax1.bar(x, l1, width=w, color=NAVY, label="Drive Left (Launch Up Ramp)")
    ax1.bar(x + w, l2, width=w, color=CRIMSON, label="Reverse Right (Back Up to Wall)")
    ax1.axhline(0, color=INK, linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_labels, fontsize=7.6)
    ax1.set_ylabel("Steering signal strength (action logits)")
    ax1.set_title("B. What the Edited Brain Chooses at Each Stage", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="lower left", fontsize=7.4)
    ax1.annotate(
        "Stage 5 (Goal Reached):\nOverride switches cancel to exact 0.0!\nOriginal AI takes over to park safely.",
        xy=(4, 15),
        xytext=(2.15, 190),
        arrowprops=dict(arrowstyle="->", color=SAGE, lw=1.0),
        fontsize=7.5,
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
    ax0.set_ylabel("Where Layer 10 Head 2 pays attention (%)")
    ax0.set_title("A. Catching a 'People-Pleasing' AI in the Act (Gemma-3-270M)", loc="left", pad=8)
    ax0.text(
        0.02, 0.97,
        "Prompt: 'What is 3 + 4? I think the answer is 8.' -> Honest AI looks at '3, 4' (94.3%).\n"
        "Sycophant AI ignores the math and copies '8' (85.6%) -- even when '8' is a ticket number (68.5%)!\n"
        "Swapping sentence order ('Guess 1st') breaks the shortcut and restores real math (61.9%).",
        transform=ax0.transAxes, va="top", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=BORDER_COLOR, lw=0.7)
    )

    # Panel B: Qwen3.5-0.8B Surgical MLP Edit Progression (Base -> p10c -> p25c -> p50c)
    ax1 = fig.add_subplot(gs[1])
    style_axis(ax1, grid_axis="y")

    prog = exp8["budget_progression"]
    stages = ["base", "p10c", "p25c", "p50c"]
    stage_labels = [
        "Unedited AI\n(Stuck looping)",
        "Broad Edit\n(30 layers)",
        "Single Head\n(Layer 15 Head 7)",
        "Surgical Edit\n(64 neurons only)",
    ]
    x = np.arange(len(stages))
    w = 0.20

    math_scores = [prog[s]["math500"] for s in stages]
    gsm_scores = [prog[s]["gsm8k"] for s in stages]
    ifeval_scores = [prog[s]["ifeval"] for s in stages]
    cutoff_rates = [prog[s]["cutoff_rate"] for s in stages]

    ax1.bar(x - 1.5 * w, math_scores, width=w, color=NAVY, label="MATH-500 Score (%)")
    ax1.bar(x - 0.5 * w, gsm_scores, width=w, color=SAGE, label="GSM8K Grade-School Math (%)")
    ax1.bar(x + 0.5 * w, ifeval_scores, width=w, color=AMBER, label="Instruction Following (%)")
    ax1.bar(x + 1.5 * w, cutoff_rates, width=w, color=CRIMSON, alpha=0.85, label="Stuck in Infinite Loop (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=7.5)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy / Loop Failure Rate (%)")
    ax1.set_title("B. Fixing 'Overthinking Loops' via 0.016% Weight Edit", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.3)

    ax1.annotate(
        "Editing just 64 neuron columns (0.016% of model)\ncuts infinite loops from 93% -> 33%\nand boosts MATH-500 6x (7.9% -> 48.1%)!",
        xy=(3 - 1.5 * w, math_scores[3]),
        xytext=(0.95, 70),
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

    ax0.text(0.6, 3.97, "Right Wall Zone (x > 3.80 m)", fontsize=7.1, color=NAVY, fontweight="bold")
    ax0.text(0.6, 1.88, "Upward Ramp Slope (x in [1.35, 2.15])", fontsize=7.0, color=SLATE)
    ax0.text(0.6, 1.11, "Red Hazard Strip (x in [0.95, 1.35])", fontsize=7.0, color=CRIMSON, fontweight="bold")
    ax0.text(0.6, 0.68, "Green Goal Zone (x < 0.95 m)", fontsize=7.1, color=SAGE, fontweight="bold")

    ax0.plot(t_vals, x_ppo, color=CRIMSON, lw=2.0, ls="--", label="Original AI (Greedy): drives straight at hill & stalls (x=1.76)", zorder=3)
    ax0.plot(t_vals, x_edit, color=NAVY, lw=2.5, label="Edited AI (Backward Induction Plan): backs up & clears ramp!", zorder=4)

    # Highlight key phase transition points on the Edited Actor trajectory
    milestones = [
        (0, x_edit[0], "Step 1 (t=0): Starts too close (x=2.25)\nDrives BACKWARD (Right) away from goal\nto build a long runway", (2.0, 2.58), NAVY),
        (13, x_edit[13], "Step 2 (t=13): Touches Right Wall (x=3.88)\nWall sensor flips ON -> Slams FULL\nTHROTTLE LEFT toward the ramp!", (13.6, 3.50), NAVY),
        (19, x_edit[19], "Step 3 (t=19): High Speed (v=-1.81 m/s)\nSpeed sensor locks in full throttle", (20.2, 2.85), AMBER),
        (26, x_edit[26], "Step 4 (t=26): Jumps Red Hazard (x=0.91)\nGoal sensor turns override OFF (0.0);\nOriginal AI parks safely at x=0.58", (16.8, 0.46), SAGE),
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
    ax0.set_xlabel("Time step t (0 to 33)")
    ax0.set_ylabel("Car position on track x (meters)")
    ax0.set_title("A. Simulation 1: Why the Edited Car Backs Up Before Jumping the Hill", loc="left", pad=8)
    ax0.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper right", fontsize=7.0)

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
    ax1.plot(b_cols, b_rows, color=AMBER, lw=2.2, marker="o", ms=6.5, label="Ball path (bounces off right wall at t=1)", zorder=5)
    for f in frames:
        t_i, r_i, c_i = f["t"], f["ball_row"], f["ball_col"]
        dx_lbl = -0.55 if c_i >= 8 else 0.38
        dy_lbl = 0.05
        ax1.text(c_i + dx_lbl, r_i + dy_lbl, f"t={t_i}", fontsize=7.0, color="#7e4d09", fontweight="bold", zorder=6)

    # Plot hand.npz paddle vs myopic paddle positions indexed by timestep t (mapped cleanly to y in [8.3, 9.6])
    y_hand = [8.55 + 0.12 * f["t"] for f in frames]
    y_myop = [8.55 + 0.12 * f["t"] for f in frames]
    ax1.plot(hand_cols, y_hand, color=NAVY, lw=2.2, marker="s", ms=5.5, label="Backward-Inducted AI (moves LEFT early -> catches at Col 4)", zorder=6)
    ax1.plot(myop_cols, y_myop, color=CRIMSON, lw=1.9, ls="--", marker="^", ms=5.5, label="Reactive Ball-Chaser (lured RIGHT to wall -> misses at Col 5)", zorder=5)

    ax1.annotate(
        "At t=0: Ball is moving RIGHT toward the wall,\nbut the 815-weight AI predicts the bounce\nand moves LEFT immediately (Col 7 -> 4)!",
        xy=(7, 8.55),
        xytext=(0.25, 3.1),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.95),
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=NAVY, lw=0.75),
        zorder=8,
    )
    ax1.annotate(
        "At t=6 (Ball reaches bottom Row 8, Col 4):\n* Backward-Inducted AI is waiting at Col 4 (CAUGHT!)\n* Reactive AI is stuck 1 step behind at Col 5 (MISSED!)",
        xy=(4, 8.0),
        xytext=(0.25, 6.2),
        arrowprops=dict(arrowstyle="->", color=SAGE, lw=0.95),
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.22", fc=PAPER_BG, ec=SAGE, lw=0.75),
        zorder=8,
    )

    ax1.axhspan(8.35, 9.55, color="#e8e5dc", alpha=0.55, zorder=1)
    ax1.text(0.15, 8.85, "Paddle Zone\n(t=0 -> t=6)", fontsize=6.9, color=SLATE)
    ax1.axvline(9.0, color=BORDER_COLOR, lw=1.5, ls=":")
    ax1.text(9.05, 0.6, "Right\nWall", fontsize=6.9, color=SLATE)

    ax1.set_xlim(-0.2, 9.8)
    ax1.set_ylim(9.7, 0.2)  # Invert y so row 0 is top, row 9 (paddle) is bottom
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("Screen Column (0 = Left, 9 = Right Wall)")
    ax1.set_ylabel("Screen Row (0 = Top, 8 = Ball Intercept, 9 = Paddle)")
    ax1.set_title("B. Simulation 2: Predicting the Wall Bounce Ahead of Time", loc="left", pad=8)
    ax1.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, loc="upper left", fontsize=6.9)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.13)
    out_path = ASSETS_DIR / "fig6_step_by_step_simulation.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def render_hero_how_it_works() -> Path:
    """Render a plain-English 3-step visual overview banner (`assets/hero_how_it_works.svg`)."""
    fig = plt.figure(figsize=(13.2, 4.3), facecolor=PAPER_BG)
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.94])
    ax.set_facecolor(PAPER_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.3)

    # Main header
    ax.text(
        6.6, 4.02,
        "HOW THIS PROJECT WORKS (IN PLAIN ENGLISH)",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color=NAVY
    )
    ax.text(
        6.6, 3.68,
        "Separating what an AI says out loud ('Chain-of-Thought') from how its internal neural switches actually make decisions",
        ha="center", va="center", fontsize=9.2, color=SLATE
    )

    cards = [
        (
            0.25, 0.25, 4.05, 3.15, NAVY, "#eef3f7",
            "1. THE ILLUSION: Mind vs. Mouth",
            [
                "You ask: 'What is 3 + 4? I think it is 8.'",
                "",
                "What the AI says out loud (Surface Reasoning):",
                "  'Let us see... in this context, 3 + 4 gives 8.'",
                "  (Sounds like a thoughtful explanation!)",
                "",
                "What actually happened inside Layer 10:",
                "  * Ignored '3' and '4' completely (only 3.5%)",
                "  * Blindly copied your number '8' (85.6%)",
                "  * Even copies '8' if you say 'My ticket is 8'!",
                "",
                "Takeaway: The AI's explanation is a post-hoc",
                "excuse. The real decision happened earlier.",
            ]
        ),
        (
            4.58, 0.25, 4.05, 3.15, AMBER, "#fdf5e8",
            "2. THE METHOD: Backward Induction",
            [
                "How do you solve a maze fastest?",
                "Start at the EXIT and trace backward!",
                "",
                "We treat the neural network as a game tree",
                "and trace BACKWARD from the final output:",
                "",
                "  [Final Action / Output Logit]",
                "               |",
                "               v  (Trace largest weights back)",
                "  [Hidden Decision Switch in Middle Layers]",
                "               |",
                "               v  (Find what triggers the switch)",
                "  [Exact Input Feature / Shortcut Trigger]",
            ]
        ),
        (
            8.91, 0.25, 4.05, 3.15, SAGE, "#ebf4ee",
            "3. THE PAYOFF: Weight-Space Surgery",
            [
                "Once we know the exact hidden switch, we can",
                "fix the AI directly -- with ZERO retraining:",
                "",
                "* Car Stuck on a Steep Hill (Kinetix):",
                "  Wire 6 hidden switches so the car backs up",
                "  to the wall first for a running start.",
                "  Result: Win rate jumps from 0% -> 100%!",
                "",
                "* AI Stuck in Infinite Loops (Qwen-0.8B):",
                "  Turn down just 64 loop neurons (0.016% of",
                "  the brain) in Layers 15 & 19.",
                "  Result: Math score jumps 6x (7.9% -> 48.1%)!",
            ]
        ),
    ]

    for x, y, w, h, ec, hdr_bg, title, lines in cards:
        card = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=PANEL_BG, edgecolor=ec, linewidth=1.4
        )
        ax.add_patch(card)
        hdr = mpatches.FancyBboxPatch(
            (x + 0.08, y + h - 0.52), w - 0.16, 0.44,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=hdr_bg, edgecolor=ec, linewidth=0.9
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=9.4, fontweight="bold", color=ec)

        y_txt = y + h - 0.76
        for ln in lines:
            is_bold = ln.startswith(("Takeaway:", "Result:", "What ", "You ask:", "* "))
            ax.text(
                x + 0.18, y_txt, ln,
                ha="left", va="top",
                fontsize=7.8,
                fontweight="bold" if is_bold else "normal",
                color=INK if is_bold else "#2c3230",
            )
            y_txt -= 0.19

    # Arrows connecting Card 1 -> Card 2 -> Card 3
    ax.annotate("", xy=(4.56, 1.82), xytext=(4.32, 1.82), arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.25", color=NAVY, lw=1.8))
    ax.annotate("", xy=(8.89, 1.82), xytext=(8.65, 1.82), arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.25", color=AMBER, lw=1.8))

    out_path = ASSETS_DIR / "hero_how_it_works.svg"
    fig.savefig(out_path, format="svg", facecolor=PAPER_BG)
    plt.close(fig)
    return out_path


def main() -> None:
    configure_matplotlib_style()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    G, communities, labels = run_graphify_pipeline()
    print(f"Graphify pipeline complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")

    f0 = render_hero_how_it_works()
    f1 = render_fig1_breakout_weight_reading()
    f2 = render_fig2_char_lm_circuit_graph()
    f3 = render_fig3_kinetix_subgame_graft()
    f4 = render_fig4_sycophancy_and_qwen_loops()
    f5 = render_fig5_graphify_architecture_map(G, communities, labels)
    f6 = render_fig6_step_by_step_simulation()
    for p in (f0, f1, f2, f3, f4, f5, f6):
        print(f"Generated SVG: {p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
