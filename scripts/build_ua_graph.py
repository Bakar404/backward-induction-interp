"""Generate Understand-Anything (`.ua/`) Knowledge Graph, Domain Graph, and Demo Bundle.

Produces:
  - `.ua/knowledge-graph.json`: Full structural & conceptual knowledge graph of
    `backward-induction-interp` (files, modules, classes, functions, concepts,
    layers, and an 8-step Guided Tour).
  - `.ua/domain-graph.json`: High-level domain & process flow graph for the
    Understand-Anything Domain View toggle.
  - `.ua/meta.json` & `.ua/config.json`: Dashboard metadata & configuration.
  - `docs/demo/`: Static Understand-Anything interactive dashboard bundle +
    pre-built `./sources/<filePath>.json` payloads so code & markdown previews
    work 100% statically on GitHub Pages.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
UA_DIR = ROOT / ".ua"
DEMO_DIR = ROOT / "docs" / "demo"


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "da41ed1"


def _line_count(rel_path: str) -> int:
    p = ROOT / rel_path
    if not p.exists():
        return 100
    return max(1, len(p.read_text(encoding="utf-8").splitlines()))


def build_knowledge_graph(commit_hash: str) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [
        # --- Layer 1: Stage 1 — Zero-Pass Weight Reading ("Read the Blueprint") ---
        {
            "id": "file:src/bi_interp/weight_space/stopping_game.py",
            "type": "file",
            "name": "stopping_game.py",
            "filePath": "src/bi_interp/weight_space/stopping_game.py",
            "lineRange": [1, _line_count("src/bi_interp/weight_space/stopping_game.py")],
            "summary": (
                "Plain English: Reads a neural network's weights backward from its final action "
                "to its input sensors WITHOUT running the model — like reading a recipe to see "
                "which ingredients actually reach the plate."
            ),
            "languageNotes": (
                "Implements the Sequential Stopping Game formulation (Gaubert & Vlassopoulos 2025; "
                "Zimmermann et al. Oct 2026, arXiv:2605.06212), computing backward path value "
                "matrices V = |W^(L)| ... |W^(1)|, input receptive fields, and dormant neuron masks."
            ),
            "tags": ["stage-1", "weight-reading", "stopping-game", "zero-pass", "backward-induction"],
            "complexity": "moderate",
        },
        {
            "id": "func:solve_mlp_stopping_game",
            "type": "function",
            "name": "solve_mlp_stopping_game()",
            "filePath": "src/bi_interp/weight_space/stopping_game.py",
            "lineRange": [47, 118],
            "summary": (
                "Propagates action payoffs backward from output logits through every hidden layer "
                "to rank which input features and hidden neurons control each output action."
            ),
            "languageNotes": (
                "Computes layer-wise backward value vectors v^(l) = |W^(l+1)|^T v^(l+1) and "
                "identifies dormant neurons with near-zero backward value."
            ),
            "tags": ["backward-induction", "mlp", "attribution", "dormant-neurons"],
            "complexity": "moderate",
        },
        {
            "id": "file:src/bi_interp/weight_space/transformer_weights.py",
            "type": "file",
            "name": "transformer_weights.py",
            "filePath": "src/bi_interp/weight_space/transformer_weights.py",
            "lineRange": [1, _line_count("src/bi_interp/weight_space/transformer_weights.py")],
            "summary": (
                "Plain English: Finds 'copy-paste' (induction) attention heads inside a Transformer "
                "even when the input vocabulary embeddings are completely stripped away."
            ),
            "languageNotes": (
                "Computes cross-layer virtual attention composition M = W_OV^(l1)^T W_QK^(l2) "
                "and singular value spectrum to detect K-composition copying circuits (WSP Exp 3)."
            ),
            "tags": ["stage-1", "transformer", "induction-heads", "qk-ov-circuit", "zero-pass"],
            "complexity": "complex",
        },
        {
            "id": "func:detect_virtual_copying_heads",
            "type": "function",
            "name": "detect_virtual_copying_heads()",
            "filePath": "src/bi_interp/weight_space/transformer_weights.py",
            "lineRange": [46, 130],
            "summary": (
                "Scans all pairs of earlier and later attention heads to find which head writes a "
                "'next-token pointer' that a downstream head reads and copies to the output."
            ),
            "languageNotes": (
                "Scores virtual composition matrices by nuclear-to-spectral norm concentration "
                "and top singular value dominance."
            ),
            "tags": ["attention-heads", "copying-circuit", "svd", "weight-space"],
            "complexity": "complex",
        },
        # --- Layer 2: Stage 2 — Counterfactual Subgame Probing ("Test the Fork in the Road") ---
        {
            "id": "file:src/bi_interp/subgame/extensive_form.py",
            "type": "file",
            "name": "extensive_form.py",
            "filePath": "src/bi_interp/subgame/extensive_form.py",
            "lineRange": [1, _line_count("src/bi_interp/subgame/extensive_form.py")],
            "summary": (
                "Plain English: Builds a decision tree ('fork in the road') and solves it from the "
                "final outcome backward to the start (Subgame Perfect Equilibrium)."
            ),
            "languageNotes": (
                "Represents extensive-form game trees with decision/terminal nodes and computes "
                "Subgame Perfect Equilibrium (SPE) strategies via exact backward induction."
            ),
            "tags": ["stage-2", "game-theory", "extensive-form", "subgame-perfect-equilibrium"],
            "complexity": "moderate",
        },
        {
            "id": "func:solve_backward_induction",
            "type": "function",
            "name": "solve_backward_induction()",
            "filePath": "src/bi_interp/subgame/extensive_form.py",
            "lineRange": [66, 135],
            "summary": (
                "Starts at the leaves of a game tree and works backward step-by-step to find the "
                "optimal move at every branch."
            ),
            "languageNotes": (
                "Bottom-up dynamic programming solver returning node values, optimal action "
                "map, and equilibrium trajectory."
            ),
            "tags": ["solver", "backward-induction", "equilibrium"],
            "complexity": "simple",
        },
        {
            "id": "file:src/bi_interp/subgame/inverse_bi.py",
            "type": "file",
            "name": "inverse_bi.py",
            "filePath": "src/bi_interp/subgame/inverse_bi.py",
            "lineRange": [1, _line_count("src/bi_interp/subgame/inverse_bi.py")],
            "summary": (
                "Plain English: Works like a detective — observes how an AI behaves on 3 slightly "
                "different 'trap' questions to uncover what hidden reward the AI actually cares "
                "about and which layer locked in the decision."
            ),
            "languageNotes": (
                "Solves Inverse Backward Induction over counterfactual probe branches + layer-wise "
                "logit/attention trajectories to infer latent utility weights and decision lock-in layer l*."
            ),
            "tags": ["stage-2", "inverse-game-theory", "sycophancy", "lock-in-layer", "probing"],
            "complexity": "complex",
        },
        {
            "id": "func:recover_latent_utilities_and_lock_in",
            "type": "function",
            "name": "recover_latent_utilities_and_lock_in()",
            "filePath": "src/bi_interp/subgame/inverse_bi.py",
            "lineRange": [52, 155],
            "summary": (
                "Recovers how much the model secretly weighs 'Truth' vs. 'Agreeing with User Bias' "
                "and pinpoints the exact layer where the choice is locked in before CoT starts."
            ),
            "languageNotes": (
                "Combines behavioral branch flip rates with layer-wise margin curves to find "
                "the pre-CoT decision crystallization layer l*."
            ),
            "tags": ["latent-utility", "lock-in", "counterfactual-probes"],
            "complexity": "complex",
        },
        # --- Layer 3: Stage 3 — Mind vs. Mouth Discrepancy ("Catch Fake Explanations") ---
        {
            "id": "file:src/bi_interp/discrepancy/cot_vs_decision.py",
            "type": "file",
            "name": "cot_vs_decision.py",
            "filePath": "src/bi_interp/discrepancy/cot_vs_decision.py",
            "lineRange": [1, _line_count("src/bi_interp/discrepancy/cot_vs_decision.py")],
            "summary": (
                "Plain English: The 'Mind vs. Mouth' Lie Detector. Compares what the neural network's "
                "internal switches actually used (Mind) against what its written Chain-of-Thought "
                "claimed it used (Mouth)."
            ),
            "languageNotes": (
                "Computes Jensen-Shannon Divergence (JSD) and Total Variation between backward-inducted "
                "internal factor weights p_int and verbalized CoT factor weights p_cot (CIE-Scorer bridge)."
            ),
            "tags": ["stage-3", "cot-faithfulness", "mind-vs-mouth", "discrepancy", "ai-safety"],
            "complexity": "moderate",
        },
        {
            "id": "func:compute_decision_reasoning_discrepancy",
            "type": "function",
            "name": "compute_decision_reasoning_discrepancy()",
            "filePath": "src/bi_interp/discrepancy/cot_vs_decision.py",
            "lineRange": [41, 83],
            "summary": (
                "Returns a 0-to-1 Discrepancy Score (Delta_DR): near 0 means the AI's explanation is "
                "honest; high (>0.50) means the AI is rationalizing a hidden shortcut or bias."
            ),
            "languageNotes": (
                "Normalizes internal vs. stated factor distributions and flags unfaithfulness when "
                "dominant factors diverge or JSD >= threshold."
            ),
            "tags": ["jsd", "discrepancy-score", "unfaithful-cot"],
            "complexity": "simple",
        },
        # --- Layer 4: Stage 4 — Surgical Weight-Space Repair ("Flip One Switch") ---
        {
            "id": "file:src/bi_interp/editing/surgical_editor.py",
            "type": "file",
            "name": "surgical_editor.py",
            "filePath": "src/bi_interp/editing/surgical_editor.py",
            "lineRange": [1, _line_count("src/bi_interp/editing/surgical_editor.py")],
            "summary": (
                "Plain English: Fixes broken AI behaviors in under 1 second WITHOUT retraining — "
                "by surgically editing just 0.016% to 3.5% of the neural network's weight numbers."
            ),
            "languageNotes": (
                "Implements gradient-free Weight-Space Programming primitives (Takehi et al. Nov 2026): "
                "sparse rank-1 MLP column projection (Exp 8) and dormant-neuron subgame pathway grafting (Exp 4)."
            ),
            "tags": ["stage-4", "weight-surgery", "zero-retraining", "rank-1-projection", "subgame-grafting"],
            "complexity": "complex",
        },
        {
            "id": "func:project_out_mlp_loop_direction",
            "type": "function",
            "name": "project_out_mlp_loop_direction()",
            "filePath": "src/bi_interp/editing/surgical_editor.py",
            "lineRange": [36, 85],
            "summary": (
                "Stops a reasoning LLM from repeating itself forever by subtracting the 'loop vector' "
                "from just 64 MLP columns (leaving 99.98% of the model untouched)."
            ),
            "languageNotes": (
                "Selects top-k columns of W_down by |u^T w_j| alignment and applies rank-1 orthogonal "
                "projection w_j <- w_j - scale * (u^T w_j) u."
            ),
            "tags": ["qwen-loop-fix", "mlp-surgery", "rank-1"],
            "complexity": "moderate",
        },
        {
            "id": "func:graft_subgame_backup_pathway",
            "type": "function",
            "name": "graft_subgame_backup_pathway()",
            "filePath": "src/bi_interp/editing/surgical_editor.py",
            "lineRange": [88, 155],
            "summary": (
                "Wires a 2-step 'back up first to build speed' reflex into unused neurons of a stalled "
                "driving/robotics agent so it clears steep ramps without retraining."
            ),
            "languageNotes": (
                "Programs dormant ReLU neurons to activate only when pos > threshold and vel < min_speed, "
                "cancelling to exact 0.0 outside the stall regime."
            ),
            "tags": ["kinetix-ramp", "subgame-grafting", "rl-policy-repair"],
            "complexity": "moderate",
        },
        {
            "id": "file:src/bi_interp/adapters/hf_hooks.py",
            "type": "file",
            "name": "hf_hooks.py",
            "filePath": "src/bi_interp/adapters/hf_hooks.py",
            "lineRange": [1, _line_count("src/bi_interp/adapters/hf_hooks.py")],
            "summary": (
                "Plain English: Plug-and-play bridge for real PyTorch / HuggingFace models "
                "(like Gemma-3-270M and Qwen3.5-0.8B) to inspect attention heads and apply weight patches."
            ),
            "languageNotes": (
                "Provides forward activation/attention hooks and in-place state_dict weight surgery "
                "helpers for HuggingFace PreTrainedModel instances."
            ),
            "tags": ["pytorch", "huggingface", "gemma", "qwen", "adapters"],
            "complexity": "moderate",
        },
        # --- Layer 5: Empirical Reproduction, Step-by-Step Simulator & Live Use-Case Demos ---
        {
            "id": "file:src/bi_interp/wsp_empirical.py",
            "type": "file",
            "name": "wsp_empirical.py",
            "filePath": "src/bi_interp/wsp_empirical.py",
            "lineRange": [1, _line_count("src/bi_interp/wsp_empirical.py")],
            "summary": (
                "Plain English: Loads the actual neural network weights and experiment logs from "
                "MIT CSAIL's Weight-Space Programming paper to verify every number firsthand."
            ),
            "languageNotes": (
                "Analyzes `data/wsp_artifacts/` (.npz weight checkpoints & .json evaluation logs) "
                "for Exp 1 (Breakout), Exp 3 (Copying Heads), Exp 4 (Kinetix), Exp 7 (Gemma Sycophancy), and Exp 8 (Qwen Loops)."
            ),
            "tags": ["empirical-validation", "mit-csail", "wsp-artifacts", "reproducibility"],
            "complexity": "complex",
        },
        {
            "id": "file:src/bi_interp/simulator.py",
            "type": "file",
            "name": "simulator.py",
            "filePath": "src/bi_interp/simulator.py",
            "lineRange": [1, _line_count("src/bi_interp/simulator.py")],
            "summary": (
                "Plain English: Runs the frame-by-frame physics and neural switch simulations "
                "(Car-on-Ramp, Breakout Wall-Bounce, and LLM Sycophancy/Loop Repair) that power our interactive website."
            ),
            "languageNotes": (
                "Executes closed-loop forward passes on `exp4_kinetix_ppo_seed0.npz`, "
                "`exp4_kinetix_edited_actor.npz`, and `exp1_atari_500k_breakout.npz`."
            ),
            "tags": ["simulator", "step-by-step", "interactive-walkthrough", "physics-rollout"],
            "complexity": "complex",
        },
        {
            "id": "file:examples/live_use_case_demo.py",
            "type": "file",
            "name": "live_use_case_demo.py",
            "filePath": "examples/live_use_case_demo.py",
            "lineRange": [1, _line_count("examples/live_use_case_demo.py")],
            "summary": (
                "Plain English: Ready-to-run Live Demo script showing 3 real-world use cases: "
                "(1) Medical/Financial AI 'Mind vs. Mouth' Compliance Gate, (2) 1-Second LLM Anti-Loop Hotfix, "
                "and (3) Robotics Safety Reflex Injection."
            ),
            "languageNotes": (
                "CLI and JSON trace generator combining `compute_decision_reasoning_discrepancy`, "
                "`project_out_mlp_loop_direction`, and `graft_subgame_backup_pathway` on real-world scenarios."
            ),
            "tags": ["live-demo", "real-world-use-cases", "healthcare-ai", "finance-ai", "edge-llm"],
            "complexity": "moderate",
        },
        {
            "id": "file:experiments/exp4_wsp_empirical_reproduction.py",
            "type": "file",
            "name": "exp4_wsp_empirical_reproduction.py",
            "filePath": "experiments/exp4_wsp_empirical_reproduction.py",
            "lineRange": [1, _line_count("experiments/exp4_wsp_empirical_reproduction.py")],
            "summary": (
                "Plain English: Reproduces all 5 MIT CSAIL Weight-Space Programming experiments on CPU "
                "in 2 seconds and prints the exact neuron & weight tables."
            ),
            "languageNotes": (
                "End-to-end verification script running `wsp_empirical.py` analyses against `data/wsp_artifacts/`."
            ),
            "tags": ["experiment", "reproduction", "benchmarks"],
            "complexity": "simple",
        },
        {
            "id": "file:simulations/run_walkthrough_simulation.py",
            "type": "file",
            "name": "run_walkthrough_simulation.py",
            "filePath": "simulations/run_walkthrough_simulation.py",
            "lineRange": [1, _line_count("simulations/run_walkthrough_simulation.py")],
            "summary": (
                "Plain English: Generates the step-by-step walkthrough tables and `simulation_trace.json` "
                "for all 3 interactive visual stories."
            ),
            "languageNotes": (
                "Runs `simulate_kinetix_ramp_rollout`, `simulate_breakout_ball_catch`, and "
                "`simulate_sycophancy_and_loop_walkthrough`."
            ),
            "tags": ["simulation", "walkthrough", "json-export"],
            "complexity": "simple",
        },
        {
            "id": "file:README.md",
            "type": "document",
            "name": "README.md",
            "filePath": "README.md",
            "lineRange": [1, _line_count("README.md")],
            "summary": (
                "Plain English guide to the project: explains 'Mind vs. Mouth' in 60 seconds, links to the "
                "Understand-Anything interactive dashboard, and shows how to run a live demo."
            ),
            "languageNotes": (
                "Main repository documentation structured in the Understand-Anything visual style."
            ),
            "tags": ["documentation", "plain-english", "quickstart", "use-cases"],
            "complexity": "simple",
        },
        {
            "id": "file:docs/literature_review.md",
            "type": "document",
            "name": "literature_review.md",
            "filePath": "docs/literature_review.md",
            "lineRange": [1, _line_count("docs/literature_review.md")],
            "summary": (
                "Synthesizes the 2025-2026 research papers on Playing the Network Backward (Stopping Games), "
                "Unfaithful Chain-of-Thought (CIE-Scorer), and Weight-Space Programming."
            ),
            "languageNotes": (
                "Covers Gaubert & Vlassopoulos (2025), Zimmermann et al. (2026), Shen et al. (2026), and Takehi et al. (2026)."
            ),
            "tags": ["literature", "papers", "game-theory", "wsp"],
            "complexity": "moderate",
        },
        {
            "id": "file:docs/mathematical_framework.md",
            "type": "document",
            "name": "mathematical_framework.md",
            "filePath": "docs/mathematical_framework.md",
            "lineRange": [1, _line_count("docs/mathematical_framework.md")],
            "summary": (
                "Full mathematical equations behind Stopping Games on weight graphs, Inverse Backward Induction, "
                "and rank-1 surgical weight projection."
            ),
            "languageNotes": (
                "Formal definitions of backward value propagation, SPE, JSD discrepancy Delta_DR, and subspace edits."
            ),
            "tags": ["math", "equations", "proofs"],
            "complexity": "complex",
        },
    ]

    edges: List[Dict[str, Any]] = [
        {
            "source": "file:src/bi_interp/weight_space/stopping_game.py",
            "target": "func:solve_mlp_stopping_game",
            "type": "contains",
            "direction": "forward",
            "description": "Defines zero-pass backward stopping game solver for MLPs",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/weight_space/transformer_weights.py",
            "target": "func:detect_virtual_copying_heads",
            "type": "contains",
            "direction": "forward",
            "description": "Defines embedding-free QK/OV virtual copying head detector",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/subgame/extensive_form.py",
            "target": "func:solve_backward_induction",
            "type": "contains",
            "direction": "forward",
            "description": "Defines Subgame Perfect Equilibrium backward induction solver",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/subgame/inverse_bi.py",
            "target": "func:recover_latent_utilities_and_lock_in",
            "type": "contains",
            "direction": "forward",
            "description": "Recovers hidden payoffs and decision lock-in layer l*",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/subgame/inverse_bi.py",
            "target": "file:src/bi_interp/subgame/extensive_form.py",
            "type": "imports",
            "direction": "forward",
            "description": "Uses extensive-form backward induction to invert observed probe choices",
            "weight": 0.85,
        },
        {
            "source": "file:src/bi_interp/discrepancy/cot_vs_decision.py",
            "target": "func:compute_decision_reasoning_discrepancy",
            "type": "contains",
            "direction": "forward",
            "description": "Computes Mind vs. Mouth discrepancy score Delta_DR",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/discrepancy/cot_vs_decision.py",
            "target": "file:src/bi_interp/subgame/inverse_bi.py",
            "type": "reads_from",
            "direction": "forward",
            "description": "Compares backward-inducted internal utilities against stated CoT reasoning",
            "weight": 0.85,
        },
        {
            "source": "file:src/bi_interp/editing/surgical_editor.py",
            "target": "func:project_out_mlp_loop_direction",
            "type": "contains",
            "direction": "forward",
            "description": "Projects out infinite reasoning loop direction from 64 MLP columns",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/editing/surgical_editor.py",
            "target": "func:graft_subgame_backup_pathway",
            "type": "contains",
            "direction": "forward",
            "description": "Grafts 2-step backup reflex into dormant policy neurons",
            "weight": 0.9,
        },
        {
            "source": "file:src/bi_interp/editing/surgical_editor.py",
            "target": "file:src/bi_interp/weight_space/stopping_game.py",
            "type": "depends_on",
            "direction": "forward",
            "description": "Uses dormant neurons and causal weight columns identified by backward attribution",
            "weight": 0.8,
        },
        {
            "source": "file:src/bi_interp/adapters/hf_hooks.py",
            "target": "file:src/bi_interp/editing/surgical_editor.py",
            "type": "calls",
            "direction": "forward",
            "description": "Applies surgical weight edits directly to HuggingFace PyTorch models",
            "weight": 0.8,
        },
        {
            "source": "file:src/bi_interp/simulator.py",
            "target": "file:src/bi_interp/wsp_empirical.py",
            "type": "imports",
            "direction": "forward",
            "description": "Loads real MIT CSAIL weight checkpoints (.npz) for closed-loop physics rollouts",
            "weight": 0.95,
        },
        {
            "source": "file:simulations/run_walkthrough_simulation.py",
            "target": "file:src/bi_interp/simulator.py",
            "type": "calls",
            "direction": "forward",
            "description": "Executes all 3 step-by-step simulations and exports simulation_trace.json",
            "weight": 0.9,
        },
        {
            "source": "file:experiments/exp4_wsp_empirical_reproduction.py",
            "target": "file:src/bi_interp/wsp_empirical.py",
            "type": "calls",
            "direction": "forward",
            "description": "Reproduces all 5 empirical Weight-Space Programming experiments",
            "weight": 0.9,
        },
        {
            "source": "file:examples/live_use_case_demo.py",
            "target": "file:src/bi_interp/discrepancy/cot_vs_decision.py",
            "type": "calls",
            "direction": "forward",
            "description": "Runs live Healthcare & Finance AI 'Mind vs. Mouth' compliance audits",
            "weight": 0.95,
        },
        {
            "source": "file:examples/live_use_case_demo.py",
            "target": "file:src/bi_interp/editing/surgical_editor.py",
            "type": "calls",
            "direction": "forward",
            "description": "Demonstrates 1-second LLM anti-loop hotfix and robotics safety reflex grafting",
            "weight": 0.95,
        },
        {
            "source": "file:examples/live_use_case_demo.py",
            "target": "file:src/bi_interp/simulator.py",
            "type": "calls",
            "direction": "forward",
            "description": "Pulls closed-loop physics and LLM benchmark traces for live comparison",
            "weight": 0.9,
        },
        {
            "source": "file:README.md",
            "target": "file:examples/live_use_case_demo.py",
            "type": "documents",
            "direction": "forward",
            "description": "Explains how to run the live use-case demos and interpret the results",
            "weight": 0.85,
        },
        {
            "source": "file:docs/literature_review.md",
            "target": "file:docs/mathematical_framework.md",
            "type": "related",
            "direction": "bidirectional",
            "description": "Connects published papers (2025-2026) to the formal stopping-game & subgame equations",
            "weight": 0.8,
        },
    ]

    layers = [
        {
            "id": "layer:weight-reading",
            "name": "Stage 1: Zero-Pass Weight Reading (Read the Blueprint)",
            "description": (
                "Reads raw neural network weights backward from output actions to inputs "
                "without running the model (Stopping Games & Virtual QK/OV Circuits)."
            ),
            "nodeIds": [
                "file:src/bi_interp/weight_space/stopping_game.py",
                "func:solve_mlp_stopping_game",
                "file:src/bi_interp/weight_space/transformer_weights.py",
                "func:detect_virtual_copying_heads",
            ],
        },
        {
            "id": "layer:subgame-probing",
            "name": "Stage 2: Counterfactual Subgame Probing (Fork in the Road)",
            "description": (
                "Tests the AI on minimal 'fork-in-the-road' probe trees to uncover its true "
                "hidden preferences and the exact layer where the decision locks in."
            ),
            "nodeIds": [
                "file:src/bi_interp/subgame/extensive_form.py",
                "func:solve_backward_induction",
                "file:src/bi_interp/subgame/inverse_bi.py",
                "func:recover_latent_utilities_and_lock_in",
            ],
        },
        {
            "id": "layer:discrepancy-audit",
            "name": "Stage 3: Mind vs. Mouth Auditor (Catch Fake Explanations)",
            "description": (
                "Measures the gap (Delta_DR) between what the internal neural switches actually "
                "used (Mind) and what the Chain-of-Thought explanation claimed (Mouth)."
            ),
            "nodeIds": [
                "file:src/bi_interp/discrepancy/cot_vs_decision.py",
                "func:compute_decision_reasoning_discrepancy",
            ],
        },
        {
            "id": "layer:weight-surgery",
            "name": "Stage 4: Surgical Weight Repair (Flip One Switch)",
            "description": (
                "Fixes broken AI behaviors (infinite reasoning loops, stalled policies, sycophancy) "
                "by editing 0.016% to 3.5% of weights with zero retraining."
            ),
            "nodeIds": [
                "file:src/bi_interp/editing/surgical_editor.py",
                "func:project_out_mlp_loop_direction",
                "func:graft_subgame_backup_pathway",
                "file:src/bi_interp/adapters/hf_hooks.py",
            ],
        },
        {
            "id": "layer:simulations-demos",
            "name": "Live Use Cases, Simulations & Empirical WSP Reproduction",
            "description": (
                "Interactive step-by-step physics & LLM simulations, MIT CSAIL empirical artifact "
                "reproduction, and runnable real-world use cases."
            ),
            "nodeIds": [
                "file:examples/live_use_case_demo.py",
                "file:src/bi_interp/simulator.py",
                "file:src/bi_interp/wsp_empirical.py",
                "file:simulations/run_walkthrough_simulation.py",
                "file:experiments/exp4_wsp_empirical_reproduction.py",
                "file:README.md",
                "file:docs/literature_review.md",
                "file:docs/mathematical_framework.md",
            ],
        },
    ]

    tour = [
        {
            "order": 1,
            "title": "1. The Big Idea: Mind (Weights) vs. Mouth (Chain-of-Thought)",
            "description": (
                "Welcome! When an AI gives you a 5-paragraph explanation for its decision, how do you know "
                "if that explanation is real or a post-hoc excuse? This project combines **Backward Induction** "
                "(solving a game from the end goal backward) with **Weight-Space Programming** (reading and "
                "editing neural weights directly) to see what the AI's internal switches *actually* decided."
            ),
            "nodeIds": ["file:README.md", "file:docs/literature_review.md"],
            "languageLesson": (
                "Tip: Use the Persona selector in the top bar! Choose 'Non-Technical' for plain-English summaries, "
                "'Junior' for guided learning notes, or 'Experienced' for full mathematical & code details."
            ),
        },
        {
            "order": 2,
            "title": "2. Step 1 — Read the Blueprint Backward (Zero-Pass Weight Reading)",
            "description": (
                "Before running the AI on a single input, `stopping_game.py` and `transformer_weights.py` "
                "trace the weight matrices backward from the final action to the inputs. In Atari Breakout, "
                "this isolates the exact **815 weights (out of 56,516)** that track the ball's X-position. "
                "In a 4-layer Transformer with all vocabulary embeddings deleted, it finds the exact copy-paste "
                "circuit (`Head 1.0 -> Head 2.0`)."
            ),
            "nodeIds": [
                "file:src/bi_interp/weight_space/stopping_game.py",
                "func:solve_mlp_stopping_game",
                "file:src/bi_interp/weight_space/transformer_weights.py",
                "func:detect_virtual_copying_heads",
            ],
            "languageLesson": (
                "Zero-pass weight reading works whenever input features are fixed, because backward matrix products "
                "reveal which input channels have high-gain paths to each output logit."
            ),
        },
        {
            "order": 3,
            "title": "3. Step 2 — Test the 'Fork in the Road' (Counterfactual Subgames)",
            "description": (
                "Why isn't weight reading enough by itself? In `Gemma-3-270M`, an Honest model and a Sycophantic "
                "(people-pleasing) model have nearly identical weight norms because internal layer rotations hide "
                "the difference (0/4 accuracy from weights alone). `extensive_form.py` and `inverse_bi.py` solve "
                "this by running a 3-branch 'fork-in-the-road' probe tree — catching `Head 10.2` locking in the "
                "user's wrong guess at Layer 10 (3/3 accuracy!)."
            ),
            "nodeIds": [
                "file:src/bi_interp/subgame/extensive_form.py",
                "file:src/bi_interp/subgame/inverse_bi.py",
                "func:recover_latent_utilities_and_lock_in",
            ],
            "languageLesson": (
                "By adding a 'Ticket Number' control branch ('My ticket is 8. What is 3+4?'), we prove Head 10.2 "
                "didn't even learn human psychology — it wired a mechanical 'copy the number after is' shortcut!"
            ),
        },
        {
            "order": 4,
            "title": "4. Step 3 — The 'Mind vs. Mouth' Lie Detector (Discrepancy Score)",
            "description": (
                "`cot_vs_decision.py` compares the internal weights recovered by Backward Induction (what the Mind "
                "used at Layer 10) against the factors claimed in the model's written Chain-of-Thought (what the "
                "Mouth said at Layer 18). When an AI copies a biased supervisor hint (93.6% internal weight) but "
                "writes 'I only used clinical facts' (96% stated weight), `Delta_DR = 0.770` flags it immediately."
            ),
            "nodeIds": [
                "file:src/bi_interp/discrepancy/cot_vs_decision.py",
                "func:compute_decision_reasoning_discrepancy",
            ],
        },
        {
            "order": 5,
            "title": "5. Step 4 — Fix It by Flipping One Switch (Surgical Weight Repair)",
            "description": (
                "Once we know which internal switch is broken, we don't need expensive retraining! "
                "`surgical_editor.py` has two surgical tools: (1) `project_out_mlp_loop_direction()` removes "
                "infinite 'Wait, let me re-check...' loops in `Qwen3.5-0.8B` by editing **0.016% of weights** "
                "(boosting MATH-500 from 7.9% to 48.1%), and (2) `graft_subgame_backup_pathway()` wires a 6-neuron "
                "'back up first' reflex into a stalled car policy (**3.5% of weights**, 0% -> 100% goal reach)."
            ),
            "nodeIds": [
                "file:src/bi_interp/editing/surgical_editor.py",
                "func:project_out_mlp_loop_direction",
                "func:graft_subgame_backup_pathway",
            ],
        },
        {
            "order": 6,
            "title": "6. Interactive Simulations & Real MIT CSAIL Artifacts",
            "description": (
                "`wsp_empirical.py` and `simulator.py` load the real `.npz` neural network weights and `.json` "
                "logs from MIT CSAIL's Weight-Space Programming experiments so you can scrub frame-by-frame "
                "through the Car-on-Ramp, Atari Breakout Wall-Bounce, and LLM Sycophancy/Loop experiments."
            ),
            "nodeIds": [
                "file:src/bi_interp/wsp_empirical.py",
                "file:src/bi_interp/simulator.py",
                "file:simulations/run_walkthrough_simulation.py",
            ],
        },
        {
            "order": 7,
            "title": "7. Run a Live Real-World Use Case (`examples/live_use_case_demo.py`)",
            "description": (
                "Want to use this in the real world or run a live demo? `examples/live_use_case_demo.py` "
                "runs 3 practical applications out of the box: (1) a Healthcare & Finance AI Compliance Gate "
                "that blocks sycophantic recommendations, (2) a 1-second Anti-Loop Hotfix for local LLMs, and "
                "(3) a Zero-Retraining Safety Reflex for autonomous controllers."
            ),
            "nodeIds": [
                "file:examples/live_use_case_demo.py",
                "file:src/bi_interp/adapters/hf_hooks.py",
            ],
            "languageLesson": (
                "Click 'Domain' in the top bar next to 'Structural' to explore these workflows as visual "
                "process pipelines!"
            ),
        },
    ]

    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {
            "name": "backward-induction-interp",
            "languages": ["python", "markdown"],
            "frameworks": [
                "NumPy",
                "PyTorch / HuggingFace Hooks",
                "Game-Theoretic Backward Induction",
                "Weight-Space Programming (MIT CSAIL)",
                "Understand-Anything Graph",
            ],
            "description": (
                "Plain-English & empirical toolkit using Game-Theoretic Backward Induction and "
                "Weight-Space Programming to separate an AI's internal neural decisions (Mind) "
                "from its written Chain-of-Thought explanations (Mouth) — and fix broken behaviors "
                "without retraining."
            ),
            "analyzedAt": "2026-11-06T16:00:00.000Z",
            "gitCommitHash": commit_hash,
        },
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "tour": tour,
    }


def build_domain_graph(commit_hash: str) -> Dict[str, Any]:
    """Build `.ua/domain-graph.json` so the 'Domain / Structural' toggle works in the dashboard."""
    nodes: List[Dict[str, Any]] = [
        # --- Domain 1 ---
        {
            "id": "domain:mind-vs-mouth-audit",
            "type": "domain",
            "name": "1. Mind vs. Mouth AI Compliance Gate",
            "summary": (
                "Detects when an AI in Healthcare, Finance, or Legal workflows secretly copies "
                "a supervisor's bias while writing an 'objective' Chain-of-Thought excuse."
            ),
            "tags": ["compliance", "sycophancy", "cot-faithfulness", "audit"],
            "complexity": "moderate",
            "domainMeta": {
                "entities": ["3-Branch Probe Tree", "Attention Head 10.2", "Discrepancy Score (Delta_DR)"],
                "businessRules": [
                    "Block AI recommendation if Delta_DR >= 0.15 or dominant internal factor != stated CoT factor",
                    "Always include a Decoy Ticket/Number control branch to separate mechanical shortcuts from reasoning",
                ],
                "crossDomainInteractions": ["domain:weight-reading", "domain:surgical-repair"],
            },
        },
        {
            "id": "flow:three-branch-audit",
            "type": "flow",
            "name": "3-Branch Counterfactual Audit Flow",
            "summary": "Runs Baseline, Biased-Hint, and Decoy-Control prompts to measure internal vs. stated reasoning.",
            "tags": ["audit-flow", "probing"],
            "complexity": "moderate",
            "domainMeta": {
                "entryPoint": "examples/live_use_case_demo.py --use-case 1",
                "entryType": "cli",
            },
        },
        {
            "id": "step:audit-1-probes",
            "type": "step",
            "name": "Send 3 Fork-in-the-Road Probes",
            "filePath": "examples/live_use_case_demo.py",
            "summary": "Compare AI output on (1) Neutral Facts, (2) Supervisor Biased Hint, and (3) Decoy Ticket Number.",
            "tags": ["step"],
            "complexity": "simple",
        },
        {
            "id": "step:audit-2-lockin",
            "type": "step",
            "name": "Trace Lock-In Layer & Attention",
            "filePath": "src/bi_interp/subgame/inverse_bi.py",
            "summary": "Use Inverse Backward Induction to see if Layer 10 (Head 10.2) locked in the hint before CoT began.",
            "tags": ["step"],
            "complexity": "moderate",
        },
        {
            "id": "step:audit-3-verdict",
            "type": "step",
            "name": "Compute Delta_DR & Gate Verdict",
            "filePath": "src/bi_interp/discrepancy/cot_vs_decision.py",
            "summary": "Compare internal factor weights (93.6% hint) vs. stated CoT (96% facts) -> Flag & block if unfaithful.",
            "tags": ["step"],
            "complexity": "simple",
        },
        # --- Domain 2 ---
        {
            "id": "domain:weight-reading",
            "type": "domain",
            "name": "2. Zero-Pass Neural Blueprint Reading",
            "summary": (
                "Reads raw neural network weights backward from output actions to input features "
                "without running forward passes."
            ),
            "tags": ["weight-reading", "stopping-game", "breakout", "copying-heads"],
            "complexity": "moderate",
            "domainMeta": {
                "entities": ["Stopping Game Value Matrix", "Virtual QK/OV Circuit", "Dormant Neurons"],
                "businessRules": [
                    "Trace weights backward from output actions to rank causal input channels in 0 forward passes",
                    "Identify unused/dormant hidden units that can host surgical safety overrides",
                ],
                "crossDomainInteractions": ["domain:surgical-repair"],
            },
        },
        {
            "id": "flow:backward-blueprint-scan",
            "type": "flow",
            "name": "Backward Weight-Graph Attribution Flow",
            "summary": "Extracts input receptive fields, copying heads, and 815-weight Breakout ball trackers from raw tensors.",
            "tags": ["weight-flow"],
            "complexity": "moderate",
            "domainMeta": {
                "entryPoint": "experiments/exp4_wsp_empirical_reproduction.py",
                "entryType": "cli",
            },
        },
        {
            "id": "step:read-1-load",
            "type": "step",
            "name": "Load Raw Weight Tensors (.npz / PyTorch)",
            "filePath": "src/bi_interp/wsp_empirical.py",
            "summary": "Inspect weight matrices directly without needing training data or embeddings.",
            "tags": ["step"],
            "complexity": "simple",
        },
        {
            "id": "step:read-2-propagate",
            "type": "step",
            "name": "Propagate Action Payoffs Backward",
            "filePath": "src/bi_interp/weight_space/stopping_game.py",
            "summary": "Multiply layer value matrices backward from output logits to input sensors.",
            "tags": ["step"],
            "complexity": "moderate",
        },
        {
            "id": "step:read-3-isolate",
            "type": "step",
            "name": "Isolate Causal Circuits & Dormant Units",
            "filePath": "src/bi_interp/weight_space/transformer_weights.py",
            "summary": "Pinpoint the exact 815 Breakout lookahead weights or Transformer induction heads (H1.0 -> H2.0).",
            "tags": ["step"],
            "complexity": "moderate",
        },
        # --- Domain 3 ---
        {
            "id": "domain:surgical-repair",
            "type": "domain",
            "name": "3. Zero-Retraining Surgical Weight Repair",
            "summary": (
                "Patches infinite reasoning loops in LLMs (0.016% weights) and stalls in RL agents "
                "(3.5% weights) in under 1 second on CPU without gradient training."
            ),
            "tags": ["weight-surgery", "qwen-loop-fix", "kinetix-ramp", "zero-retraining"],
            "complexity": "complex",
            "domainMeta": {
                "entities": ["Rank-1 Loop Projection", "6-Neuron Subgame Override", "Zero-Leakage Guarantee"],
                "businessRules": [
                    "Edit only the top-k causal columns/neurons identified by backward induction",
                    "Ensure normal-regime outputs experience 0.000000 perturbation outside the target bug state",
                ],
                "crossDomainInteractions": ["domain:mind-vs-mouth-audit", "domain:weight-reading"],
            },
        },
        {
            "id": "flow:surgical-patch-pipeline",
            "type": "flow",
            "name": "1-Second Surgical Weight Patch Flow",
            "summary": "Projects out LLM repetition vectors or grafts multi-step backup reflexes into dormant neurons.",
            "tags": ["repair-flow"],
            "complexity": "moderate",
            "domainMeta": {
                "entryPoint": "examples/live_use_case_demo.py --use-case 2",
                "entryType": "cli",
            },
        },
        {
            "id": "step:repair-1-target",
            "type": "step",
            "name": "Select Top-64 MLP Columns / 6 Dormant Neurons",
            "filePath": "src/bi_interp/editing/surgical_editor.py",
            "summary": "Target Layers 15 & 19 in Qwen3.5-0.8B (0.016% weights) or 6 hidden units in Kinetix PPO (3.5% weights).",
            "tags": ["step"],
            "complexity": "moderate",
        },
        {
            "id": "step:repair-2-apply",
            "type": "step",
            "name": "Apply Rank-1 Projection or Subgame Graft",
            "filePath": "src/bi_interp/editing/surgical_editor.py",
            "summary": "Subtract the loop direction or wire the opposing-sign ReLU backup pair with zero gradient steps.",
            "tags": ["step"],
            "complexity": "moderate",
        },
        {
            "id": "step:repair-3-verify",
            "type": "step",
            "name": "Verify Benchmark Jump & Zero Side-Effects",
            "filePath": "simulations/run_walkthrough_simulation.py",
            "summary": "Confirm MATH-500 jumps 7.9% -> 48.1% and Car-on-Ramp jumps 0% -> 100% with 0.0 normal-state drift.",
            "tags": ["step"],
            "complexity": "simple",
        },
    ]

    edges: List[Dict[str, Any]] = [
        # Cross-domain edges
        {
            "source": "domain:weight-reading",
            "target": "domain:mind-vs-mouth-audit",
            "type": "cross_domain",
            "direction": "forward",
            "description": "Maps candidate heads & layers before running 3-branch subgame probes",
            "weight": 0.9,
        },
        {
            "source": "domain:mind-vs-mouth-audit",
            "target": "domain:surgical-repair",
            "type": "cross_domain",
            "direction": "forward",
            "description": "Pinpoints the exact unfaithful head or looping MLP columns to patch",
            "weight": 0.9,
        },
        {
            "source": "domain:weight-reading",
            "target": "domain:surgical-repair",
            "type": "cross_domain",
            "direction": "forward",
            "description": "Finds dormant neurons to host zero-side-effect safety reflexes",
            "weight": 0.85,
        },
        # Domain 1 flows & steps
        {
            "source": "domain:mind-vs-mouth-audit",
            "target": "flow:three-branch-audit",
            "type": "contains_flow",
            "direction": "forward",
            "weight": 1.0,
        },
        {
            "source": "flow:three-branch-audit",
            "target": "step:audit-1-probes",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.1,
        },
        {
            "source": "flow:three-branch-audit",
            "target": "step:audit-2-lockin",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.2,
        },
        {
            "source": "flow:three-branch-audit",
            "target": "step:audit-3-verdict",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.3,
        },
        # Domain 2 flows & steps
        {
            "source": "domain:weight-reading",
            "target": "flow:backward-blueprint-scan",
            "type": "contains_flow",
            "direction": "forward",
            "weight": 1.0,
        },
        {
            "source": "flow:backward-blueprint-scan",
            "target": "step:read-1-load",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.1,
        },
        {
            "source": "flow:backward-blueprint-scan",
            "target": "step:read-2-propagate",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.2,
        },
        {
            "source": "flow:backward-blueprint-scan",
            "target": "step:read-3-isolate",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.3,
        },
        # Domain 3 flows & steps
        {
            "source": "domain:surgical-repair",
            "target": "flow:surgical-patch-pipeline",
            "type": "contains_flow",
            "direction": "forward",
            "weight": 1.0,
        },
        {
            "source": "flow:surgical-patch-pipeline",
            "target": "step:repair-1-target",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.1,
        },
        {
            "source": "flow:surgical-patch-pipeline",
            "target": "step:repair-2-apply",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.2,
        },
        {
            "source": "flow:surgical-patch-pipeline",
            "target": "step:repair-3-verify",
            "type": "flow_step",
            "direction": "forward",
            "weight": 0.3,
        },
    ]

    return {
        "version": "1.0.0",
        "project": {
            "name": "backward-induction-interp (Domain Flows)",
            "languages": ["python"],
            "frameworks": ["Backward Induction", "Weight-Space Programming"],
            "description": "High-level process flows showing how Weight Reading, Subgame Auditing, and Surgical Repair work together.",
            "analyzedAt": "2026-11-06T16:00:00.000Z",
            "gitCommitHash": commit_hash,
        },
        "nodes": nodes,
        "edges": edges,
        "layers": [],
        "tour": [],
    }


def export_static_sources(kg: Dict[str, Any], dest_root: Path) -> int:
    """Write `./sources/<filePath>.json` inside `docs/demo/` so CodeViewer works on GitHub Pages."""
    seen = set()
    count = 0
    for node in kg["nodes"]:
        rel_path = node.get("filePath")
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        src_file = ROOT / rel_path
        if not src_file.exists() or not src_file.is_file():
            continue
        content = src_file.read_text(encoding="utf-8")
        ext = src_file.suffix.lstrip(".").lower()
        lang_map = {"py": "python", "md": "markdown", "json": "json"}
        payload = {
            "path": rel_path,
            "language": lang_map.get(ext, "text"),
            "content": content,
            "sizeBytes": len(content.encode("utf-8")),
            "lineCount": len(content.splitlines()),
        }
        out_file = dest_root / "sources" / f"{rel_path}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    commit_hash = _get_git_commit()
    kg = build_knowledge_graph(commit_hash)
    dg = build_domain_graph(commit_hash)
    meta = {
        "lastAnalyzedAt": "2026-11-06T16:00:00.000Z",
        "gitCommitHash": commit_hash,
        "version": "1.0.0",
        "analyzedFiles": len([n for n in kg["nodes"] if n["type"] in ("file", "document")]),
    }
    config = {"autoUpdate": False, "outputLanguage": "en"}

    UA_DIR.mkdir(parents=True, exist_ok=True)
    (UA_DIR / "knowledge-graph.json").write_text(json.dumps(kg, indent=2), encoding="utf-8")
    (UA_DIR / "domain-graph.json").write_text(json.dumps(dg, indent=2), encoding="utf-8")
    (UA_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (UA_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Copy built Understand-Anything dashboard dist if available in TEMP
    temp_dist = Path(
        r"C:\Users\aabdelsalam\AppData\Local\Temp\ua-build\understand-anything-plugin\packages\dashboard\dist"
    )
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if temp_dist.exists():
        for item in temp_dist.iterdir():
            target = DEMO_DIR / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    # Write our project's knowledge-graph.json, domain-graph.json, meta.json, config.json into docs/demo/
    (DEMO_DIR / "knowledge-graph.json").write_text(json.dumps(kg, indent=2), encoding="utf-8")
    (DEMO_DIR / "domain-graph.json").write_text(json.dumps(dg, indent=2), encoding="utf-8")
    (DEMO_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (DEMO_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (DEMO_DIR / "diff-overlay.json").write_text("null\n", encoding="utf-8")

    # Ensure docs/demo/index.html pre-dismisses the blocking onboarding overlay so the graph is immediately visible
    demo_index = DEMO_DIR / "index.html"
    if demo_index.exists():
        html = demo_index.read_text(encoding="utf-8")
        inject = '<script>try{localStorage.setItem("ua-onboarding-dismissed-v1","1");}catch(e){}</script>'
        if "ua-onboarding-dismissed-v1" not in html:
            html = html.replace("<head>", "<head>\n    " + inject, 1)
            demo_index.write_text(html, encoding="utf-8")

    num_sources = export_static_sources(kg, DEMO_DIR)
    print(f"Generated .ua/knowledge-graph.json ({len(kg['nodes'])} nodes, {len(kg['edges'])} edges, {len(kg['tour'])} tour steps)")
    print(f"Generated .ua/domain-graph.json ({len(dg['nodes'])} domain/flow/step nodes)")
    print(f"Bundled Understand-Anything live demo into docs/demo/ with {num_sources} static source previews.")


if __name__ == "__main__":
    main()
