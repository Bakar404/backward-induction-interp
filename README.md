# `backward-induction-interp`: Playing Neural Networks & Reasoning Subgames Backward

> **Core Research Question:**  
> *Can game theory and backward induction—united with **Weight-Space Programming**—reveal an AI model's internal neural network decisions and the true factors taken into consideration, separating **internal neural decisions** from **surface-level reasoning (Chain-of-Thought)**?*

---

## Overview & Motivation

Modern AI interpretability faces a fundamental gap between **what a model says it is thinking** (surface Chain-of-Thought / reasoning tokens) and **what its neural weights actually compute** (internal decision circuits and latent preferences).

Between 2024 and late 2026, three major research threads emerged independently:
1. **Playing the Network Backward (Game-Theoretic Attribution):** *Gaubert & Vlassopoulos (2025)* and *Zimmermann et al. (Oct 2026, [arXiv:2605.06212](https://arxiv.org/abs/2605.06212))* proved that a neural network's forward pass is mathematically equivalent to the game value of a two-player sequential **Stopping Game** played on the network graph (layers = rounds, neurons = states, inputs = terminal payoffs). Backward attribution rules (gradients, $\alpha\beta$-LRP, attention routing) emerge as equilibrium trajectory distributions played **backward** from output logits to input features.
2. **Unfaithful Reasoning vs. Internal Lookahead Circuits:** *Unlocking the Future ([arXiv:2406.16033](https://arxiv.org/abs/2406.16033))* and *CIE-Scorer (Shen et al., May 2026, [arXiv:2605.25603](https://arxiv.org/abs/2605.25603))* demonstrated that language models often lock in future decisions in early-to-middle hidden layers while emitting post-hoc CoT rationales that omit or contradict the true causal drivers (such as sycophantic hints or shortcut heuristics).
3. **Sparks of Weight-Space Programming (WSP):** *Takehi, Teoh, Zhang, & Khattab (MIT CSAIL, Nov 2, 2026, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/))* showed that frontier LLMs can natively **read and write neural network weights** without gradient descent:
   - **Zero-Pass Weight Reading (WSP Exp 1–3):** Identifying which game an MLP plays (Breakout vs. Pong vs. Pac-Man vs. Freeway), what geometric shape a diffusion model generates, or which attention heads are copying/induction heads in a 4-layer transformer with input/output embeddings removed—purely by tracing weight matrices backward and forward.
   - **The Pure-Weight Basis-Rotation Bottleneck (WSP Exp 7 — Sycophancy):** From raw weights alone (with frozen embeddings), GPT-6 Astra failed (**0/4**) to distinguish an honest vs. sycophantic Gemma-3-270M because internal basis rotations obscure how weight differences compose into behavior. However, once allowed a tiny budget of **strategic probing inputs + attention routing patterns** (isolating Head `10.2` attending to the user's suggested answer), it succeeded **3/3**.
   - **Surgical Weight-Space Editing (WSP Exp 4 & Exp 8):** Fixing a PPO policy trapped in a ramp local minimum (editing 3.5% of weights to back up before accelerating) and fixing infinite reasoning loops in `Qwen3.5-0.8B` by subtracting a single looping direction from 64 MLP output columns in layers 15 and 19 (**0.016%** of weights, boosting MATH-500 from **7.9% → 48.1%** and IFEval from **21.0% → 62.3%**).

---

## What We Contribute: A 3-Stage Unified Framework

`backward-induction-interp` bridges these three threads into a single mathematical and programmatic toolkit (`bi_interp`):

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Static Weight-Space Backward Induction (Zero Forward Passes)             │
│  • Casts raw weight matrices {W^(l)} & Transformer (W_QK, W_OV) as a sequential   │
│    Stopping Game from output logits back to input receptive fields.               │
│  • Extracts positive/negative game trajectories, virtual copying/induction heads, │
│    and effective output-subspace rank bottlenecks natively from weights.          │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼ (When internal basis rotations hide behavior, e.g. WSP Exp 7)
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Strategic Subgame Probing & Decision-vs-Reasoning Discrepancy            │
│  • Constructs minimal Extensive-Form Game Trees (e.g. Honest Truth vs.            │
│    Sycophantic Hint; Myopic Greedy Action vs. Long-Horizon Backup).               │
│  • Applies Inverse Backward Induction across behavioral + layer-wise activation   │
│    trajectories to recover:                                                       │
│      1. Latent Neural Payoff Weights (u_internal) vs. Stated CoT Weights (u_CoT)  │
│      2. Decision Lock-In Layer (l*): where the subgame equilibrium crystallizes   │
│      3. Decision-Reasoning Discrepancy Score (Δ_DR)                               │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼ (Counterfactual verification & zero-gradient repair)
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Backward-Induction Guided Weight-Space Editing                           │
│  • Uses backward-inducted circuit attributions to apply surgical, gradient-free   │
│    weight edits:                                                                  │
│      1. Rank-1 MLP Direction Projection (removing reasoning loops, WSP Exp 8)     │
│      2. Subgame Pathway Grafting (overriding myopic local minima, WSP Exp 4)      │
│      3. Attention Routing Suppression (neutralizing sycophantic heads, WSP Exp 7) │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
backward-induction-interp/
├── README.md                              # Project overview, architecture, quickstart
├── pyproject.toml                         # Package configuration (bi-interp)
├── docs/
│   ├── literature_review.md               # Synthesis of 2024-2026 papers & open problems
│   ├── mathematical_framework.md          # Formal equations for all 3 stages
│   └── weight_space_programming_bridge.md # Deep dive: how Backward Induction complements WSP
├── src/
│   └── bi_interp/
│       ├── weight_space/
│       │   ├── stopping_game.py           # Backward Stopping Game on MLP weight graphs (Zero-Pass)
│       │   └── transformer_weights.py     # Embedding-free QK/OV virtual circuit & copying-head detector
│       ├── subgame/
│       │   ├── extensive_form.py          # Extensive-form game tree & Subgame Perfect Equilibrium (SPE)
│       │   └── inverse_bi.py              # Inverse Backward Induction & layer-wise decision lock-in (l*)
│       ├── discrepancy/
│       │   └── cot_vs_decision.py         # Internal Decision vs. Stated CoT Reasoning divergence metrics
│       ├── editing/
│       │   └── surgical_editor.py         # Gradient-free surgical weight editing (rank-1 removal & grafting)
│       └── adapters/
│           └── hf_hooks.py                # PyTorch / HuggingFace hooks for real LLMs (Qwen, Gemma, Llama)
├── experiments/
│   ├── exp1_zero_pass_weight_reading.py   # Stage 1 Demo: Read game identity & copying heads from weights alone
│   ├── exp2_sycophancy_cot_subgame.py     # Stage 2 Demo: Expose sycophantic neural decisions vs. unfaithful CoT
│   └── exp3_surgical_weight_repair.py     # Stage 3 Demo: Fix ramp local minima & MLP reasoning loops via weight edits
└── tests/
    ├── test_weight_space.py
    ├── test_subgame_and_discrepancy.py
    └── test_surgical_editing.py
```

---

## Quickstart

### 1. Install in editable mode
```bash
pip install -e ".[dev]"
```

### 2. Run the 3 Stage Experiments
Each experiment runs in seconds on CPU and demonstrates one stage of the framework:

```bash
# Stage 1: Zero-Pass Weight-Space Backward Induction (Game ID + Transformer Copying Heads)
python experiments/exp1_zero_pass_weight_reading.py

# Stage 2: Extensive-Form Subgame Probing (Internal Neural Decision vs. Unfaithful CoT)
python experiments/exp2_sycophancy_cot_subgame.py

# Stage 3: Surgical Weight-Space Editing (Fixing Myopic Local Minima & MLP Reasoning Loops)
python experiments/exp3_surgical_weight_repair.py
```

### 3. Run the Test Suite
```bash
pytest -v
```

---

## Key Documentation
- **[Literature Review (`docs/literature_review.md`)](docs/literature_review.md):** Detailed survey of backward game attribution, CoT unfaithfulness, inverse game theory, and weight-space programming.
- **[Mathematical Framework (`docs/mathematical_framework.md`)](docs/mathematical_framework.md):** Formal definitions of the Stopping Decomposition, Inverse Subgame Backward Induction, Decision Lock-In Layer $\ell^*$, and Rank-1 Weight-Space Projections.
- **[Weight-Space Programming Bridge (`docs/weight_space_programming_bridge.md`)](docs/weight_space_programming_bridge.md):** Why pure static weight inspection fails under basis rotations (WSP Exp 7) and how hybrid backward induction solves it.
