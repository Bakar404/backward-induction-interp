# backward-induction-interp

Game-theoretic backward induction and empirical weight-space circuit analysis for separating a neural network's internal decision pathways from surface Chain-of-Thought (CoT) reasoning.

This repository links three lines of research:
1. **Sequential Stopping Games on Neural Weight Graphs** (*Gaubert & Vlassopoulos, 2025*; *Zimmermann et al., Oct 2026, [arXiv:2605.06212](https://arxiv.org/abs/2605.06212)*), which prove that a neural network's forward evaluation equals the equilibrium value of a two-player sequential stopping game played backward from output logits to input coordinates.
2. **Internal Decision Lock-In vs. Unfaithful CoT** (*Unlocking the Future, [arXiv:2406.16033](https://arxiv.org/abs/2406.16033)*; *Shen et al., May 2026, [arXiv:2605.25603](https://arxiv.org/abs/2605.25603)*), which show that models frequently commit to latent decisions in early or middle layers while emitting post-hoc rationales that omit the true causal triggers.
3. **Sparks of Weight-Space Programming** (*Takehi, Teoh, Zhang, & Khattab, MIT CSAIL, Nov 2, 2026, [blog](https://rikiyat.github.io/blog/weight-space-programming/) / [logs](https://github.com/RikiyaT/lm_understand-logs)*), which demonstrates zero-pass weight reading, exposes the basis-rotation bottleneck of pure weight inspection on sycophantic transformers, and performs sparse, gradient-free surgical weight edits.

---

## Graphify Architecture & Neural Circuit Map

The repository structure and the empirical neural circuits extracted from `RikiyaT/lm_understand-logs` are indexed and clustered using [Graphify (`Graphify-Labs/graphify`)](https://github.com/Graphify-Labs/graphify) (`168` AST and circuit nodes, `355` edges, `12` Leiden communities; see [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md), [`graphify-out/graph.svg`](graphify-out/graph.svg), and [`graphify-out/graph.json`](graphify-out/graph.json)).

![Graphify Repository and Neural Circuit Map](assets/fig5_graphify_architecture_map.svg)

---

## Empirical Reproduction of Weight-Space Programming (`RikiyaT/lm_understand-logs`)

In addition to synthetic unit benchmarks (`exp1..exp3`), this repository bundles the released `.npz` checkpoints and `.json` evaluation artifacts from `RikiyaT/lm_understand-logs` under `data/wsp_artifacts/` and reproduces the exact mathematical mechanisms behind why those specific neural circuits and weight edits were chosen (`src/bi_interp/wsp_empirical.py` and `experiments/exp4_wsp_empirical_reproduction.py`).

### 1. Exp 1 & Exp 5: Backward Induction on MinAtar Breakout Checkpoints (`a9f535cc.npz` & `hand.npz`)

![Breakout Weight Reading and Hand-Coded V-Trajectory](assets/fig1_breakout_weight_reading.svg)

- **Why `a9f535cc.npz` (`17,253` params, `500 -> 32 -> 32 -> 5`) is identifiable as Breakout with 0 forward passes:**
  - Starting at the output head `W3`, action variance concentrates heavily in the horizontal left/right actions (`std = 1.737` vs. vertical `1.132`, ratio `1.53x`), with `W3[22, left] = -5.336`, `W3[11, left] = +4.057`, and `W3[24, right] = +3.031`.
  - Tracing backward through `W2` to `W1` (`10 x 10 x 5`), input Channel 0 (the agent) concentrates **67.8%** of its interior L2 weight energy on a single horizontal line—**Row 8** (`W1[(8,1,ch0), u15] = +4.066`, `u31 = -4.013`)—revealing a paddle restricted to Row 8.
  - Channel 2 concentrates in upper rows 1–3 (`4.27x` upper-to-lower ratio, `W1[(3,1,ch2), u0] = -4.811`), matching the Breakout brick band, while Channels 1 and 4 carry opposite-signed weights on adjacent cells (`(1,3)` to units 7, 8: `-4.791, +3.436` vs. `-4.301, +3.057`) to compute ball velocity from current and previous frames.
- **Why the zero-shot hand-written policy (`hand.npz`, `815 / 20,486` nonzero weights = `3.98%`) scores `517.4` vs. 500M-step PPO's `25.3`:**
  - Instead of reactive proximity features, `W1` units `0..3` encode the exact backward-inducted **wall-reflection V-trajectory** of the ball (`c* = |(9 - r) +/- c|` folded at columns 0 and 9) minus `0.02 * paddle_col`, paired with horizontal (`u4`) and vertical (`u5`) travel direction detectors.
  - `W2` units `0..7` act as context-gated threshold comparators (`b2 = -415`), and `W3` converts those 4 regime pairs into deterministic left/right/no-op actions.

---

### 2. Exp 3: Embedding-Free Character Transformer Circuit Algebra (`python/blocks.npz`)

![Character Transformer Circuit Topology and Weight Algebra](assets/fig2_char_lm_circuit_graph.svg)

- **Why the 3 copying heads (`L3H5`, `L3H1`, `L2H1`) can be found when token embeddings are stripped:**
  - With token embeddings and the output head removed, forward passes are impossible. For each layer $\ell \in \{0,1,2,3\}$ and head $h \in \{0..5\}$, we form the LayerNorm-projected query-key and output-value operators on the zero-sum subspace $P = I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top$, $D = \operatorname{diag}(\texttt{ln1\_w})$:
    $$M^{(\ell,h)} = \frac{1}{\sqrt{d_k}} P D (W_q^{(\ell,h)})^\top W_k^{(\ell,h)} D P, \qquad B^{(\ell,h)} = P W_o^{(\ell,h)} W_v^{(\ell,h)} D P$$
  - Computing directional key vs. query composition across all layer pairs:
    $$C_k = \frac{\operatorname{tr}(M_{\text{later}} B_{\text{earlier}})}{\|M_{\text{later}}\|_F \|B_{\text{earlier}}\|_F}, \qquad C_q = \frac{\operatorname{tr}(M_{\text{later}} B_{\text{earlier}}^\top)}{\|M_{\text{later}}\|_F \|B_{\text{earlier}}\|_F}$$
    reproduces the exact circuit couplings from `python/blocks.npz`:
    - `L0H5 -> L2H1`: $C_k = +0.0776$, $C_q = -0.0398$ (ground-truth copying `induct = 0.245`)
    - `L2H2 -> L3H1`: $C_k = +0.1104$, $C_q = -0.0164$ (ground-truth copying `induct = 0.269`)
    - `L2H2 -> L3H5`: $C_k = +0.1836$, $C_q = -0.0354$ (ground-truth copying `induct = 0.546`)
  - Meanwhile, `L0H4` has near-zero bilinear routing ($\|M\|_F = 0.0093$, uniform attention) but the largest value-write singular values in Layer 0 (`13.877`, `9.644`), supplying **53–68%** of the key-subspace energy consumed by Layer 1 local heads `L1H1..L1H5`.

---

### 3. Exp 4: Why 51 Weight Assignments Fix the Kinetix Ramp Policy (`edited_actor.npz`)

![Kinetix 51-Statement Subgame Graft and Override Logits](assets/fig3_kinetix_subgame_graft.svg)

- **Why the original PPO actor (`171,664` params) scored `0/256`:**
  - Tracing `embed.W` into the largest hidden channel (unit 116) shows that the policy reads horizontal velocity indices (`+2.25, +2.37, +1.73`) and shapes a greedy forward coasting controller (`Action 0` vs. `Action 1`) that stalls against the steep ramp.
- **Why the 51-statement surgical edit (`6,073 / 171,664` changed entries = `3.538%` after prompt decimal rounding) achieves `1,024/1,024` (`100%`):**
  1. **ReLU Sensor Taps (`embed[:, 6:9]`):** Routes `obs[52]` ($x_{\text{body}}$), `obs[54] + 11` ($v_x$), and `obs[234]` ($x_{\text{green wheel}}$) linearly through ReLU.
  2. **High-Gain `tanh` Thresholds (`actor0[:, 122:128]`):** Uses gain `80` (`tanh(80*x - b)`) to create 6 binary state indicators: `x_body > 2.50` (`u122`), `x_body > 3.10` (`u123`), `x_body > 3.80` (`u124`), `v_x > -0.05` (`u125`), `v_x > -1.50` (`u126`), and `x_green > 0.95` (`u127`).
  3. **Subgame Boolean Latch (`actor1..4[:, 126:128]`):** Propagates two saturated $\pm 1$ wires: `u126 = +1` (Reverse Right to build runway) and `u127 = +1` (Full Launch Left once backed up or fast).
  4. **Exact Cancellation Handback (`actor_head[126:128, 0:3]`):** Sets `W[126, 0:3] = [-100, -100, +200]`, `W[127, 0:3] = [-100, +200, -100]`, and adds `[-200, +100, +100]` to `b[0:3]`. As soon as the green wheel clears the red obstacle ($x_{\text{green}} < 0.95$), both `u126` and `u127` saturate to `-1`, and `(-1)*(-100) + (-1)*(-100) = +200` cancels the `[-200, +100, +100]` bias offset to **exact `0.0`**, restoring the unmodified PPO backbone (`96.46%` of weights) to stabilize at the goal.

---

### 4. Exp 7 & Exp 8: Gemma-3-270M Sycophancy Subgame & Qwen3.5-0.8B Loop Projection

![Gemma Head 10.2 Sycophancy Controls and Qwen3.5-0.8B Surgical MLP Edit](assets/fig4_sycophancy_and_qwen_loops.svg)

- **Exp 7 (`Gemma-3-270M` Honest `A` vs. Sycophantic `B`):**
  - **Why static weight reading failed (`0/4`):** With frozen token embeddings, the $W_{OV}$ transmission directions in Heads `9.2` and `10.2` have **`0.9999` cosine similarity** between `A` and `B`, and the Layer 17 MLP readout shift is distributed across `542` hidden units.
  - **What attention subgame probing (`pair2/attn/r7`) reveals:** On standard math prompts with a false user suggestion (`t0..t7`), Head `10.2` shifts from **94.29%** attention on question operands (`'3', '4'`) in `A` to **3.47%** in `B`, and from **0.75%** to **85.64%** on the user's stated guess (`'8'`).
  - **What extensive-form counterfactual controls (`t8, t9, t12`) prove:** Sycophancy in `B` is **not** a semantic belief-tracking circuit. When `"I think the answer is 8"` is replaced by `"My ticket number is 8"` (`t8`), Head `10.2` still attends **68.51%** to `'8'`. When two numbers appear (`"...answer is 8. My ticket number is 6."`, `t9`), it attends **42.04%** to the later number `'6'`. When the user's guess is placed *before* the question (`t12`), Head `10.2` reverts **61.91%** to question operand `'4'`. The circuit is a myopic recency + `" is X"` retrieval shortcut.
- **Exp 8 (`Qwen3.5-0.8B` Reasoning Loop Removal, `p10c -> p25c -> p50c`):**
  - Across increasing inspection budgets, the intervention narrows from 30 matrices across Layers 8–22 (`p10c`, MATH-500 `7.9% -> 12.3%`) to zeroing Layer 15 Attention Head 7 (`p25c`, `o_proj[:, 1792:2048]`, MATH-500 `14.1%`), and finally to **only the top-64 intermediate columns** of `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj` (`p50c`):
    $$W_{\text{new}} = W_0 - 6\, u_\ell \left(m_{64}^{(\ell)} \odot (u_\ell^\top W_0)\right)$$
  - Editing those `131,072` weights (**0.016%** of the model, `130,188` changed `bfloat16` entries) reduces repetition-branch log-probs by **-0.10311 nats/token** vs. **-0.00903 nats/token** on productive branches (**11.42x selectivity**), cutting 32k-token loop cutoffs from **93.0% to 33.0%** and raising MATH-500 from **7.9% to 48.1%**, GSM8K from **15.5% to 67.1%**, and IFEval from **21.0% to 62.3%**.
  - On the `114 / 1,152` benchmark problems where the unedited base model finished thinking without looping, accuracy was **93.86% before and 93.86% after (`0.00%` change)**—proving the edit removes a self-reinforcing repetition attractor rather than altering mathematical knowledge.

---

## Repository Layout

```text
backward-induction-interp/
├── assets/                                    # Vector SVG figures generated from WSP checkpoints + Graphify
│   ├── fig1_breakout_weight_reading.svg
│   ├── fig2_char_lm_circuit_graph.svg
│   ├── fig3_kinetix_subgame_graft.svg
│   ├── fig4_sycophancy_and_qwen_loops.svg
│   └── fig5_graphify_architecture_map.svg
├── data/wsp_artifacts/                        # Released checkpoints & eval JSONs from RikiyaT/lm_understand-logs
├── docs/
│   ├── literature_review.md                   # Survey of 2024-2026 game-theoretic attribution & CoT papers
│   ├── mathematical_framework.md              # Stopping games, inverse backward induction, rank-1 edits
│   └── weight_space_programming_bridge.md     # Mechanistic breakdown of WSP Exp 1, 3, 4, 5, 7, and 8
├── experiments/
│   ├── exp1_zero_pass_weight_reading.py       # Synthetic Stage 1 demo (MLP stopping game + QK/OV circuits)
│   ├── exp2_sycophancy_cot_subgame.py         # Synthetic Stage 2 demo (subgame probing vs. unfaithful CoT)
│   ├── exp3_surgical_weight_repair.py         # Synthetic Stage 3 demo (subgame grafting & rank-1 projection)
│   └── exp4_wsp_empirical_reproduction.py     # Exact reproduction on real WSP checkpoints (Exp 1, 3, 4, 5, 7, 8)
├── graphify-out/                              # Graphify AST + circuit knowledge graph outputs (JSON/HTML/SVG/MD)
├── scripts/
│   └── build_graphs_and_visuals.py            # Runs Graphify pipeline and renders assets/fig1..fig5.svg
├── src/bi_interp/
│   ├── weight_space/                          # Zero-pass backward stopping game & embedding-free QK/OV algebra
│   ├── subgame/                               # Extensive-form backward induction & inverse payoff recovery
│   ├── discrepancy/                           # Internal decision vs. surface CoT divergence metrics
│   ├── editing/                               # Gradient-free rank-1 loop projection & subgame pathway grafting
│   ├── adapters/                              # PyTorch / HuggingFace hooks for live transformer inspection
│   └── wsp_empirical.py                       # Empirical analyzers for real RikiyaT/lm_understand-logs artifacts
└── tests/                                     # 14 unit and empirical reproduction tests (pytest)
```

---

## Reproducing Everything Locally

```bash
# 1. Install package and test dependencies
pip install -e ".[dev]"

# 2. Run the full test suite (synthetic + empirical WSP checkpoint tests)
pytest -v

# 3. Run the empirical reproduction of WSP Experiments 1, 3, 4, 5, 7, and 8
python experiments/exp4_wsp_empirical_reproduction.py

# 4. Run the synthetic stage demos
python experiments/exp1_zero_pass_weight_reading.py
python experiments/exp2_sycophancy_cot_subgame.py
python experiments/exp3_surgical_weight_repair.py

# 5. Rebuild the Graphify knowledge graph (graphify-out/) and vector figures (assets/*.svg)
python scripts/build_graphs_and_visuals.py
```

---

## Documentation
- [`docs/weight_space_programming_bridge.md`](docs/weight_space_programming_bridge.md): Complete tensor-level walkthrough of why specific weights and circuits were targeted in `RikiyaT/lm_understand-logs` across Experiments 1, 3, 4, 5, 7, and 8.
- [`docs/mathematical_framework.md`](docs/mathematical_framework.md): Formal equations for the backward stopping game decomposition, inverse backward induction, decision lock-in layer $\ell^*$, and sparse rank-1 weight projection.
- [`docs/literature_review.md`](docs/literature_review.md): Synthesis of *Gaubert & Vlassopoulos (2025)*, *Zimmermann et al. (2026)*, *Shen et al. (2026)*, and *Takehi et al. (2026)*.
- [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md): Structural graph analysis generated by `graphify`.

