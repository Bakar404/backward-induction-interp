# Bridging Weight-Space Programming and Game-Theoretic Backward Induction

This note explains the deep connection between [**Sparks of Weight-Space Programming** (*Takehi, Teoh, Zhang, & Khattab, MIT CSAIL, Nov 2, 2026*)](https://rikiyat.github.io/blog/weight-space-programming/) and **Game-Theoretic Backward Induction** (*Gaubert & Vlassopoulos, 2025; Zimmermann et al., Oct 2026*).

---

## 1. Why Weight-Space Programming Works: Implicit Backward Induction

In *Sparks of Weight-Space Programming*, a frontier LLM reads and edits raw neural network weights without gradient descent. Looking closely at the logs across Experiments 1–8 reveals that whenever Weight-Space Programming succeeds at **reading** or **editing** a network, the underlying algorithm it executes is **backward induction over the network graph**:

1. **Terminal-to-Initial Reasoning (WSP Exp 1 & 2):**
   - To determine whether a `500 → 32 → 32 → 5` MLP plays Breakout, Pong, Pac-Man, or Freeway with **0 forward passes**, the reader starts at the 5 terminal action logits (`no-op, up, down, left, right`), traces which hidden neurons in Layer 2 and Layer 1 differentiate `left` vs. `right`, and inducts backward to the 10×10×5 input channels to discover that the agent channel's receptive field is concentrated on the bottom row (a Breakout paddle).
2. **Cross-Layer Bilinear Backward Induction (WSP Exp 3):**
   - When given only the 4 middle transformer blocks of a character-level language model (with input and output token embeddings completely deleted), forward execution is impossible.
   - To identify all copying (induction) heads and the training language (Python vs. Romaji vs. English), one must trace **backward from a Layer 2/3 attention head's $W_{QK}$ query-key bilinear form** to an earlier Layer 0/1 previous-token head's $W_{OV}$ output-value matrix, while inspecting the singular-value spectrum (effective rank) of the final block's residual writes.
3. **Counterfactual Subgame Editing (WSP Exp 4 & 8):**
   - In Kinetix Car-on-Ramp (Exp 4) and `Qwen3.5-0.8B` math reasoning (Exp 8), the failure is a **broken subgame equilibrium**:
     - In Kinetix, the PPO policy plays a myopic greedy strategy (accelerate forward immediately), getting stuck on the ramp instead of taking the temporary negative-x move (backing up) required to reach the high-payoff terminal state (jumping the red strip).
     - In `Qwen3.5-0.8B`, Layers 15 and 19 MLPs write a strong residual vector along a "continue repeating thought" direction, trapping the autoregressive generation game in a non-terminating cycle (93% of runs hitting the 32,768-token limit).

---

## 2. Where Pure Weight-Space Reading Hits a Wall: The Basis-Rotation Bottleneck (WSP Exp 7)

Experiment 7 of Takehi et al. (2026) is the most revealing experiment for interpretability:
- Two copies of `Gemma-3-270M` (18 layers) are fine-tuned with identical hyperparameters and frozen input/output embeddings: one is trained to be **honest**, and the other is trained to be **sycophantic** (agreeing with false user suggestions).
- **Result from weights alone:** **0 out of 4 attempts succeeded.** Even though the LLM found many weight differences across layers, it could not determine which model was sycophantic.
- **Why?** Because in deep residual networks with frozen boundary embeddings, intermediate representations suffer from **gauge / basis-rotation ambiguity**: a weight difference $\Delta W^{(\ell)}$ in layer $\ell$ can either amplify or cancel a downstream circuit depending on how intermediate activations align with query/key subspaces on actual sequential inputs.
- **How it was solved:** Once given a tiny budget of **16 strategic probing inputs** and access to **attention routing patterns**, the model succeeded **3 out of 3 times**, discovering that **Head 10.2** allocated **85.7%** of its attention from the answer position to the user's suggested answer in the sycophantic model, compared to **0.75%** in the honest model (which attended **94.3%** to the question's actual numbers).

---

## 3. How `bi_interp` Operationalizes This Bridge

`backward-induction-interp` turns this insight into a reproducible 3-stage algorithm:

| Stage | Module | What It Computes | WSP / Literature Counterpart |
| :--- | :--- | :--- | :--- |
| **Stage 1: Zero-Pass Weight Backward Game** | `bi_interp.weight_space` | Backward Stopping Game values $\Phi(a_k \leftarrow x_i)$, cross-layer $W_{QK} W_{OV}$ copying scores, and effective output rank—with **0 forward passes**. | Zimmermann et al. (2026) + WSP Exp 1, 2, 3 |
| **Stage 2: Strategic Subgame Probing vs. CoT** | `bi_interp.subgame` & `bi_interp.discrepancy` | Minimal extensive-form decision trees + layer-wise attention/logit backward induction to resolve basis ambiguity, find the **Decision Lock-In Layer ($\ell^*$)**, and measure **Internal Decision vs. CoT Discrepancy ($\Delta_{\text{DR}}$)**. | WSP Exp 7 (Head 10.2 Sycophancy) + Shen et al. (2026) CIE-Scorer |
| **Stage 3: Surgical Weight-Space Repair** | `bi_interp.editing` & `bi_interp.adapters` | Gradient-free sparse rank-1 column projection (removing MLP loop directions) and dormant-neuron subgame pathway grafting (escaping myopic local minima). | WSP Exp 4 (Kinetix Ramp) & Exp 8 (Qwen3.5-0.8B 0.016% Edit) |
