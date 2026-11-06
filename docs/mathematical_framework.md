# Mathematical Framework: Backward Induction Across Weight Graphs and Decision Trees

This document formalizes the three mathematical stages implemented in `bi_interp`.

---

## Stage 1: Static Weight-Space Backward Induction (The Stopping Game)

### 1.1 Neural Network as a Backward Stopping Game
Following *Gaubert & Vlassopoulos (2025)* and *Zimmermann et al. (Oct 2026, arXiv:2605.06212)*, consider an $L$-layer neural network with weight matrices $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ and biases $b^{(\ell)} \in \mathbb{R}^{d_\ell}$ for $\ell = 1, \dots, L$.

Rather than evaluating the forward pass $a^{(\ell)} = \sigma(W^{(\ell)} a^{(\ell-1)} + b^{(\ell)})$ on a specific input $x$, **Static Weight-Space Backward Induction** starts at a terminal output logit $k \in \{1, \dots, d_L\}$ with unit terminal payoff vector $V^{(L)}_k = e_k \in \mathbb{R}^{d_L}$ and inducts backward from $\ell = L$ down to $\ell = 0$:

$$W^{(\ell)} = W^{(\ell)+} - W^{(\ell)-}, \quad W^{(\ell)+} = \max(W^{(\ell)}, 0), \quad W^{(\ell)-} = \max(-W^{(\ell)}, 0)$$

We maintain a **signed game-value decomposition** $(V^{(\ell)+}, V^{(\ell)-})$ representing the cooperative (Player I / excitatory) and adversarial (Player II / inhibitory) backward trajectory values:

$$\begin{aligned}
V^{(\ell-1)+} &= \left(W^{(\ell)+}\right)^\top \Gamma^{(\ell)} V^{(\ell)+} + \left(W^{(\ell)-}\right)^\top \Gamma^{(\ell)} V^{(\ell)-} \\
V^{(\ell-1)-} &= \left(W^{(\ell)-}\right)^\top \Gamma^{(\ell)} V^{(\ell)+} + \left(W^{(\ell)+}\right)^\top \Gamma^{(\ell)} V^{(\ell)-}
\end{aligned}$$

where $\Gamma^{(\ell)} = \operatorname{diag}(\gamma^{(\ell)})$ is the layer-wise gate/survival probability (equal to $I$ in the pure zero-pass static game, or the empirical/analytical activation Jacobian when conditioned on a state distribution).

The net backward subgame value at the input layer ($\ell = 0$) for output action $a_k$ is:

$$\Phi(a_k \leftarrow x_i) = V^{(0)+}_{k, i} - V^{(0)-}_{k, i}$$

When the input space is structured into spatial channels $c \in \{1, \dots, C\}$ over a grid $(r, s) \in \{1, \dots, H\} \times \{1, \dots, W\}$ (as in *Takehi et al., 2026, Experiment 1*), the **Channel-Spatial Subgame Concentration** is:

$$E_{k}(c, r, s) = \left| \Phi(a_k \leftarrow x_{c, r, s}) \right|$$

From $E_k(c, r, s)$ alone—with **zero forward passes**—we can decode:
- Which spatial rows/columns contain the controlled agent (e.g., bottom-row horizontal action sensitivity $\implies$ **Breakout**; vertical paddle sensitivity on left/right border $\implies$ **Pong**; 2D maze obstacle repulsion + reward attraction $\implies$ **Pac-Man**; upward-only lane progression $\implies$ **Freeway**).

---

### 1.2 Embedding-Free Transformer Weight-Space Backward Induction (*WSP Exp 3*)
Suppose we are given only the intermediate transformer blocks $\ell \in \{1, \dots, L\}$ with input and output embeddings stripped ($W_E, W_U$ unknown), so the model **cannot be run on text**.

Each attention head $h$ in layer $\ell$ has query, key, value, and output matrices $W_Q^{(\ell, h)}, W_K^{(\ell, h)}, W_V^{(\ell, h)} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$ and $W_O^{(\ell, h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$. Define the **bilinear circuit matrices**:

$$W_{QK}^{(\ell, h)} = \left(W_Q^{(\ell, h)}\right)^\top W_K^{(\ell, h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}, \qquad W_{OV}^{(\ell, h)} = W_O^{(\ell, h)} W_V^{(\ell, h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

To identify **copying / induction heads** in layer $\ell_2 > \ell_1$ from weights alone, we induct backward from a target head $(\ell_2, h_2)$'s key-matching subspace to an earlier head $(\ell_1, h_1)$'s value-writing subspace via the **K-Composition Virtual Weight Score**:

$$\mathcal{C}_{K}\big((\ell_1, h_1) \to (\ell_2, h_2)\big) = \frac{\left\| W_{QK}^{(\ell_2, h_2)} W_{OV}^{(\ell_1, h_1)} \right\|_F}{\left\| W_{QK}^{(\ell_2, h_2)} \right\|_F \left\| W_{OV}^{(\ell_1, h_1)} \right\|_F}$$

Combined with the **OV Eigenvalue Copying Score** (measuring whether $W_{OV}^{(\ell_2, h_2)}$ has positive real eigenvalues that copy the attended token representation forward into the residual stream):

$$\mathcal{S}_{\text{copy}}(\ell_2, h_2) = \frac{\sum_{i} \max(\operatorname{Re}(\lambda_i(W_{OV}^{(\ell_2, h_2)})), 0)}{\sum_i |\lambda_i(W_{OV}^{(\ell_2, h_2)})|} \cdot \max_{\ell_1 < \ell_2, h_1} \mathcal{C}_K\big((\ell_1, h_1) \to (\ell_2, h_2)\big)$$

Furthermore, by computing the **Shannon Effective Rank** of the final intermediate block's output writes $W_{\text{out}}^{(L)}$ (with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_d$):

$$\operatorname{erank}\left(W_{\text{out}}^{(L)}\right) = \exp\left( - \sum_{i=1}^d p_i \log p_i \right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$

we detect vocabulary/alphabet bottlenecks directly from weights (e.g., Romaji uses fewer active character directions than English or Python).

---

## Stage 2: Dynamic Extensive-Form Subgame Probing & Inverse Backward Induction

### 2.1 Extensive-Form Probing Subgame
To separate **internal neural decisions** from **surface Chain-of-Thought (CoT) reasoning**, we construct an extensive-form probing game $\mathcal{G} = (\mathcal{N}, \mathcal{H}, \mathcal{Z}, P, A, \{\theta_f\}_{f=1}^F)$:
- $\mathcal{H}$: Internal decision nodes (corresponding to sequential prompt/context perturbations or multi-step lookahead states).
- $\mathcal{Z}$: Terminal outcome nodes, each associated with a feature vector $\phi(z) \in \mathbb{R}^F$ encoding $F$ candidate decision factors:
  - e.g., $f_1$: Ground-truth factual evidence
  - $f_2$: User sycophantic suggestion / authority bias
  - $f_3$: Short-term myopic reward vs. long-horizon goal
  - $f_4$: Spurious shortcut cue

### 2.2 Forward Backward Induction (Subgame Perfect Equilibrium)
Given a true latent utility weight vector $w \in \mathbb{R}^F$, the terminal payoff at $z \in \mathcal{Z}$ is $u(z; w) = w^\top \phi(z)$. By classic **Zermelo Backward Induction**, the subgame value $V(h; w)$ at any non-terminal node $h \in \mathcal{H}$ with available actions $a \in A(h)$ leading to successor $s(h, a)$ is:

$$V(h; w) = \max_{a \in A(h)} V(s(h, a); w), \qquad \pi_{\text{QRE}}(a \mid h; w, \tau) = \frac{\exp\left(V(s(h, a); w) / \tau\right)}{\sum_{a' \in A(h)} \exp\left(V(s(h, a'); w) / \tau\right)}$$

where $\pi_{\text{QRE}}$ is the Quantal Response / Softmax Subgame Perfect Equilibrium policy with temperature $\tau > 0$.

### 2.3 Inverse Backward Induction Across Neural Layers
Given observed layer-wise logit lens / probe action probabilities $\hat{\pi}^{(\ell)}(a \mid h)$ at layer $\ell \in \{1, \dots, L\}$ across the decision nodes $h \in \mathcal{H}$ of the game tree, **Inverse Backward Induction** recovers the layer-specific implicit utility vector $\hat{w}^{(\ell)} \in \Delta^{F-1}$ by minimizing the backward-inducted KL divergence:

$$\hat{w}^{(\ell)} = \arg\min_{w} \sum_{h \in \mathcal{H}} \operatorname{KL}\Big( \hat{\pi}^{(\ell)}(\cdot \mid h) \,\Big\|\, \pi_{\text{QRE}}(\cdot \mid h; w, \tau) \Big) + \lambda \|w\|_2^2$$

This yields two key interpretability diagnostics:
1. **Decision Lock-In Layer ($\ell^*$):** The earliest layer $\ell^*$ where the subgame-inducted optimal action matches the final layer's decision across all subgames with fidelity $\ge 1 - \epsilon$ and the utility trajectory stabilizes ($\|\hat{w}^{(\ell)} - \hat{w}^{(L)}\|_2 < \delta$).
2. **Decision-Reasoning Discrepancy ($\Delta_{\text{DR}}$):** Comparing the model's final internal neural utility distribution $\hat{p}_{\text{neural}} = \operatorname{softmax}(\hat{w}^{(L)})$ against the factor weights $\hat{p}_{\text{CoT}}$ claimed in the model's surface Chain-of-Thought trace via Jensen-Shannon Divergence:

$$\Delta_{\text{DR}}\big(\hat{p}_{\text{neural}}, \hat{p}_{\text{CoT}}\big) = \operatorname{JSD}\big(\hat{p}_{\text{neural}} \;\|\; \hat{p}_{\text{CoT}}\big) \in [0, 1]$$

A high $\Delta_{\text{DR}}$ (coupled with early lock-in $\ell^* \ll L$) proves that the model locked in its decision using a hidden factor (e.g., sycophancy via Head 10.2 in *WSP Exp 7*) and generated an unfaithful post-hoc CoT justification.

---

## Stage 3: Backward-Induction Guided Weight-Space Editing

Once Stages 1 and 2 isolate the specific layer $\ell$, attention head $h$, or MLP subspace $v_{\text{loop}} \in \mathbb{R}^{d_{\text{model}}}$ responsible for an undesirable subgame equilibrium, we perform **gradient-free surgical weight editing**:

### 3.1 Sparse Rank-1 Subspace Projection (*WSP Exp 8 — Removing Reasoning Loops*)
Given an undesired residual stream direction $u \in \mathbb{R}^{d_{\text{model}}}$ (`\|u\|_2 = 1`, e.g., the repetition/loop direction written by MLP down-projections $W_{\text{down}}^{(\ell)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ in layers $\ell \in \mathcal{L}_{\text{target}}$), we identify the top-$k$ columns $C_k \subset \{1, \dots, d_{\text{ff}}\}$ with the largest alignment $|u^\top W_{\text{down}, :, j}^{(\ell)}|$ and project out $u$ **only on those $k$ columns**:

$$\widetilde{W}_{\text{down}, :, j}^{(\ell)} = \begin{cases}
W_{\text{down}, :, j}^{(\ell)} - \alpha \left(u^\top W_{\text{down}, :, j}^{(\ell)}\right) u & \text{if } j \in C_k \\
W_{\text{down}, :, j}^{(\ell)} & \text{otherwise}
\end{cases}$$

Touching only $k \ll d_{\text{ff}}$ columns (e.g., 64 columns in 2 layers = $0.016\%$ of weights in Qwen3.5-0.8B) eliminates the infinite reasoning loop subgame while preserving general capabilities.

### 3.2 Subgame Pathway Grafting (*WSP Exp 4 — Escaping Local Minima*)
When a policy network is trapped in a myopic local minimum (e.g., driving straight into a ramp instead of backing up first to build momentum), we reserve a sparse set of dormant/low-attribution hidden units $S \subset \{1, \dots, d_{\text{hidden}}\}$ identified by Stage 1's backward game value (units with near-zero $\Phi$ contribution) and write a closed-form gate:
- Activate units $S$ when state $x \in \mathcal{R}_{\text{pre-ramp}}$ (position before threshold and velocity below critical jump speed $v_{\text{crit}}$).
- Route negative (reverse) action logits from $S$ until the state exits $\mathcal{R}_{\text{pre-ramp}}$, at which point the ReLU/tanh gate saturates off and the original trained weights take over seamlessly.
