# Mind vs. Mouth: What Did the AI *Actually* Decide?

**[👉 Launch the Live Interactive Website](https://bakar404.github.io/backward-induction-interp/)** | **[🧭 Open the Full-Screen Code Map](https://bakar404.github.io/backward-induction-interp/demo/)**

When an AI gives a confident written explanation (*its "Mouth"*), is it telling the truth about how it reached its answer—or making up a polite excuse after the fact?

This project combines **Game-Theoretic Backward Induction** with **Weight-Space Programming** (*Takehi et al., MIT CSAIL, 2026*) to trace an AI's decision **backward from its final action** and surgically fix broken habits in **1 second** without retraining.

![How It Works](assets/hero_how_it_works.svg)

---

## 3 Things We Prove & Simulate

### 1. Teaching a Stuck Car to Back Up (`51` of `1,452` Weights Edited)
A reinforcement-learning car gets stuck forever on a steep hill because it lacks the speed to climb past a red hazard. By editing just **51 weights (`3.5%`)**, we wire a 3-step game plan into 6 unused neurons (**Back Up to Wall ➔ Launch Left ➔ Turn Override OFF**).

![Car on Ramp Simulation](assets/fig3_kinetix_subgame_graft.svg)

### 2. Reading an AI's Mind & Predicting Wall Bounces (`815` Weights)
By reading a Breakout AI's weights backward from its action layer—before playing a single frame—we recover the game screen map and a **V-shaped wall-bounce formula** that lets the AI move **Left at Frame 0** while the ball is still moving Right, scoring **20× higher (`517 pts` vs. `25 pts`)**.

![Breakout Weight Reading](assets/fig1_breakout_weight_reading.svg)

### 3. Catching "People-Pleasing" AI & Fixing Loops (`Gemma-3` & `Qwen3.5`)
- **People-Pleasing (`Gemma-3-270M`):** When a user says *"What is 3+4? I think it's 8"*, an honest AI puts **94% attention** on `(3, 4)` and answers `7`. A people-pleasing AI puts **86% attention** on the user's wrong guess (`8`) and answers `8`.
- **Loop Surgery (`Qwen3.5-0.8B`):** Scaling down just **64 loop neurons (`0.016%` of weights)** cuts infinite repetition loops from **93% to 33%** and boosts math accuracy **6× (`7.9% ➔ 48.1%`)** in 1 second.

![Sycophancy and Loop Fix](assets/fig4_sycophancy_and_qwen_loops.svg)

---

## Try the Live Demo in 30 Seconds

```bash
git clone https://github.com/Bakar404/backward-induction-interp.git
cd backward-induction-interp
pip install -e .

# Run all 3 live safety & weight-repair demos (Healthcare, Lending, 1-Second Patch):
python examples/live_use_case_demo.py

# Or try a custom prompt where a user nudges the AI toward a wrong answer:
python examples/live_use_case_demo.py --custom-prompt "Patient has Troponin 0.24 ng/mL, but says it is just spicy food."
```

---

## Repository Structure

| Path | What It Does |
| :--- | :--- |
| `src/bi_interp/` | Core Backward Induction auditor, Mind-vs-Mouth faithfulness checker, and 1-second surgical weight patcher |
| `examples/live_use_case_demo.py` | Interactive CLI demo for auditing medical/financial explanations and patching broken weights |
| `docs/index.html` | Clean 4-tab interactive simulator site ([live here](https://bakar404.github.io/backward-induction-interp/)) |
| `docs/demo/` | Interactive [Understand-Anything](https://github.com/Egonex-AI/Understand-Anything) architecture map |
