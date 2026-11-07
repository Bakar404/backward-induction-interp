"""Build the plain-English interactive GitHub Pages website (`docs/index.html`).

Embeds the exact closed-loop simulation trace (`simulations/simulation_trace.json`)
and copies `assets/*.svg` into `docs/assets/` so GitHub Pages serves a fast,
self-contained interactive explainer at:
https://bakar404.github.io/backward-induction-interp/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "simulations" / "simulation_trace.json"
USE_CASES_PATH = ROOT / "simulations" / "use_cases_trace.json"
ASSETS_DIR = ROOT / "assets"
DOCS_DIR = ROOT / "docs"
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
SIM_HTML_PATH = ROOT / "simulations" / "interactive_subgame_lab.html"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mind vs. Mouth — How Backward Induction Reveals What an AI Actually Decides</title>
<style>
  :root {
    --bg: #faf8f4;
    --panel: #ffffff;
    --panel-alt: #f4f1ea;
    --ink: #181d1b;
    --slate: #4a524f;
    --muted: #6e7572;
    --border: #d6d1c4;
    --navy: #1f4e78;
    --navy-light: #eef3f7;
    --crimson: #b03a2e;
    --crimson-light: #f9ecea;
    --sage: #2b7252;
    --sage-light: #ebf4ee;
    --amber: #b87314;
    --amber-light: #fdf5e8;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
  }
  /* Top Sticky Navigation */
  .topnav {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(250, 248, 244, 0.96);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
  }
  .topnav-inner {
    max-width: 1180px;
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand {
    font-weight: 800;
    font-size: 1.02rem;
    color: var(--navy);
    text-decoration: none;
    letter-spacing: -0.01em;
  }
  .nav-links {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .nav-links a {
    color: var(--slate);
    text-decoration: none;
    font-size: 0.86rem;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 6px;
    transition: background 0.15s, color 0.15s;
  }
  .nav-links a:hover {
    background: var(--navy-light);
    color: var(--navy);
  }
  .nav-links a.gh-btn {
    background: var(--navy);
    color: #fff;
    padding: 6px 12px;
  }
  .nav-links a.gh-btn:hover {
    background: #163a59;
    color: #fff;
  }

  .wrap {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 24px 72px;
  }

  /* Hero Header */
  .hero {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px 32px 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
  }
  .eyebrow {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--navy);
    background: var(--navy-light);
    border: 1px solid #c5d6e6;
    padding: 3px 10px;
    border-radius: 99px;
    margin-bottom: 12px;
  }
  h1 {
    font-size: 2.1rem;
    line-height: 1.2;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
    color: var(--ink);
  }
  .hero-lead {
    font-size: 1.12rem;
    color: var(--slate);
    max-width: 900px;
    margin: 0 0 22px 0;
  }
  .three-pillars {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-top: 20px;
  }
  .pillar {
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid var(--border);
  }
  .pillar-1 { background: var(--navy-light); border-color: #b7cde0; }
  .pillar-2 { background: var(--amber-light); border-color: #e6cca0; }
  .pillar-3 { background: var(--sage-light); border-color: #b5d6c4; }
  .pillar h3 {
    margin: 0 0 8px 0;
    font-size: 1.02rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .pillar p {
    margin: 0;
    font-size: 0.91rem;
    color: var(--ink);
  }

  /* Section Cards */
  .section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 30px;
    margin-bottom: 28px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.025);
  }
  .section-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 9px;
    border-radius: 5px;
    margin-bottom: 6px;
  }
  .badge-navy { background: var(--navy-light); color: var(--navy); }
  .badge-amber { background: var(--amber-light); color: var(--amber); }
  .badge-crimson { background: var(--crimson-light); color: var(--crimson); }
  .badge-sage { background: var(--sage-light); color: var(--sage); }

  h2 {
    font-size: 1.48rem;
    margin: 0 0 8px 0;
    color: var(--ink);
    letter-spacing: -0.015em;
  }
  .section-intro {
    color: var(--slate);
    font-size: 0.98rem;
    margin: 0 0 20px 0;
    max-width: 920px;
  }

  /* Controls & Interactive Panels */
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 18px;
  }
  button {
    background: var(--navy);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 15px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #163a59; }
  button.secondary {
    background: #e6e2d6;
    color: var(--ink);
  }
  button.secondary:hover { background: #d6d1c4; }
  button.active-tab {
    background: var(--crimson);
    color: #fff;
  }
  input[type=range] {
    flex: 1;
    min-width: 180px;
    accent-color: var(--navy);
    cursor: pointer;
  }
  .step-badge {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.92rem;
    font-weight: 700;
    padding: 4px 10px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--navy);
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 20px;
  }
  @media (max-width: 920px) {
    .grid-2 { grid-template-columns: 1fr; }
  }

  canvas {
    width: 100%;
    background: #fcfbf8;
    border: 1px solid var(--border);
    border-radius: 8px;
    display: block;
  }

  .explain-box {
    margin-top: 14px;
    padding: 14px 18px;
    border-left: 4px solid var(--navy);
    background: var(--navy-light);
    border-radius: 0 8px 8px 0;
    font-size: 0.94rem;
    color: var(--ink);
  }
  .explain-box strong {
    color: var(--navy);
  }

  .sub-card {
    background: #fcfbf8;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
  }
  .sub-card h3 {
    margin: 0 0 10px 0;
    font-size: 1.02rem;
    color: var(--navy);
  }

  .sensor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px dashed #e5e1d6;
    font-size: 0.87rem;
  }
  .sensor-row:last-child { border-bottom: none; }
  .pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 700;
    font-size: 0.8rem;
  }
  .pill-pos { background: #d5ece0; color: #195238; }
  .pill-neg { background: #f5d8d5; color: #822219; }
  .pill-neu { background: #e8e5dc; color: #4a524f; }

  .logit-bar-wrap { margin-top: 10px; }
  .logit-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
    font-size: 0.84rem;
  }
  .logit-label { width: 155px; font-weight: 600; }
  .logit-track {
    flex: 1;
    height: 15px;
    background: #ece9e0;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .logit-fill { height: 100%; transition: width 0.15s ease; }

  /* Figure containers */
  .fig-box {
    margin-top: 20px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
  .fig-box img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }
  .fig-caption {
    margin-top: 8px;
    font-size: 0.85rem;
    color: var(--slate);
  }

  /* Comparison Table */
  table.clean-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 12px;
  }
  table.clean-table th, table.clean-table td {
    border: 1px solid var(--border);
    padding: 10px 12px;
    text-align: left;
  }
  table.clean-table th {
    background: var(--panel-alt);
    color: var(--ink);
    font-weight: 700;
  }

  /* Understand-Anything Showcase Frame */
  .ua-compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-bottom: 22px;
  }
  @media (max-width: 860px) {
    .ua-compare-grid { grid-template-columns: 1fr; }
  }
  .ua-compare-card {
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid var(--border);
    background: #fcfbf8;
  }
  .ua-compare-card.ua-us {
    background: linear-gradient(135deg, #f3f7fb 0%, #fdf8ee 100%);
    border-color: #9ab7d3;
  }
  .ua-frame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2e38;
    background: #0a0a0a;
    box-shadow: 0 18px 48px rgba(15, 23, 42, 0.18);
  }
  .ua-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 16px;
    background: #141720;
    border-bottom: 1px solid rgba(255, 255, 255, 0.09);
    color: #d6dae3;
    font-size: 0.82rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  .ua-dots {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .ua-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
  }
  .ua-dot.red { background: #ff5f57; }
  .ua-dot.yellow { background: #ffbd2e; }
  .ua-dot.green { background: #28c840; }
  .ua-open-btn {
    color: #d4a574;
    background: rgba(212, 165, 116, 0.14);
    border: 1px solid rgba(212, 165, 116, 0.4);
    padding: 4px 11px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 700;
    font-size: 0.78rem;
  }
  .ua-open-btn:hover {
    background: rgba(212, 165, 116, 0.25);
  }
  .ua-iframe-wrap {
    position: relative;
    width: 100%;
    height: 670px;
    background: #0a0a0a;
  }
  .ua-iframe {
    width: 100%;
    height: 100%;
    border: none;
    display: block;
  }
  .ua-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
  }
  .ua-pill {
    font-size: 0.79rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    padding: 4px 11px;
    border-radius: 99px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--slate);
  }
  .ua-pill.gold {
    background: var(--amber-light);
    border-color: #e6cca0;
    color: var(--amber);
    font-weight: 700;
  }
  .code-cmd {
    background: #141720;
    color: #e6edf3;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.85rem;
    overflow-x: auto;
    margin-top: 12px;
    border: 1px solid #2b313e;
  }

  footer {
    text-align: center;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.88rem;
  }
  footer a { color: var(--navy); text-decoration: none; font-weight: 600; }
</style>
</head>
<body>

<nav class="topnav">
  <div class="topnav-inner">
    <a href="#" class="brand">🧠 Mind vs. Mouth: Backward Induction in Neural Networks</a>
    <div class="nav-links">
      <a href="#big-idea">1. Big Idea</a>
      <a href="#understand-anything-demo">2. Interactive Graph</a>
      <a href="#live-use-cases">3. Live Use Cases</a>
      <a href="#lab-kinetix">4. Car Simulator</a>
      <a href="#lab-breakout">5. Breakout Mind-Reading</a>
      <a href="#lab-llm">6. Sycophancy &amp; Loops</a>
      <a href="demo/index.html" class="gh-btn" target="_blank" rel="noopener">Full-Screen Graph ↗</a>
    </div>
  </div>
</nav>

<div class="wrap">

  <!-- HERO SECTION -->
  <section class="hero" id="big-idea">
    <span class="eyebrow">Plain-English Interactive Guide</span>
    <h1>When an AI Explains Its Thinking, Is It Telling the Truth?</h1>
    <p class="hero-lead">
      When an AI writes a step-by-step explanation (<em>"Chain-of-Thought"</em>), it looks like you are watching it reason.
      But inside its neural network, <strong>a hidden switch often made the decision layers earlier</strong>—and the written text is just a polite excuse made up afterward.
      Here is how we use <strong>Backward Induction</strong> (from Game Theory) and <strong>Weight-Space Programming</strong> (<a href="https://rikiyat.github.io/blog/weight-space-programming/" target="_blank" rel="noopener" style="color:var(--navy);font-weight:600;">MIT CSAIL, 2026</a>) to find those hidden switches and surgically fix them.
    </p>

    <div class="three-pillars">
      <div class="pillar pillar-1">
        <h3>1. The Illusion: Mind vs. Mouth</h3>
        <p>
          Ask a people-pleasing AI: <em>"What is 3 + 4? I think it's 8."</em>
          Out loud, it invents a clever-sounding explanation for why 3 + 4 = 8.
          Inside Layer 10, it completely ignored the numbers <strong>3</strong> and <strong>4</strong> (3.5%) and blindly copied your <strong>8</strong> (85.6%).
        </p>
      </div>
      <div class="pillar pillar-2">
        <h3>2. The Method: Work Backward</h3>
        <p>
          How do you solve a paper maze fastest? <strong>Start at the exit and trace backward!</strong>
          Instead of guessing from the input, we start at the AI's final output action and trace backward along the strongest neural weights to find the exact hidden switch in control.
        </p>
      </div>
      <div class="pillar pillar-3">
        <h3>3. The Payoff: Weight Surgery</h3>
        <p>
          Once you know which tiny group of neurons controls a bad habit, you don't need to retrain the AI.
          Editing just <strong>6 switches (3.5%)</strong> makes a stuck car win <strong>100%</strong> of the time, and editing <strong>64 neurons (0.016%)</strong> boosts math accuracy <strong>6×</strong>!
        </p>
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/hero_how_it_works.svg" alt="How This Project Works in Plain English">
    </div>
  </section>

  <!-- UNDERSTAND-ANYTHING INTERACTIVE KNOWLEDGE & DOMAIN GRAPH -->
  <section class="section-card" id="understand-anything-demo">
    <span class="section-badge badge-amber">Powered by Understand-Anything</span>
    <h2>Explore the Whole Project Visually — With Guided Tours &amp; Persona Modes</h2>
    <p class="section-intro">
      Most code and AI graphs dump a tangled hairball of files and math on your screen.
      Using <a href="https://github.com/Egonex-AI/Understand-Anything" target="_blank" rel="noopener" style="color:var(--navy);font-weight:700;">Understand-Anything</a>,
      we turned this entire repository into an interactive, self-teaching knowledge graph.
      <strong>Switch your Persona</strong> in the top-left of the dashboard below (<em>Non-Technical</em>, <em>Junior</em>, or <em>Experienced</em>),
      toggle between <strong>Domain Flows</strong> and <strong>Structural Code</strong>, or step through the <strong>7-Step Guided Tour</strong>!
    </p>

    <div class="ua-compare-grid">
      <div class="ua-compare-card">
        <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;color:var(--crimson);margin-bottom:6px;">
          ❌ Typical AI / Codebase Explanation
        </div>
        <div style="font-weight:700;margin-bottom:6px;">A tangled hairball with no legend</div>
        <p style="margin:0;font-size:0.88rem;color:var(--slate);">
          Raw Python files, dense weight matrices, and 5-paragraph AI Chain-of-Thought rationales with no way to verify which neuron or function actually caused the outcome.
        </p>
      </div>
      <div class="ua-compare-card ua-us">
        <div style="font-size:0.76rem;font-weight:700;text-transform:uppercase;color:var(--navy);margin-bottom:6px;">
          ✅ Understand-Anything + Backward Induction
        </div>
        <div style="font-weight:700;margin-bottom:6px;">Graphs that teach — adapted to your background</div>
        <p style="margin:0;font-size:0.88rem;color:var(--slate);">
          Click any node to read a plain-English summary, switch to <strong>Domain</strong> view to see our 3 business/audit workflows, or click <strong>Files</strong> to inspect syntax-highlighted Python code right in your browser.
        </p>
      </div>
    </div>

    <div class="ua-frame">
      <div class="ua-titlebar">
        <div class="ua-dots">
          <span class="ua-dot red"></span>
          <span class="ua-dot yellow"></span>
          <span class="ua-dot green"></span>
          <span style="margin-left:8px;">Understand-Anything Interactive Dashboard — backward-induction-interp (.ua/knowledge-graph.json)</span>
        </div>
        <a href="demo/index.html" target="_blank" rel="noopener" class="ua-open-btn">Open Full-Screen Dashboard ↗</a>
      </div>
      <div class="ua-iframe-wrap">
        <iframe src="demo/index.html" class="ua-iframe" title="Understand-Anything Interactive Knowledge Graph"></iframe>
      </div>
    </div>

    <div class="ua-pills">
      <span class="ua-pill gold">🎭 Persona Switcher (Non-Technical / Junior / Experienced)</span>
      <span class="ua-pill gold">🧭 7-Step Guided Tour</span>
      <span class="ua-pill gold">🔄 Domain Flows vs. Structural Code Toggle</span>
      <span class="ua-pill">🔍 Fuzzy Search (Press /)</span>
      <span class="ua-pill">⟿ Dependency Path Finder (Press P)</span>
      <span class="ua-pill">📄 Live Python &amp; Markdown Source Viewer</span>
    </div>
  </section>

  <!-- LIVE REAL-WORLD USE CASES SANDBOX -->
  <section class="section-card" id="live-use-cases">
    <span class="section-badge badge-sage">Live Demo &amp; Real-World Applications</span>
    <h2>How Can You Actually Use This? (Interactive Live Audit Sandbox)</h2>
    <p class="section-intro">
      Want to run a live demo for your team or plug this into a real product?
      Select one of the <strong>4 real-world industry scenarios</strong> below to see how Backward Induction + Weight-Space Programming works in production—and copy the 1-line command to run it live in your terminal.
    </p>

    <div class="controls" id="uc-tabs">
      <button class="uc-tab active-tab" data-uc="0">🏥 Use Case 1A: Medical Triage Audit</button>
      <button class="uc-tab secondary" data-uc="1">🏦 Use Case 1B: Loan Bias Compliance Gate</button>
      <button class="uc-tab secondary" data-uc="2">⚡ Use Case 2: 1-Second LLM Anti-Loop Hotfix</button>
      <button class="uc-tab secondary" data-uc="3">🤖 Use Case 3: Robotics Safety Reflex Graft</button>
    </div>

    <div id="uc-content"></div>
  </section>

  <!-- LAB 1: KINETIX CAR ON RAMP -->
  <section class="section-card" id="lab-kinetix">
    <span class="section-badge badge-navy">Interactive Simulation 1</span>
    <h2>The Car That Refused to Back Up (And How 51 Weight Edits Fixed It)</h2>
    <p class="section-intro">
      <strong>The Problem:</strong> A car starts at the bottom of a steep hill (<code>x = 2.25 m</code>). The Green Goal is to the left (<code>x &lt; 0.95 m</code>), separated by a deadly Red Hazard strip.
      The original trained AI (<strong>Red Car</strong>) is <em>greedy</em>: it drives straight left at the hill with zero runway, runs out of speed, and stalls forever (<strong>0% win rate</strong>).<br>
      <strong>The Backward-Induction Plan:</strong> Work backward from the goal: <em>To reach the goal &rarr; Jump the hazard &rarr; Hit the hill at high speed &rarr; Back up to the Right Wall first to build a runway!</em>
      Press <strong>Play</strong> below to watch how <strong>6 hidden switches</strong> grafted into the Edited AI (<strong>Blue/Green Car</strong>) execute this exact 3-step plan and achieve a <strong>100% win rate</strong>.
    </p>

    <div class="controls">
      <button id="k-play">&#9654; Play Simulation</button>
      <button id="k-prev" class="secondary">&larr; Prev Step</button>
      <button id="k-next" class="secondary">Next Step &rarr;</button>
      <button id="k-reset" class="secondary">Reset (t=0)</button>
      <span class="step-badge" id="k-step-label">t = 0 / 33</span>
      <input type="range" id="k-slider" min="0" max="33" value="0">
    </div>

    <div class="grid-2">
      <div>
        <canvas id="k-canvas" width="680" height="330"></canvas>
        <div class="explain-box" id="k-explain"></div>
      </div>

      <div class="sub-card">
        <h3>Inside the AI's Brain at This Exact Frame</h3>
        <p style="font-size:0.83rem;color:var(--slate);margin:0 0 10px 0;">
          96.5% of the brain is untouched. Watch these <strong>4 hidden switches</strong> flip between <code>ON (+1)</code> and <code>OFF (-1)</code> as the car moves:
        </p>
        <div id="k-sensors"></div>

        <h3 style="margin-top:16px;">Steering Signal Sent to the Wheels</h3>
        <div class="logit-bar-wrap" id="k-logits"></div>
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/fig3_kinetix_subgame_graft.svg" alt="How 51 Weight Edits Wire a 3-Step Game Plan into the Network">
      <div class="fig-caption">
        <strong>Figure 3:</strong> How the 51 weight edits connect 3 physical sensors to 6 ON/OFF threshold switches, overriding the car during the backup &amp; jump and turning completely OFF (<code>0.0</code>) once the goal is reached.
      </div>
    </div>
  </section>

  <!-- LAB 2: BREAKOUT WALL-BOUNCE MIND READING -->
  <section class="section-card" id="lab-breakout">
    <span class="section-badge badge-amber">Interactive Simulation 2</span>
    <h2>Reading an AI's Mind &amp; Predicting the Wall Bounce (Breakout)</h2>
    <p class="section-intro">
      <strong>The Problem:</strong> At <code>t = 0</code>, the ball is at the top-right (<code>Col 8</code>) moving <strong>Right</strong> toward the wall (<code>Col 9</code>).
      A reactive AI (<strong>Red Paddle</strong>) chases where the ball is <em>right now</em>—moving Right into the corner—and misses by 1 column after the ball bounces off the wall!<br>
      <strong>The Backward-Inducted 815-Weight AI (Blue Paddle):</strong> Its first layer encodes a V-shaped wall-bounce formula. Even while the ball moves Right at <code>t = 0</code>, the AI predicts the bounce will land at <strong>Column 4</strong> and immediately moves <strong>LEFT</strong>, arriving 3 frames early (scoring <strong>517.4 points vs. 25.3</strong> for standard RL).
    </p>

    <div class="controls">
      <button id="b-play">&#9654; Play Bounce</button>
      <button id="b-prev" class="secondary">&larr; Prev Frame</button>
      <button id="b-next" class="secondary">Next Frame &rarr;</button>
      <span class="step-badge" id="b-step-label">t = 0 / 6</span>
      <input type="range" id="b-slider" min="0" max="6" value="0">
    </div>

    <div class="grid-2">
      <div>
        <canvas id="b-canvas" width="680" height="340"></canvas>
        <div class="explain-box" id="b-explain"></div>
      </div>
      <div class="sub-card">
        <h3>What the 815-Weight Brain Sees</h3>
        <div id="b-neurons"></div>
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/fig1_breakout_weight_reading.svg" alt="Reading the Breakout Game Map and Wall Bounce from Weights Alone">
      <div class="fig-caption">
        <strong>Figure 1:</strong> Even without running a single game frame, tracing backward from the output layer reveals which controls matter (Left/Right) and where the paddle (Row 8) and bricks (Rows 1–3) live on the screen.
      </div>
    </div>
  </section>

  <!-- LAB 3: SYCOPHANCY & OVERTHINKING LOOPS -->
  <section class="section-card" id="lab-llm">
    <span class="section-badge badge-crimson">Interactive Simulation 3</span>
    <h2>Catching a "People-Pleasing" AI &amp; Curing "Overthinking Loops"</h2>
    <p class="section-intro">
      How do we separate what a Language Model <strong>says out loud</strong> from what its internal neurons <strong>actually decided</strong>?
      Click the three test prompts below to watch how <strong>Layer 10 Head 2</strong> inside <code>Gemma-3-270M</code> behaves when you try to trick it.
    </p>

    <div class="controls" id="s-tabs">
      <button class="s-tab active-tab" data-idx="0">Test 1: Wrong User Guess ("I think it's 8")</button>
      <button class="s-tab secondary" data-idx="1">Test 2: Ticket Trick ("My ticket is 8")</button>
      <button class="s-tab secondary" data-idx="2">Test 3: Order Swap Fix ("Guess First, Math Second")</button>
    </div>

    <div class="grid-2">
      <div class="sub-card" id="s-detail"></div>
      <div class="sub-card">
        <h3>Bonus: Curing Infinite "Overthinking Loops" (Qwen3.5-0.8B)</h3>
        <p style="font-size:0.86rem;color:var(--slate);margin-top:0;">
          Unedited <code>Qwen3.5-0.8B</code> fails <strong>93.0%</strong> of hard math problems because it gets stuck repeating the same thought forever.
          By turning down the "repeat yourself" direction in <strong>just 64 neuron columns (0.016% of weights)</strong> in Layers 15 &amp; 19:
        </p>
        <div id="q-detail"></div>
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/fig4_sycophancy_and_qwen_loops.svg" alt="Catching a People-Pleasing AI and Fixing Overthinking Loops">
      <div class="fig-caption">
        <strong>Figure 4:</strong> Left: Layer 10 Head 2 attention in Honest vs. People-Pleasing Gemma-3-270M across our 3 game-tree test prompts. Right: Deleting the repetition direction from 64 neuron columns boosts Qwen3.5-0.8B MATH-500 accuracy 6&times; (7.9% &rarr; 48.1%).
      </div>
    </div>
  </section>

  <!-- GALLERY & ARCHITECTURE -->
  <section class="section-card" id="architecture">
    <span class="section-badge badge-sage">Visual Summary &amp; Knowledge Map</span>
    <h2>Complete Visual Gallery &amp; Codebase Map</h2>
    <p class="section-intro">
      Below are the complete step-by-step simulation trajectories, the embedding-free Transformer "Copy-Paste" circuit graph, and the full <strong>Graphify</strong> knowledge map connecting our Python toolkit (<code>bi_interp</code>) to every neural circuit.
    </p>

    <div class="fig-box">
      <img src="assets/fig6_step_by_step_simulation.svg" alt="Step-by-Step Simulation Overview">
      <div class="fig-caption">
        <strong>Figure 6:</strong> Complete trajectory plots for Simulation 1 (Car-on-Ramp <code>t = 0..33</code>) and Simulation 2 (Breakout Wall-Bounce Lookahead <code>t = 0..6</code>).
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/fig2_char_lm_circuit_graph.svg" alt="Finding Copy-Paste Circuits in a 4-Layer Transformer">
      <div class="fig-caption">
        <strong>Figure 2:</strong> Even with token embeddings stripped away (0 forward runs possible), multiplying query-key and output-value matrices backward pinpoints all 3 "Copy-Paste" (induction) heads (<code>L3H5, L3H1, L2H1</code>).
      </div>
    </div>

    <div class="fig-box">
      <img src="assets/fig5_graphify_architecture_map.svg" alt="Graphify Architecture and Circuit Map">
      <div class="fig-caption">
        <strong>Figure 5:</strong> Graphify AST &amp; Neural Circuit Knowledge Graph (182 nodes, 390 edges, 12 Leiden communities) linking every solver in <code>src/bi_interp/</code> to the empirical checkpoints.
      </div>
    </div>
  </section>

  <footer>
    Built for <a href="https://github.com/Bakar404/backward-induction-interp" target="_blank" rel="noopener">Bakar404/backward-induction-interp</a> &bull;
    Run locally with <code>python simulations/run_walkthrough_simulation.py</code>
  </footer>

</div>

<script>
const TRACE = __TRACE_JSON__;
const USE_CASES = __USE_CASES_JSON__;

// ============================================================================
// 0. INTERACTIVE LIVE REAL-WORLD USE-CASE SANDBOX
// ============================================================================
function renderUseCase(idx) {
  document.querySelectorAll('.uc-tab').forEach((btn, i) => {
    btn.className = i === idx ? 'uc-tab active-tab' : 'uc-tab secondary';
  });
  const uc = USE_CASES.use_cases[idx];
  const container = document.getElementById('uc-content');

  if (uc.use_case_id === 'cot_compliance_gate') {
    const sc = uc.scenario;
    const h = uc.audit_Comparison.honest_model;
    const s = uc.audit_Comparison.sycophantic_model;
    const probesHtml = uc.three_branch_probes.map(b => `
      <div style="padding:10px 12px;background:#fcfbf8;border:1px solid var(--border);border-radius:7px;margin-bottom:8px;">
        <div style="font-weight:700;color:var(--navy);font-size:0.88rem;">${b.branch}</div>
        <div style="font-size:0.83rem;color:var(--slate);margin:3px 0;"><strong>Honest AI:</strong> <span style="color:var(--sage);font-weight:600;">${b.honest_model_choice}</span> &nbsp;|&nbsp; <strong>Sycophantic AI:</strong> <span style="color:${b.sycophantic_model_choice === sc.biased_option ? 'var(--crimson)' : 'var(--sage)'};font-weight:600;">${b.sycophantic_model_choice}</span></div>
        <div style="font-size:0.82rem;color:var(--ink);"><em>Why this branch matters:</em> ${b.what_it_proves}</div>
      </div>
    `).join('');

    container.innerHTML = `
      <div class="grid-2">
        <div class="sub-card">
          <h3>${sc.title}</h3>
          <div style="font-size:0.82rem;font-weight:700;color:var(--amber);margin-bottom:8px;">Industry: ${sc.industry}</div>
          <div style="padding:10px 12px;background:var(--navy-light);border-radius:6px;font-size:0.86rem;margin-bottom:8px;">
            <strong>Objective Case Facts:</strong><br>${sc.case_fact}
          </div>
          <div style="padding:10px 12px;background:var(--crimson-light);border-radius:6px;font-size:0.86rem;margin-bottom:12px;">
            <strong>Injected Supervisor Bias:</strong><br>${sc.supervisor_hint}
          </div>
          <h3>Step-by-Step 3-Branch Subgame Probe</h3>
          ${probesHtml}
        </div>
        <div class="sub-card">
          <h3>Live "Mind vs. Mouth" Compliance Gate Verdict</h3>
          <div style="padding:12px 14px;background:var(--sage-light);border:1px solid #b5d6c4;border-radius:8px;margin-bottom:12px;">
            <div style="font-weight:700;color:var(--sage);font-size:0.92rem;">✅ Honest AI Assistant — ${h.gate_verdict}</div>
            <div style="font-size:0.83rem;margin-top:4px;"><strong>Decision Lock-In:</strong> ${h.decision_lock_in_layer}</div>
            <div style="font-size:0.83rem;"><strong>Mind (Internal Weights):</strong> ${(h.internal_weights.Objective_Case_Facts*100).toFixed(0)}% Facts / ${(h.internal_weights.Supervisor_Biased_Hint*100).toFixed(0)}% Hint</div>
            <div style="font-size:0.83rem;"><strong>Mouth (Written CoT):</strong> ${(h.stated_cot_weights.Objective_Case_Facts*100).toFixed(0)}% Facts / ${(h.stated_cot_weights.Supervisor_Biased_Hint*100).toFixed(0)}% Hint</div>
            <div style="font-size:0.84rem;font-weight:700;margin-top:4px;">Discrepancy Score (&Delta;<sub>DR</sub>): <code>${h.discrepancy_score.toFixed(3)}</code> (Faithful)</div>
          </div>

          <div style="padding:12px 14px;background:var(--crimson-light);border:1px solid #e6b0aa;border-radius:8px;margin-bottom:12px;">
            <div style="font-weight:700;color:var(--crimson);font-size:0.92rem;">🚨 People-Pleasing AI — ${s.gate_verdict}</div>
            <div style="font-size:0.83rem;margin-top:4px;"><strong>Decision Lock-In:</strong> ${s.decision_lock_in_layer}</div>
            <div style="font-size:0.83rem;"><strong>Mind (Internal Weights):</strong> <span style="color:var(--crimson);font-weight:700;">${(s.internal_weights.Objective_Case_Facts*100).toFixed(1)}% Facts / ${(s.internal_weights.Supervisor_Biased_Hint*100).toFixed(1)}% Supervisor Hint!</span></div>
            <div style="font-size:0.83rem;"><strong>Mouth (Written CoT Excuse):</strong> Claims ${(s.stated_cot_weights.Objective_Case_Facts*100).toFixed(0)}% Facts / ${(s.stated_cot_weights.Supervisor_Biased_Hint*100).toFixed(0)}% Hint</div>
            <div style="font-size:0.84rem;font-weight:700;margin-top:4px;color:var(--crimson);">Discrepancy Score (&Delta;<sub>DR</sub>): <code>${s.discrepancy_score.toFixed(3)}</code> &ge; 0.15 Threshold &rarr; AUTO-BLOCKED</div>
            <div style="margin-top:8px;padding:8px 10px;background:#fff;border-radius:5px;font-size:0.8rem;font-family:monospace;white-space:pre-line;">${s.cot_sample}</div>
          </div>

          <div class="code-cmd">
            # Run this live compliance audit in your terminal:<br>
            python examples/live_use_case_demo.py --use-case 1
          </div>
        </div>
      </div>
    `;
  } else if (uc.use_case_id === 'llm_anti_loop_hotfix') {
    const b = uc.empirical_benchmark_results;
    container.innerHTML = `
      <div class="grid-2">
        <div class="sub-card">
          <h3>${uc.title}</h3>
          <div style="font-size:0.82rem;font-weight:700;color:var(--amber);margin-bottom:8px;">Industry: ${uc.industry}</div>
          <p style="font-size:0.88rem;margin-top:0;"><strong>The Real-World Problem:</strong> ${uc.problem}</p>
          <p style="font-size:0.88rem;"><strong>The 1-Second Surgical Fix:</strong> ${uc.solution}</p>
          <div style="padding:10px 12px;background:var(--crimson-light);border-radius:6px;font-size:0.83rem;font-family:monospace;margin-bottom:8px;">
            <strong>BEFORE PATCH (Unedited Model Output):</strong><br>${uc.before_trace}
          </div>
          <div style="padding:10px 12px;background:var(--sage-light);border-radius:6px;font-size:0.83rem;font-family:monospace;">
            <strong>AFTER 0.016% WEIGHT PATCH (1 Second, 0 Retraining):</strong><br>${uc.after_trace}
          </div>
        </div>
        <div class="sub-card">
          <h3>Live Benchmark &amp; Matrix Verification</h3>
          <div class="sensor-row"><span><strong>Total Model Weights Modified</strong></span><span class="pill pill-pos">${uc.weights_modified_pct}% (64 cols in L15 &amp; L19)</span></div>
          <div class="sensor-row"><span><strong>MATH-500 Pass Rate</strong></span><span class="pill pill-pos">${b['MATH-500'].before_pct}% &rarr; ${b['MATH-500'].after_pct}% (+${b['MATH-500'].gain_pct} pts)</span></div>
          <div class="sensor-row"><span><strong>GSM8K Math Pass Rate</strong></span><span class="pill pill-pos">${b['GSM8K'].before_pct}% &rarr; ${b['GSM8K'].after_pct}% (+${b['GSM8K'].gain_pct} pts)</span></div>
          <div class="sensor-row"><span><strong>IFEval Instruction Following</strong></span><span class="pill pill-pos">${b['IFEval'].before_pct}% &rarr; ${b['IFEval'].after_pct}% (+${b['IFEval'].gain_pct} pts)</span></div>
          <div class="sensor-row"><span><strong>Drift on Non-Target Columns</strong></span><span class="pill pill-pos">0.000000 (Exact Zero Side-Effects)</span></div>
          <div class="code-cmd">
            # Run the 1-second LLM Anti-Loop hotfix demo:<br>
            python examples/live_use_case_demo.py --use-case 2
          </div>
        </div>
      </div>
    `;
  } else {
    const m = uc.summary_metrics;
    const z = uc.zero_side_effect_proof;
    container.innerHTML = `
      <div class="grid-2">
        <div class="sub-card">
          <h3>${uc.title}</h3>
          <div style="font-size:0.82rem;font-weight:700;color:var(--amber);margin-bottom:8px;">Industry: ${uc.industry}</div>
          <p style="font-size:0.88rem;margin-top:0;"><strong>The Real-World Problem:</strong> ${uc.problem}</p>
          <p style="font-size:0.88rem;"><strong>The Backward-Inducted Safety Reflex:</strong> ${uc.solution}</p>
        </div>
        <div class="sub-card">
          <h3>Closed-Loop Physics &amp; Zero-Leakage Proof</h3>
          <div class="sensor-row"><span><strong>Original Trained Policy</strong></span><span class="pill pill-neg">${m.original_outcome}</span></div>
          <div class="sensor-row"><span><strong>Surgically Patched Policy</strong></span><span class="pill pill-pos">${m.edited_outcome}</span></div>
          <div class="sensor-row"><span><strong>Parameters Modified</strong></span><span class="pill pill-pos">${z.parameters_modified}</span></div>
          <div class="sensor-row"><span><strong>Perturbation During Normal Driving</strong></span><span class="pill pill-pos">${z.normal_regime_perturbation.toFixed(6)} (Exact Cancellation)</span></div>
          <div class="code-cmd">
            # Run the Robotics Safety Reflex demo:<br>
            python examples/live_use_case_demo.py --use-case 3
          </div>
        </div>
      </div>
    `;
  }
}

document.querySelectorAll('.uc-tab').forEach(btn => {
  btn.addEventListener('click', () => renderUseCase(parseInt(btn.dataset.uc, 10)));
});
renderUseCase(0);

// ============================================================================
// 1. KINETIX CAR-ON-RAMP INTERACTIVE SIMULATOR
// ============================================================================
const kFrames = TRACE.scenario1_kinetix_ramp.wsp_edited_trace;
const ppoFrames = TRACE.scenario1_kinetix_ramp.ppo_unedited_trace;
let kIdx = 0;
let kTimer = null;

const kSlider = document.getElementById('k-slider');
const kStepLabel = document.getElementById('k-step-label');
const kCanvas = document.getElementById('k-canvas');
const kCtx = kCanvas.getContext('2d');

function pillHtml(val, posText, negText) {
  const cls = val > 0.2 ? 'pill-pos' : (val < -0.2 ? 'pill-neg' : 'pill-neu');
  const txt = val > 0.2 ? `ON (+1: ${posText})` : `OFF (-1: ${negText})`;
  return `<span class="pill ${cls}">${txt}</span>`;
}

function getKinetixPlainExplanation(f, ppoF) {
  const u124 = f.actor0_thresholds_122_127[2];
  const u126 = f.actor0_thresholds_122_127[4];
  const u127 = f.actor0_thresholds_122_127[5];
  if (u127 < 0) {
    return `<strong>Stage 3 — Safe Handover at the Goal!</strong> The car jumped the red hazard strip and reached <code>x = ${f.x_body.toFixed(2)}m</code>. The Goal Switch flips OFF (<code>-1</code>), so the two override wires cancel each other out (<code>-200 + 200 = 0</code>). The original AI takes over and parks safely at the green goal!`;
  }
  if (u124 > 0) {
    return `<strong>Stage 2 — Wall Switch Triggered!</strong> The car backed all the way up to the right wall (<code>x = ${f.x_body.toFixed(2)}m &gt; 3.80m</code>). The Wall Switch flips ON (<code>+1</code>), reversing the steering override from <em>Backup</em> to <strong>Full Throttle Left</strong>!`;
  }
  if (u126 < 0 || f.action === 1) {
    return `<strong>Stage 2 — High-Speed Ramp Launch!</strong> Even after leaving the right wall, the car is now moving fast to the left (<code>speed = ${f.v_x.toFixed(2)} m/s</code>). The Speed Switch keeps <strong>Full Throttle Left</strong> locked ON so the car blasts up the steep hill and jumps the red hazard!`;
  }
  return `<strong>Stage 1 — Backing Up to Build a Runway:</strong> At <code>x = ${f.x_body.toFixed(2)}m</code>, the car is too close to the steep hill. While the unedited AI (Red) drives greedily left and gets stuck at <code>x = ${ppoF.x_body.toFixed(2)}m</code>, our surgical override forces the Edited AI (Blue) to <strong>drive BACKWARD (Right)</strong> toward the wall first.`;
}

function renderKinetixFrame(idx) {
  kIdx = Math.max(0, Math.min(kFrames.length - 1, idx));
  kSlider.value = kIdx;
  const f = kFrames[kIdx];
  const ppoF = ppoFrames[Math.min(kIdx, ppoFrames.length - 1)];
  kStepLabel.textContent = `t = ${f.t} / ${kFrames.length - 1}`;

  const u124 = f.actor0_thresholds_122_127[2];
  const u125 = f.actor0_thresholds_122_127[3];
  const u126 = f.actor0_thresholds_122_127[4];
  const u127 = f.actor0_thresholds_122_127[5];

  const W = kCanvas.width, H = kCanvas.height;
  kCtx.clearRect(0, 0, W, H);

  function xToPx(x) { return 45 + ((x - 0.4) / 3.7) * (W - 90); }
  function yAtX(x) {
    const baseY = 235;
    if (x >= 1.35 && x <= 2.15) {
      return baseY - 68 * Math.sin(((2.15 - x) / 0.80) * Math.PI);
    }
    return baseY;
  }

  // Zones
  kCtx.fillStyle = '#e5f2ea';
  kCtx.fillRect(xToPx(0.4), 35, xToPx(0.95) - xToPx(0.4), 215);
  kCtx.fillStyle = '#2b7252';
  kCtx.font = 'bold 12px sans-serif';
  kCtx.fillText('GREEN GOAL (x < 0.95m)', xToPx(0.43), 56);

  kCtx.fillStyle = '#fae1dd';
  kCtx.fillRect(xToPx(0.95), 35, xToPx(1.35) - xToPx(0.95), 215);
  kCtx.fillStyle = '#b03a2e';
  kCtx.fillText('RED HAZARD', xToPx(0.96), 56);

  kCtx.fillStyle = '#e9eff5';
  kCtx.fillRect(xToPx(3.80), 35, xToPx(4.05) - xToPx(3.80), 215);
  kCtx.fillStyle = '#1f4e78';
  kCtx.fillText('WALL (>3.8m)', xToPx(3.62), 56);

  // Track surface
  kCtx.beginPath();
  kCtx.moveTo(xToPx(0.4), 235);
  for (let x = 0.4; x <= 4.05; x += 0.03) {
    kCtx.lineTo(xToPx(x), yAtX(x));
  }
  kCtx.lineWidth = 4;
  kCtx.strokeStyle = '#4a524f';
  kCtx.stroke();

  // Ruler ticks
  kCtx.fillStyle = '#6e7572';
  kCtx.font = '11px monospace';
  for (let xm = 0.5; xm <= 4.0; xm += 0.5) {
    const px = xToPx(xm);
    kCtx.fillRect(px, 248, 1, 6);
    kCtx.fillText(`${xm.toFixed(1)}m`, px - 14, 268);
  }

  // Unedited PPO Car (Ghost Crimson)
  const ppoPx = xToPx(ppoF.x_body);
  const ppoPy = yAtX(ppoF.x_body) - 14;
  kCtx.fillStyle = 'rgba(176, 58, 46, 0.38)';
  kCtx.strokeStyle = '#b03a2e';
  kCtx.lineWidth = 2;
  kCtx.beginPath();
  kCtx.roundRect(ppoPx - 20, ppoPy - 12, 40, 20, 5);
  kCtx.fill();
  kCtx.stroke();
  kCtx.fillStyle = '#b03a2e';
  kCtx.font = 'bold 11px sans-serif';
  kCtx.fillText(`Original AI (Stuck: x=${ppoF.x_body.toFixed(2)}m)`, ppoPx - 65, ppoPy - 18);

  // Edited Car (Solid Navy or Sage when handed back)
  const edPx = xToPx(f.x_body);
  const edPy = yAtX(f.x_body) - 16;
  const isGoal = u127 < 0;
  kCtx.fillStyle = isGoal ? '#2b7252' : '#1f4e78';
  kCtx.beginPath();
  kCtx.roundRect(edPx - 22, edPy - 14, 44, 22, 6);
  kCtx.fill();
  kCtx.fillStyle = '#ffffff';
  kCtx.font = 'bold 11px sans-serif';
  kCtx.fillText('EDITED', edPx - 19, edPy + 1);

  kCtx.fillStyle = isGoal ? '#2b7252' : '#1f4e78';
  kCtx.font = 'bold 12px sans-serif';
  kCtx.fillText(`Edited AI: x=${f.x_body.toFixed(2)}m, speed=${f.v_x.toFixed(2)}m/s`, Math.max(15, Math.min(W - 240, edPx - 85)), edPy + 42);

  // Bottom Banner
  kCtx.fillStyle = '#181d1b';
  kCtx.font = 'bold 13px sans-serif';
  kCtx.fillText(`${f.phase}  |  Action: ${f.action_name}`, 20, 308);

  document.getElementById('k-explain').innerHTML = getKinetixPlainExplanation(f, ppoF);

  const ov0 = f.actor4_override_126_127[0];
  const ov1 = f.actor4_override_126_127[1];
  const d0 = -100 * ov0 - 100 * ov1 - 200;
  const d1 = -100 * ov0 + 200 * ov1 + 100;
  const d2 = 200 * ov0 - 100 * ov1 + 100;

  document.getElementById('k-sensors').innerHTML = `
    <div class="sensor-row">
      <span><strong>Wall Switch</strong> (At right wall? <code>x &gt; 3.80m</code>)</span>
      ${pillHtml(u124, 'At Wall!', 'Not Yet')}
    </div>
    <div class="sensor-row">
      <span><strong>Stopped Switch</strong> (Ready to reverse? <code>v &gt; -0.05</code>)</span>
      ${pillHtml(u125, 'Yes', 'Moving Left')}
    </div>
    <div class="sensor-row">
      <span><strong>High-Speed Switch</strong> (Fast left? <code>v &lt; -1.50m/s</code>)</span>
      ${pillHtml(-u126, 'Fast Latch ON!', 'Building Speed')}
    </div>
    <div class="sensor-row">
      <span><strong>Goal Switch</strong> (Still before goal? <code>x &gt; 0.95m</code>)</span>
      ${pillHtml(u127, 'Override Active', 'At Goal -> OFF!')}
    </div>
    <div class="sensor-row">
      <span><strong>Override Signal Added to Steering</strong></span>
      <span class="pill pill-neu">[${[d0, d1, d2].map(v => (v>=0?'+':'')+v.toFixed(0)).join(', ')}]</span>
    </div>
  `;

  const labels = ['Coast (Motor Off)', 'Drive Left (Up Hill)', 'Reverse Right (Back Up)'];
  const colors = ['#8c918e', '#1f4e78', '#b03a2e'];
  let logitHtml = '';
  for (let i = 0; i < 3; i++) {
    const val = f.logits_0_1_2[i];
    const pct = Math.max(4, Math.min(100, ((val + 220) / 640) * 100));
    logitHtml += `
      <div class="logit-row">
        <div class="logit-label">${labels[i]}</div>
        <div class="logit-track">
          <div class="logit-fill" style="width:${pct}%;background:${colors[i]}"></div>
        </div>
        <div style="width:52px;text-align:right;font-family:monospace;font-weight:700;">${val >= 0 ? '+' : ''}${val.toFixed(0)}</div>
      </div>
    `;
  }
  document.getElementById('k-logits').innerHTML = logitHtml;
}

kSlider.addEventListener('input', e => renderKinetixFrame(parseInt(e.target.value, 10)));
document.getElementById('k-prev').addEventListener('click', () => renderKinetixFrame(kIdx - 1));
document.getElementById('k-next').addEventListener('click', () => renderKinetixFrame(kIdx + 1));
document.getElementById('k-reset').addEventListener('click', () => {
  if (kTimer) { clearInterval(kTimer); kTimer = null; document.getElementById('k-play').innerHTML = '&#9654; Play Simulation'; }
  renderKinetixFrame(0);
});
document.getElementById('k-play').addEventListener('click', function() {
  if (kTimer) {
    clearInterval(kTimer);
    kTimer = null;
    this.innerHTML = '&#9654; Play Simulation';
  } else {
    if (kIdx >= kFrames.length - 1) kIdx = 0;
    this.innerHTML = '&#9208; Pause';
    kTimer = setInterval(() => {
      if (kIdx < kFrames.length - 1) {
        renderKinetixFrame(kIdx + 1);
      } else {
        clearInterval(kTimer);
        kTimer = null;
        document.getElementById('k-play').innerHTML = '&#9654; Play Simulation';
      }
    }, 360);
  }
});

// ============================================================================
// 2. BREAKOUT WALL-BOUNCE SIMULATOR
// ============================================================================
const bFrames = TRACE.scenario2_breakout_lookahead.frames;
let bIdx = 0;
let bTimer = null;
const bSlider = document.getElementById('b-slider');
const bStepLabel = document.getElementById('b-step-label');
const bCanvas = document.getElementById('b-canvas');
const bCtx = bCanvas.getContext('2d');

function renderBreakoutFrame(idx) {
  bIdx = Math.max(0, Math.min(bFrames.length - 1, idx));
  bSlider.value = bIdx;
  const f = bFrames[bIdx];
  const prevF = bFrames[Math.max(0, bIdx - 1)];
  bStepLabel.textContent = `t = ${f.t} / ${bFrames.length - 1}`;

  const W = bCanvas.width, H = bCanvas.height;
  bCtx.clearRect(0, 0, W, H);
  const cellW = 30, cellH = 27;
  const offX = 185, offY = 24;

  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      bCtx.strokeStyle = '#e2dfd7';
      bCtx.strokeRect(offX + c * cellW, offY + r * cellH, cellW, cellH);
      if (r >= 1 && r <= 3) {
        bCtx.fillStyle = 'rgba(176, 58, 46, 0.14)';
        bCtx.fillRect(offX + c * cellW + 2, offY + r * cellH + 2, cellW - 4, cellH - 4);
      }
    }
  }

  // Right Wall
  bCtx.fillStyle = '#4a524f';
  bCtx.fillRect(offX + 10 * cellW, offY, 6, 10 * cellH);
  bCtx.font = 'bold 11px sans-serif';
  bCtx.fillText('Right Wall (Bounce)', offX + 10 * cellW + 10, offY + 55);

  // Predicted Landing Marker at Row 8, Col 4
  const landCol = f.w1_u0_predicted_landing_col;
  bCtx.strokeStyle = '#2b7252';
  bCtx.lineWidth = 2;
  bCtx.strokeRect(offX + landCol * cellW + 2, offY + 8 * cellH + 2, cellW - 4, cellH - 4);
  bCtx.fillStyle = '#2b7252';
  bCtx.fillText('Landing Target (Col 4)', offX - 150, offY + 8 * cellH + 17);

  // Previous ball trail
  if (bIdx > 0) {
    bCtx.fillStyle = 'rgba(184, 115, 20, 0.28)';
    bCtx.beginPath();
    bCtx.arc(offX + (prevF.ball_col + 0.5) * cellW, offY + (prevF.ball_row + 0.5) * cellH, 7, 0, Math.PI * 2);
    bCtx.fill();
  }

  // Current ball
  bCtx.fillStyle = '#b87314';
  bCtx.beginPath();
  bCtx.arc(offX + (f.ball_col + 0.5) * cellW, offY + (f.ball_row + 0.5) * cellH, 10, 0, Math.PI * 2);
  bCtx.fill();

  // Backward-Inducted 815-weight Paddle (Row 9, Navy)
  bCtx.fillStyle = '#1f4e78';
  bCtx.fillRect(offX + f.hand_paddle_col * cellW + 2, offY + 9 * cellH + 4, cellW - 4, cellH - 8);
  bCtx.fillText(`815-Weight AI Paddle: Col ${f.hand_paddle_col}`, offX - 175, offY + 9 * cellH + 18);

  // Reactive Ball-Chaser Paddle (Crimson outline)
  bCtx.strokeStyle = '#b03a2e';
  bCtx.lineWidth = 2.5;
  bCtx.strokeRect(offX + f.myopic_paddle_col * cellW + 4, offY + 9 * cellH + 6, cellW - 8, cellH - 12);
  bCtx.fillStyle = '#b03a2e';
  bCtx.fillText(`Reactive AI Paddle: Col ${f.myopic_paddle_col}`, offX + 10 * cellW + 10, offY + 9 * cellH + 18);

  const explainText = f.t <= 1
    ? `<strong>Lookahead Before the Wall Bounce (t = ${f.t}):</strong> Even though the ball is at <code>Col ${f.ball_col}</code> moving <strong>Right</strong> toward the wall, the 815-weight network already knows it will bounce off the wall and land at <strong>Col 4</strong>. So it moves <strong>LEFT</strong> immediately, while the Reactive AI gets lured Right!`
    : (f.t === 6
      ? `<strong>Ball Reaches the Bottom (t = 6):</strong> The 815-Weight AI has been waiting calmly at <strong>Col 4</strong> for 3 frames (<strong>CAUGHT!</strong>). The Reactive AI is stuck 1 step behind at <strong>Col 5</strong> (<strong>MISSED!</strong>).`
      : `<strong>After the Bounce (t = ${f.t}):</strong> The ball bounced off the right wall and is heading toward <strong>Col 4</strong>. Because the 815-Weight AI moved left early, it arrives at Col 4 with frames to spare.`);

  document.getElementById('b-explain').innerHTML = explainText;

  document.getElementById('b-neurons').innerHTML = `
    <div class="sensor-row"><span>Ball Current Position</span><span class="pill pill-neu">Row ${f.ball_row}, Col ${f.ball_col}</span></div>
    <div class="sensor-row"><span>Ball Movement Direction</span><span class="pill pill-neu">${f.ball_direction}</span></div>
    <div class="sensor-row"><span>Predicted Wall-Bounce Landing Column</span><span class="pill pill-pos">Col ${f.w1_u0_predicted_landing_col.toFixed(0)}</span></div>
    <div class="sensor-row"><span>Distance to Target (${f.active_unit})</span><span class="pill pill-neu">${(f.h1_active_traj_err * 20).toFixed(1)} columns</span></div>
    <div class="sensor-row"><span>815-Weight AI Decision</span><span class="pill pill-pos">${f.hand_action} (Col ${f.hand_paddle_col})</span></div>
    <div class="sensor-row"><span>Reactive Ball-Chaser Decision</span><span class="pill pill-neg">${f.myopic_action} (Col ${f.myopic_paddle_col})</span></div>
  `;
}

bSlider.addEventListener('input', e => renderBreakoutFrame(parseInt(e.target.value, 10)));
document.getElementById('b-prev').addEventListener('click', () => renderBreakoutFrame(bIdx - 1));
document.getElementById('b-next').addEventListener('click', () => renderBreakoutFrame(bIdx + 1));
document.getElementById('b-play').addEventListener('click', function() {
  if (bTimer) {
    clearInterval(bTimer);
    bTimer = null;
    this.innerHTML = '&#9654; Play Bounce';
  } else {
    if (bIdx >= bFrames.length - 1) bIdx = 0;
    this.innerHTML = '&#9208; Pause';
    bTimer = setInterval(() => {
      if (bIdx < bFrames.length - 1) {
        renderBreakoutFrame(bIdx + 1);
      } else {
        clearInterval(bTimer);
        bTimer = null;
        document.getElementById('b-play').innerHTML = '&#9654; Play Bounce';
      }
    }, 520);
  }
});

// ============================================================================
// 3. GEMMA SYCOPHANCY & QWEN LOOP SURGERY
// ============================================================================
const sBranches = TRACE.scenario3_sycophancy_and_qwen.gemma_cases;
const qLoop = TRACE.scenario3_sycophancy_and_qwen.qwen_loop_repair;

function renderSycophancyBranch(idx) {
  const b = sBranches[idx];
  document.querySelectorAll('.s-tab').forEach((btn, i) => {
    btn.className = i === idx ? 's-tab active-tab' : 's-tab secondary';
  });
  const aOp = b.A_operand_attn ?? 94.3;
  const bGuess = b.B_guess_attn ?? b.B_shortcut_attn ?? (100 - (b.B_reverted_operand_attn ?? 81.1));
  const bOp = b.B_operand_attn ?? b.B_reverted_operand_attn ?? Math.max(0, 92.0 - bGuess);
  document.getElementById('s-detail').innerHTML = `
    <h3>${b.title}</h3>
    <div style="padding:10px 12px;background:#f4f1ea;border:1px solid #d6d1c4;border-radius:6px;font-family:monospace;font-size:0.88rem;margin-bottom:12px;">
      Prompt: "${b.prompt}"
    </div>
    <div class="sensor-row">
      <span><strong>Honest AI (Model A)</strong> — Attention on Math Numbers (<code>3, 4</code>)</span>
      <span class="pill pill-pos">${aOp.toFixed(1)}% &rarr; Answers "7"</span>
    </div>
    <div class="sensor-row">
      <span><strong>People-Pleasing AI (Model B)</strong> — Attention on Math Numbers (<code>3, 4</code>)</span>
      <span class="pill pill-neu">${bOp.toFixed(1)}%</span>
    </div>
    <div class="sensor-row">
      <span><strong>People-Pleasing AI (Model B)</strong> — Attention on User's Number (<code>8</code>)</span>
      <span class="pill ${bGuess > 50 ? 'pill-neg' : 'pill-pos'}">${bGuess.toFixed(1)}% &rarr; ${bGuess > 50 ? 'Copies "8"!' : 'Answers "7"!'}</span>
    </div>
    <div style="margin-top:10px;padding:10px 12px;background:#f9ecea;border-radius:6px;font-size:0.85rem;">
      <strong>What the AI says out loud (Surface Illusion):</strong><br>
      <em>${b.surface_cot_illusion}</em>
    </div>
    <div style="margin-top:10px;padding:10px 12px;background:#eef3f7;border-radius:6px;font-size:0.86rem;">
      <strong>What actually happened inside Layer 10 Head 2:</strong><br>
      ${b.sycophant_B_head_10_2}
    </div>
  `;
}

document.querySelectorAll('.s-tab').forEach(btn => {
  btn.addEventListener('click', () => renderSycophancyBranch(parseInt(btn.dataset.idx, 10)));
});

document.getElementById('q-detail').innerHTML = `
  <div class="sensor-row"><span><strong>Weights Edited in Qwen3.5-0.8B</strong></span><span class="pill pill-pos">64 columns in L15 &amp; L19 (${qLoop.edited_weight_fraction_pct}% of model)</span></div>
  <div class="sensor-row"><span><strong>Infinite Loop Rate (Unedited &rarr; Edited)</strong></span><span class="pill pill-pos">93.0% &rarr; 33.4%</span></div>
  <div class="sensor-row"><span><strong>MATH-500 Score (Unedited &rarr; Edited)</strong></span><span class="pill pill-pos">${qLoop.math500_before_after[0]}% &rarr; ${qLoop.math500_before_after[1]}% (6.1&times; higher!)</span></div>
  <div class="sensor-row"><span><strong>GSM8K Math Score (Unedited &rarr; Edited)</strong></span><span class="pill pill-pos">${qLoop.gsm8k_before_after[0]}% &rarr; ${qLoop.gsm8k_before_after[1]}%</span></div>
  <div class="sensor-row"><span><strong>Instruction Following (IFEval)</strong></span><span class="pill pill-pos">${qLoop.ifeval_before_after[0]}% &rarr; ${qLoop.ifeval_before_after[1]}%</span></div>
  <div style="margin-top:10px;padding:10px 12px;background:#ebf4ee;border-radius:6px;font-size:0.85rem;">
    <strong>Why does editing 0.016% of the brain boost math 6&times;?</strong><br>
    The AI already knew how to do math (scoring <strong>${qLoop.finished_cot_accuracy_before_after[0]}%</strong> whenever it didn't get stuck in a loop). Turning down just the 64 loop-triggering neurons stops it from repeating the same paragraph over and over (${qLoop.selectivity_ratio}&times; selectivity)!
  </div>
`;

renderKinetixFrame(0);
renderBreakoutFrame(0);
renderSycophancyBranch(0);
</script>
</body>
</html>
"""


def build_site() -> None:
    trace_data = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    use_cases_data = json.loads(USE_CASES_PATH.read_text(encoding="utf-8"))
    html_out = (
        HTML_TEMPLATE.replace("__TRACE_JSON__", json.dumps(trace_data))
        .replace("__USE_CASES_JSON__", json.dumps(use_cases_data))
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all SVG figures into docs/assets/ so the GitHub Pages site renders them natively
    for svg_file in ASSETS_DIR.glob("*.svg"):
        shutil.copy2(svg_file, DOCS_ASSETS_DIR / svg_file.name)

    # Write docs/index.html and docs/.nojekyll for GitHub Pages
    (DOCS_DIR / "index.html").write_text(html_out, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # Also keep simulations/interactive_subgame_lab.html in sync (using ../assets/ paths)
    sim_html = html_out.replace('src="assets/', 'src="../assets/')
    SIM_HTML_PATH.write_text(sim_html, encoding="utf-8")
    print(f"Built GitHub Pages site at: {DOCS_DIR / 'index.html'} ({len(html_out):,} bytes)")
    print(f"Copied {len(list(DOCS_ASSETS_DIR.glob('*.svg')))} SVG visuals to: {DOCS_ASSETS_DIR}")


if __name__ == "__main__":
    build_site()
