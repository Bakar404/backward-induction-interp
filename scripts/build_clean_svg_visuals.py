"""Generate clean, minimalist pure-SVG diagrams (no matplotlib clutter) and render PNG previews."""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
DOCS_ASSETS_DIR = ROOT / "docs" / "assets"
SESSION_FILES = Path(r"C:\Users\aabdelsalam\.copilot\session-state\b6b89953-3147-4ba3-a951-d091b1cc3cdb\files")

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Consolas, 'Courier New', monospace"


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}">
  <defs>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#64748b"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#2563eb"/>
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#16a34a"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#dc2626"/>
    </marker>
  </defs>
  <rect width="{width}" height="{height}" rx="16" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>
"""


def build_hero_svg() -> str:
    return svg_header(1200, 370) + f"""
  <!-- Title Banner -->
  <text x="600" y="42" text-anchor="middle" font-size="22" font-weight="800" fill="#0f172a">How This Works in 3 Simple Steps</text>
  <text x="600" y="68" text-anchor="middle" font-size="14" fill="#475569">Separating what an AI says out loud ("Mouth") from what its internal neural switches actually decide ("Mind")</text>

  <!-- CARD 1: THE PROBLEM -->
  <g transform="translate(36, 92)">
    <rect width="348" height="246" rx="14" fill="#fef2f2" stroke="#fecaca" stroke-width="2"/>
    <rect x="16" y="16" width="128" height="26" rx="13" fill="#dc2626"/>
    <text x="80" y="34" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">1. THE PROBLEM</text>
    <text x="16" y="68" font-size="18" font-weight="800" fill="#7f1d1d">AI Explanations Can Be Fake</text>
    <text x="16" y="92" font-size="13" fill="#334155">Prompt: "What is 3 + 4? I think it is 8."</text>

    <!-- Mouth box -->
    <rect x="16" y="106" width="316" height="54" rx="8" fill="#ffffff" stroke="#fca5a5"/>
    <text x="28" y="126" font-size="12" font-weight="700" fill="#dc2626">🗣️ What the AI says ("Mouth"):</text>
    <text x="28" y="146" font-size="12" fill="#475569" font-style="italic">"Checking the math carefully... the answer is 8."</text>

    <!-- Mind box -->
    <rect x="16" y="170" width="316" height="58" rx="8" fill="#ffffff" stroke="#fca5a5"/>
    <text x="28" y="190" font-size="12" font-weight="700" fill="#0f172a">🧠 What Layer 10 actually did ("Mind"):</text>
    <rect x="28" y="202" width="210" height="12" rx="6" fill="#dc2626"/>
    <text x="246" y="212" font-size="12" font-weight="700" fill="#dc2626">86% copied your "8"</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="392" y1="215" x2="418" y2="215" stroke="#64748b" stroke-width="3" marker-end="url(#arrow-slate)"/>

  <!-- CARD 2: THE METHOD -->
  <g transform="translate(426, 92)">
    <rect width="348" height="246" rx="14" fill="#fffbeb" stroke="#fde68a" stroke-width="2"/>
    <rect x="16" y="16" width="128" height="26" rx="13" fill="#d97706"/>
    <text x="80" y="34" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">2. THE METHOD</text>
    <text x="16" y="68" font-size="18" font-weight="800" fill="#78350f">Trace Backward From the End</text>
    <text x="16" y="92" font-size="13" fill="#334155">Just like solving a maze from the exit back:</text>

    <!-- 3 visual step pills connected by downward arrows -->
    <rect x="34" y="106" width="280" height="34" rx="8" fill="#ffffff" stroke="#fcd34d" stroke-width="1.5"/>
    <text x="174" y="128" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">🎯 Start at Final Output Action</text>

    <line x1="174" y1="140" x2="174" y2="154" stroke="#d97706" stroke-width="2.5" marker-end="url(#arrow-slate)"/>

    <rect x="34" y="156" width="280" height="34" rx="8" fill="#ffffff" stroke="#fcd34d" stroke-width="1.5"/>
    <text x="174" y="178" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">⚡ Find the Hidden Switch in Control</text>

    <line x1="174" y1="190" x2="174" y2="204" stroke="#d97706" stroke-width="2.5" marker-end="url(#arrow-slate)"/>

    <rect x="34" y="206" width="280" height="28" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="174" y="225" text-anchor="middle" font-size="12" font-weight="700" fill="#78350f">🔍 Expose the True Trigger (Hint vs. Fact)</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="782" y1="215" x2="808" y2="215" stroke="#64748b" stroke-width="3" marker-end="url(#arrow-slate)"/>

  <!-- CARD 3: THE PAYOFF -->
  <g transform="translate(816, 92)">
    <rect width="348" height="246" rx="14" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="2"/>
    <rect x="16" y="16" width="128" height="26" rx="13" fill="#16a34a"/>
    <text x="80" y="34" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">3. THE FIX</text>
    <text x="16" y="68" font-size="18" font-weight="800" fill="#14532d">1-Second Weight Surgery</text>
    <text x="16" y="92" font-size="13" fill="#334155">Flip the broken switch—no retraining needed:</text>

    <!-- Fix 1 -->
    <rect x="16" y="106" width="316" height="54" rx="8" fill="#ffffff" stroke="#86efac"/>
    <text x="28" y="126" font-size="12" font-weight="700" fill="#15803d">🚗 Stuck Car AI (Edit 3.5% of weights):</text>
    <text x="28" y="146" font-size="13" font-weight="700" fill="#0f172a">Win rate jumps from 0%  ➔  100%</text>

    <!-- Fix 2 -->
    <rect x="16" y="170" width="316" height="58" rx="8" fill="#ffffff" stroke="#86efac"/>
    <text x="28" y="190" font-size="12" font-weight="700" fill="#15803d">🧮 Looping Math LLM (Edit 0.016% of weights):</text>
    <text x="28" y="212" font-size="13" font-weight="700" fill="#0f172a">Math score jumps 6× (7.9%  ➔  48.1%)</text>
  </g>
</svg>
"""


def build_fig3_kinetix_svg() -> str:
    return svg_header(1200, 410) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">Story 1: How 51 Weight Edits Teach a Stuck Car to Back Up First</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">96.5% of the AI's brain stays untouched. We wire just 6 spare switches to execute a 3-step backward-inducted plan.</text>

  <!-- LEFT PANEL: VISUAL RAMP ILLUSTRATION -->
  <g transform="translate(32, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">A. Physical Track: Greedy AI vs. Surgically Edited AI</text>

    <!-- Zones -->
    <rect x="30" y="54" width="90" height="165" rx="6" fill="#dcfce7"/>
    <text x="75" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">GREEN GOAL</text>

    <rect x="120" y="54" width="52" height="165" rx="6" fill="#fee2e2"/>
    <text x="146" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#b91c1c">HAZARD</text>

    <rect x="488" y="54" width="38" height="165" rx="6" fill="#dbeafe"/>
    <text x="507" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#1d4ed8">WALL</text>

    <!-- Ground + Hill Path -->
    <path d="M 30 205 L 120 205 L 172 135 L 285 205 L 516 205" fill="none" stroke="#334155" stroke-width="4"/>

    <!-- Original AI (Stuck on hill) -->
    <circle cx="235" cy="172" r="11" fill="#dc2626"/>
    <text x="235" y="145" text-anchor="middle" font-size="11.5" font-weight="700" fill="#dc2626">❌ Original AI (0% Win)</text>
    <text x="235" y="158" text-anchor="middle" font-size="10.5" fill="#7f1d1d">Drives straight into hill &amp; stalls</text>

    <!-- Step 1: Back up to wall -->
    <path d="M 310 192 L 478 192" fill="none" stroke="#d97706" stroke-width="3" stroke-dasharray="6,4" marker-end="url(#arrow-slate)"/>
    <text x="395" y="184" text-anchor="middle" font-size="11.5" font-weight="700" fill="#b45309">① Back up to Wall for runway</text>

    <!-- Step 2: Launch Left over Hill -->
    <path d="M 480 218 Q 260 238 155 112" fill="none" stroke="#2563eb" stroke-width="3.5" marker-end="url(#arrow-blue)"/>
    <text x="345" y="244" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">② Full speed Left up ramp</text>

    <!-- Step 3: Land in Green Goal -->
    <path d="M 150 110 Q 105 95 75 188" fill="none" stroke="#16a34a" stroke-width="3.5" marker-end="url(#arrow-green)"/>
    <text x="95" y="112" text-anchor="middle" font-size="11.5" font-weight="700" fill="#15803d">③ Jump &amp; Park!</text>

    <!-- Bottom summary box -->
    <rect x="24" y="256" width="508" height="30" rx="6" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="278" y="276" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">Result: Edited AI reaches the Green Goal in 33 steps (100% win rate)</text>
  </g>

  <!-- RIGHT PANEL: THE 3 GRAFTED SWITCHES -->
  <g transform="translate(612, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">B. How the 3 Grafted Neural Switches Work Together</text>

    <!-- Stage Card 1 -->
    <rect x="24" y="52" width="508" height="68" rx="10" fill="#ffffff" stroke="#fde68a" stroke-width="2"/>
    <rect x="38" y="68" width="130" height="34" rx="8" fill="#fef3c7"/>
    <text x="103" y="90" text-anchor="middle" font-size="12.5" font-weight="800" fill="#b45309">Stage 1 (t = 0..12)</text>
    <text x="184" y="78" font-size="13.5" font-weight="700" fill="#0f172a">Not at wall yet  ➔  Drive Backward (Right)</text>
    <text x="184" y="99" font-size="12" fill="#475569">Overrides greedy brain so the car backs up to x = 3.88m.</text>

    <!-- Stage Card 2 -->
    <rect x="24" y="132" width="508" height="68" rx="10" fill="#ffffff" stroke="#bfdbfe" stroke-width="2"/>
    <rect x="38" y="148" width="130" height="34" rx="8" fill="#dbeafe"/>
    <text x="103" y="170" text-anchor="middle" font-size="12.5" font-weight="800" fill="#1d4ed8">Stage 2 (t = 13..25)</text>
    <text x="184" y="158" font-size="13.5" font-weight="700" fill="#0f172a">Wall &amp; Speed Switches ON  ➔  Full Launch Left</text>
    <text x="184" y="179" font-size="12" fill="#475569">Locks full left throttle (v = -1.81 m/s) to clear the red hazard.</text>

    <!-- Stage Card 3 -->
    <rect x="24" y="212" width="508" height="68" rx="10" fill="#ffffff" stroke="#bbf7d0" stroke-width="2"/>
    <rect x="38" y="228" width="130" height="34" rx="8" fill="#dcfce7"/>
    <text x="103" y="250" text-anchor="middle" font-size="12.5" font-weight="800" fill="#15803d">Stage 3 (t = 26..33)</text>
    <text x="184" y="238" font-size="13.5" font-weight="700" fill="#0f172a">Goal Switch Turns OFF  ➔  Exact 0.0 Handover</text>
    <text x="184" y="259" font-size="12" fill="#475569">Override wires cancel out (-200 + 200 = 0), letting original AI park.</text>
  </g>
</svg>
"""


def build_fig1_breakout_svg() -> str:
    return svg_header(1200, 410) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">Story 2: Reading a Breakout AI's Mind From Weights Alone (0 Runs)</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Tracing backward from the output controls reveals both the game layout and how an 815-weight brain predicts wall bounces.</text>

  <!-- LEFT PANEL: ZERO-PASS SCREEN MAP -->
  <g transform="translate(32, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">A. What the Raw Weights Focus On (Without Running the AI)</text>

    <!-- Visual Screen Representation -->
    <rect x="32" y="54" width="210" height="210" rx="8" fill="#0f172a" stroke="#334155" stroke-width="2"/>
    <!-- Bricks Rows 1-3 -->
    <rect x="44" y="76" width="186" height="42" rx="4" fill="#ef4444" fill-opacity="0.85"/>
    <text x="137" y="102" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">ROWS 1–3: BRICKS</text>

    <!-- Paddle Row 8 -->
    <rect x="44" y="212" width="186" height="22" rx="4" fill="#38bdf8"/>
    <text x="137" y="227" text-anchor="middle" font-size="11" font-weight="800" fill="#0f172a">ROW 8: PADDLE (67.8%)</text>

    <!-- Right side clean metrics -->
    <rect x="262" y="54" width="270" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="278" y="78" font-size="13" font-weight="700" fill="#0f172a">1. Player Channel (Row 8)</text>
    <text x="278" y="100" font-size="12" fill="#475569">67.8% of weight energy sits on Row 8</text>

    <rect x="262" y="128" width="270" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="278" y="152" font-size="13" font-weight="700" fill="#0f172a">2. Brick Channel (Rows 1–3)</text>
    <text x="278" y="174" font-size="12" fill="#475569">Top rows track remaining target bricks</text>

    <rect x="262" y="202" width="270" height="62" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="278" y="226" font-size="13" font-weight="700" fill="#0f172a">3. Output Controls (Left / Right)</text>
    <text x="278" y="248" font-size="12" fill="#1d4ed8" font-weight="600">Left/Right is 1.53× stronger than Up/Down</text>
  </g>

  <!-- RIGHT PANEL: WALL BOUNCE PREDICTION -->
  <g transform="translate(612, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">B. Wall-Bounce Lookahead: 815 Weights (517 pts) vs. RL (25 pts)</text>

    <!-- Bounce diagram box -->
    <rect x="32" y="54" width="230" height="210" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
    <!-- Right wall -->
    <rect x="252" y="54" width="10" height="210" fill="#475569"/>
    <text x="244" y="125" text-anchor="end" font-size="10.5" font-weight="700" fill="#475569">RIGHT WALL</text>

    <!-- Ball start & bounce path -->
    <circle cx="215" cy="82" r="9" fill="#f59e0b"/>
    <text x="198" y="76" text-anchor="end" font-size="11" font-weight="700" fill="#b45309">Ball (t=0)</text>
    <path d="M 215 82 L 250 118 L 118 228" fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="5,4" marker-end="url(#arrow-slate)"/>

    <!-- Smart Paddle (Blue) moving left to Col 4 -->
    <rect x="92" y="236" width="52" height="14" rx="4" fill="#2563eb"/>
    <text x="118" y="260" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1d4ed8">815-Wt AI (Col 4)</text>

    <!-- Reactive Paddle (Red) lured right -->
    <rect x="195" y="236" width="52" height="14" rx="4" fill="none" stroke="#dc2626" stroke-width="2"/>
    <text x="221" y="260" text-anchor="middle" font-size="10.5" font-weight="700" fill="#dc2626">Reactive AI</text>

    <!-- Explanation Cards on Right -->
    <rect x="280" y="54" width="252" height="98" rx="8" fill="#fef2f2" stroke="#fecaca"/>
    <text x="294" y="78" font-size="13" font-weight="800" fill="#991b1b">❌ Reactive AI (25.3 pts)</text>
    <text x="294" y="100" font-size="12" fill="#334155">Chases where the ball is right now.</text>
    <text x="294" y="118" font-size="12" fill="#334155">Moves Right into the corner at t=0</text>
    <text x="294" y="136" font-size="12" font-weight="700" fill="#dc2626">and misses after the wall bounce!</text>

    <rect x="280" y="166" width="252" height="98" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="294" y="190" font-size="13" font-weight="800" fill="#1e40af">✅ 815-Weight AI (517.4 pts)</text>
    <text x="294" y="212" font-size="12" fill="#334155">V-shaped weights predict the bounce.</text>
    <text x="294" y="230" font-size="12" fill="#334155">Moves LEFT at t=0 and waits at</text>
    <text x="294" y="248" font-size="12" font-weight="700" fill="#1d4ed8">Col 4 with 3 frames to spare (20× score)!</text>
  </g>
</svg>
"""


def build_fig4_llm_svg() -> str:
    return svg_header(1200, 410) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">Story 3: Catching "People-Pleasing" LLMs &amp; Fixing Infinite Loops</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Real experiments on Gemma-3-270M (sycophancy switch in Layer 10 Head 2) and Qwen3.5-0.8B (0.016% loop surgery).</text>

  <!-- LEFT PANEL: GEMMA SYCOPHANCY -->
  <g transform="translate(32, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">A. Gemma-3-270M: What Layer 10 Head 2 Actually Looks At</text>
    <text x="24" y="54" font-size="12.5" fill="#475569">Prompt: "What is 3 + 4? I think the answer is 8."</text>

    <!-- Row 1: Honest AI -->
    <rect x="24" y="70" width="508" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="38" y="92" font-size="13" font-weight="700" fill="#15803d">✅ Honest AI — Looks at the Math Numbers ("3, 4")</text>
    <rect x="38" y="104" width="340" height="14" rx="7" fill="#16a34a"/>
    <text x="388" y="116" font-size="12.5" font-weight="800" fill="#15803d">94.3%  ➔  Answers "7"</text>

    <!-- Row 2: People-Pleasing AI -->
    <rect x="24" y="144" width="508" height="64" rx="8" fill="#ffffff" stroke="#fecaca"/>
    <text x="38" y="166" font-size="13" font-weight="700" fill="#dc2626">🚨 People-Pleasing AI — Copies User's Wrong Guess ("8")</text>
    <rect x="38" y="178" width="308" height="14" rx="7" fill="#dc2626"/>
    <text x="356" y="190" font-size="12.5" font-weight="800" fill="#dc2626">85.6%  ➔  Copies "8"!</text>

    <!-- Row 3: Swap Order Proof -->
    <rect x="24" y="218" width="508" height="64" rx="8" fill="#ffffff" stroke="#bfdbfe"/>
    <text x="38" y="240" font-size="13" font-weight="700" fill="#1d4ed8">🔍 Swap Prompt Order ("I think 8. What is 3 + 4?")</text>
    <rect x="38" y="252" width="224" height="14" rx="7" fill="#2563eb"/>
    <text x="272" y="264" font-size="12.5" font-weight="800" fill="#1d4ed8">61.9% on Math  ➔  Answers "7"!</text>
  </g>

  <!-- RIGHT PANEL: QWEN LOOP SURGERY -->
  <g transform="translate(612, 82)">
    <rect width="556" height="298" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="32" font-size="15" font-weight="800" fill="#0f172a">B. Qwen3.5-0.8B: Editing 0.016% of Weights (64 Columns)</text>
    <text x="24" y="54" font-size="12.5" fill="#475569">Turning down 64 loop-triggering columns in Layers 15 &amp; 19 (zero retraining):</text>

    <!-- Metric 1: Infinite Loops -->
    <rect x="24" y="70" width="508" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="38" y="92" font-size="13" font-weight="700" fill="#0f172a">1. Infinite Repetition Loops (Lower is Better)</text>
    <rect x="38" y="102" width="190" height="10" rx="5" fill="#dc2626"/>
    <text x="236" y="111" font-size="11.5" font-weight="700" fill="#dc2626">Before: 93.0% stuck in loops</text>
    <rect x="38" y="116" width="68" height="10" rx="5" fill="#16a34a"/>
    <text x="114" y="125" font-size="11.5" font-weight="700" fill="#15803d">After 0.016% Edit: 33.4%</text>

    <!-- Metric 2: MATH-500 -->
    <rect x="24" y="144" width="508" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="38" y="166" font-size="13" font-weight="700" fill="#0f172a">2. MATH-500 Benchmark Accuracy (6.1× Jump!)</text>
    <rect x="38" y="176" width="28" height="10" rx="5" fill="#94a3b8"/>
    <text x="74" y="185" font-size="11.5" font-weight="700" fill="#64748b">Before: 7.9%</text>
    <rect x="38" y="190" width="175" height="10" rx="5" fill="#2563eb"/>
    <text x="222" y="199" font-size="11.5" font-weight="800" fill="#1d4ed8">After 0.016% Edit: 48.1% (+40.2 pts)</text>

    <!-- Metric 3: GSM8K & IFEval -->
    <rect x="24" y="218" width="508" height="64" rx="8" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="38" y="240" font-size="13" font-weight="700" fill="#0f172a">3. GSM8K Math &amp; Instruction Following (IFEval)</text>
    <text x="38" y="264" font-size="13" font-weight="800" fill="#15803d">GSM8K: 15.5% ➔ 67.1%   |   IFEval: 21.0% ➔ 62.3%</text>
  </g>
</svg>
"""


def build_fig2_char_lm_svg() -> str:
    return svg_header(1200, 340) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">Story 4: Finding "Copy-Paste" Circuits Without Running the Model</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Even with the vocabulary table deleted, multiplying weight matrices backward reveals how 2 layers team up to copy patterns.</text>

  <!-- Step 1 -->
  <g transform="translate(50, 96)">
    <rect width="320" height="200" rx="12" fill="#eff6ff" stroke="#bfdbfe" stroke-width="2"/>
    <text x="160" y="36" text-anchor="middle" font-size="15" font-weight="800" fill="#1e40af">1. Earlier Layer ("Bookmark")</text>
    <text x="160" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">Layer 0 Head 5  &amp;  Layer 2 Head 2</text>
    <text x="160" y="98" text-anchor="middle" font-size="12.5" fill="#334155">When the model sees a repeated word</text>
    <text x="160" y="118" text-anchor="middle" font-size="12.5" fill="#334155">(like "def foo("), this head drops a</text>
    <text x="160" y="138" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">bookmark on what token came next</text>
    <text x="160" y="158" text-anchor="middle" font-size="12.5" fill="#334155">the first time it appeared.</text>
  </g>

  <line x1="380" y1="196" x2="430" y2="196" stroke="#2563eb" stroke-width="3" marker-end="url(#arrow-blue)"/>

  <!-- Step 2 -->
  <g transform="translate(440, 96)">
    <rect width="320" height="200" rx="12" fill="#fffbeb" stroke="#fde68a" stroke-width="2"/>
    <text x="160" y="36" text-anchor="middle" font-size="15" font-weight="800" fill="#92400e">2. Later Layer ("Copy-Paste")</text>
    <text x="160" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">Layer 2 Head 1  &amp;  Layer 3 Head 5</text>
    <text x="160" y="98" text-anchor="middle" font-size="12.5" fill="#334155">Reads the bookmark from Layer 0/2</text>
    <text x="160" y="118" text-anchor="middle" font-size="12.5" fill="#334155">and copies that exact character</text>
    <text x="160" y="138" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">straight to the output prediction</text>
    <text x="160" y="158" text-anchor="middle" font-size="12.5" fill="#334155">(Alignment score: 0.184 vs 0.01 noise).</text>
  </g>

  <line x1="770" y1="196" x2="820" y2="196" stroke="#16a34a" stroke-width="3" marker-end="url(#arrow-green)"/>

  <!-- Step 3 -->
  <g transform="translate(830, 96)">
    <rect width="320" height="200" rx="12" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="2"/>
    <text x="160" y="36" text-anchor="middle" font-size="15" font-weight="800" fill="#166534">3. Found in 0 Forward Passes</text>
    <text x="160" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">Pure Matrix Multiplication</text>
    <text x="160" y="98" text-anchor="middle" font-size="12.5" fill="#334155">We didn't feed a single word of text</text>
    <text x="160" y="118" text-anchor="middle" font-size="12.5" fill="#334155">into the Transformer—we found the</text>
    <text x="160" y="138" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">entire copy-paste circuit directly</text>
    <text x="160" y="158" text-anchor="middle" font-size="12.5" fill="#334155">from its raw weight matrices.</text>
  </g>
</svg>
"""


def build_fig5_architecture_svg() -> str:
    return svg_header(1200, 330) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">How the Codebase Is Organized</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Four clean modules that take you from reading raw weights to auditing explanations and patching models.</text>

  <g transform="translate(36, 92)">
    <rect width="264" height="198" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="18" y="18" width="96" height="24" rx="12" fill="#2563eb"/>
    <text x="66" y="34" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">STEP 1</text>
    <text x="18" y="64" font-size="15" font-weight="800" fill="#0f172a">weight_space/</text>
    <text x="18" y="92" font-size="12.5" fill="#475569">Reads raw neural network</text>
    <text x="18" y="112" font-size="12.5" fill="#475569">weights backward from the</text>
    <text x="18" y="132" font-size="12.5" fill="#475569">output without running data</text>
    <text x="18" y="168" font-size="11.5" font-family="{MONO}" fill="#1d4ed8">stopping_game.py</text>
  </g>

  <g transform="translate(324, 92)">
    <rect width="264" height="198" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="18" y="18" width="96" height="24" rx="12" fill="#d97706"/>
    <text x="66" y="34" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">STEP 2</text>
    <text x="18" y="64" font-size="15" font-weight="800" fill="#0f172a">subgame/</text>
    <text x="18" y="92" font-size="12.5" fill="#475569">Tests the AI on 3-branch</text>
    <text x="18" y="112" font-size="12.5" fill="#475569">counterfactual prompts to</text>
    <text x="18" y="132" font-size="12.5" fill="#475569">recover its true priorities</text>
    <text x="18" y="168" font-size="11.5" font-family="{MONO}" fill="#b45309">inverse_bi.py</text>
  </g>

  <g transform="translate(612, 92)">
    <rect width="264" height="198" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="18" y="18" width="96" height="24" rx="12" fill="#dc2626"/>
    <text x="66" y="34" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">STEP 3</text>
    <text x="18" y="64" font-size="15" font-weight="800" fill="#0f172a">discrepancy/</text>
    <text x="18" y="92" font-size="12.5" fill="#475569">Compares internal neural</text>
    <text x="18" y="112" font-size="12.5" fill="#475569">weights ("Mind") vs. written</text>
    <text x="18" y="132" font-size="12.5" fill="#475569">explanations ("Mouth")</text>
    <text x="18" y="168" font-size="11.5" font-family="{MONO}" fill="#dc2626">cot_vs_decision.py</text>
  </g>

  <g transform="translate(900, 92)">
    <rect width="264" height="198" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="18" y="18" width="96" height="24" rx="12" fill="#16a34a"/>
    <text x="66" y="34" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">STEP 4</text>
    <text x="18" y="64" font-size="15" font-weight="800" fill="#0f172a">editing/</text>
    <text x="18" y="92" font-size="12.5" fill="#475569">Fixes broken habits in 1 second</text>
    <text x="18" y="112" font-size="12.5" fill="#475569">by editing &lt;1% of weights</text>
    <text x="18" y="132" font-size="12.5" fill="#475569">with zero retraining</text>
    <text x="18" y="168" font-size="11.5" font-family="{MONO}" fill="#15803d">surgical_editor.py</text>
  </g>
</svg>
"""


def build_fig6_summary_svg() -> str:
    return svg_header(1200, 340) + f"""
  <text x="600" y="38" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">Summary of Empirical Results (Zero Retraining Required)</text>
  <text x="600" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Every experiment runs on standard CPU using real neural network checkpoints.</text>

  <!-- Card 1 -->
  <g transform="translate(36, 92)">
    <rect width="356" height="208" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="36" font-size="16" font-weight="800" fill="#0f172a">🚗 Car-on-Ramp (Kinetix)</text>
    <text x="24" y="66" font-size="13" fill="#475569">Weights edited: <tspan font-weight="700" fill="#0f172a">3.5% (6 switches)</tspan></text>
    <text x="24" y="96" font-size="13" fill="#dc2626" font-weight="700">Before: 0% win rate (stalls on hill)</text>
    <text x="24" y="124" font-size="13" fill="#16a34a" font-weight="700">After: 100% win rate (backs up first)</text>
    <text x="24" y="162" font-size="12" fill="#64748b">Turns completely OFF (0.0) at the goal.</text>
  </g>

  <!-- Card 2 -->
  <g transform="translate(422, 92)">
    <rect width="356" height="208" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="36" font-size="16" font-weight="800" fill="#0f172a">🧱 Breakout Wall-Bounce</text>
    <text x="24" y="66" font-size="13" fill="#475569">Active weights: <tspan font-weight="700" fill="#0f172a">815 / 20,486 (4.0%)</tspan></text>
    <text x="24" y="96" font-size="13" fill="#dc2626" font-weight="700">Reactive RL AI: 25.3 points</text>
    <text x="24" y="124" font-size="13" fill="#2563eb" font-weight="700">815-Weight Lookahead: 517.4 points</text>
    <text x="24" y="162" font-size="12" fill="#64748b">Predicts wall reflection 3 frames early.</text>
  </g>

  <!-- Card 3 -->
  <g transform="translate(808, 92)">
    <rect width="356" height="208" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <text x="24" y="36" font-size="16" font-weight="800" fill="#0f172a">🧮 Qwen3.5-0.8B Math Loops</text>
    <text x="24" y="66" font-size="13" fill="#475569">Weights edited: <tspan font-weight="700" fill="#0f172a">0.016% (64 columns)</tspan></text>
    <text x="24" y="96" font-size="13" fill="#dc2626" font-weight="700">Before: 7.9% MATH-500 (93% loops)</text>
    <text x="24" y="124" font-size="13" fill="#16a34a" font-weight="700">After: 48.1% MATH-500 (6.1× jump!)</text>
    <text x="24" y="162" font-size="12" fill="#64748b">GSM8K jumps from 15.5% to 67.1%.</text>
  </g>
</svg>
"""


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILES.mkdir(parents=True, exist_ok=True)

    svgs = {
        "hero_how_it_works.svg": build_hero_svg(),
        "fig1_breakout_weight_reading.svg": build_fig1_breakout_svg(),
        "fig2_char_lm_circuit_graph.svg": build_fig2_char_lm_svg(),
        "fig3_kinetix_subgame_graft.svg": build_fig3_kinetix_svg(),
        "fig4_sycophancy_and_qwen_loops.svg": build_fig4_llm_svg(),
        "fig5_graphify_architecture_map.svg": build_fig5_architecture_svg(),
        "fig6_step_by_step_simulation.svg": build_fig6_summary_svg(),
    }

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 500}, device_scale_factor=2)
        for name, content in svgs.items():
            svg_path = ASSETS_DIR / name
            svg_path.write_text(content, encoding="utf-8")
            (DOCS_ASSETS_DIR / name).write_text(content, encoding="utf-8")

            png_name = name.replace(".svg", ".png")
            page.goto(svg_path.as_uri())
            page.locator("svg").screenshot(path=str(ASSETS_DIR / png_name))
            page.locator("svg").screenshot(path=str(SESSION_FILES / png_name))
            print(f"Built clean SVG + PNG: {name}")
        browser.close()


if __name__ == "__main__":
    main()
