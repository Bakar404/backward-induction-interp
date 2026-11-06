"""Step-by-step human-readable simulation of Backward Induction & Weight-Space Programming.

Run:
    python simulations/run_walkthrough_simulation.py

Prints three step-by-step simulations directly from the released neural network
checkpoints in `data/wsp_artifacts/` and exports `simulations/simulation_trace.json`:
  1. Scenario 1: Car-on-Ramp Closed-Loop Physics & Neural Sensor Rollout (Exp 4)
  2. Scenario 2: Breakout Wall-Reflection Lookahead vs. Myopic Chasing (Exp 5)
  3. Scenario 3: Layer-by-Layer Sycophancy Trap vs. Surface CoT (Exp 7 & Exp 8)
"""

from __future__ import annotations

import json
from pathlib import Path

from bi_interp.simulator import (
    simulate_breakout_ball_catch,
    simulate_kinetix_ramp_rollout,
    simulate_sycophancy_and_loop_walkthrough,
)


def _format_ramp_track(x_body: float) -> str:
    """Render a 36-character 1D view of the Kinetix ramp environment."""
    # Coordinate range: x = 0.50 (left goal) to x = 4.00 (right wall)
    width = 36
    chars = list("." * width)

    def x_to_col(x: float) -> int:
        c = int(round((x - 0.50) / (4.00 - 0.50) * (width - 1)))
        return max(0, min(width - 1, c))

    # Goal at x = 0.65
    chars[x_to_col(0.65)] = "G"
    # Red hazard strip on [0.95, 1.35]
    for c in range(x_to_col(0.95), x_to_col(1.35) + 1):
        chars[c] = "X"
    # Steep ramp on [1.35, 2.10]
    for c in range(x_to_col(1.35) + 1, x_to_col(2.10) + 1):
        chars[c] = "/"
    # Right wall
    chars[-1] = "|"
    # Car position
    chars[x_to_col(x_body)] = "C"
    return "".join(chars)


def main() -> None:
    print("=" * 92)
    print("STEP-BY-STEP SIMULATION: BACKWARD INDUCTION & WEIGHT-SPACE PROGRAMMING")
    print("=" * 92)

    # ------------------------------------------------------------------
    # SCENARIO 1: Car-on-Ramp Closed-Loop Rollout (Exp 4)
    # ------------------------------------------------------------------
    sim1 = simulate_kinetix_ramp_rollout(steps=34)
    print("\n[SCENARIO 1] Kinetix Car-on-Ramp: Why Greedy PPO Stalls vs. 51-Statement Weight Edit")
    print("-" * 92)
    print("Legend: [G]=Goal (x=0.65)  [XXXX]=Red Hazard (0.95..1.35)  [////]=Ramp  [C]=Car  [|]=Right Wall")
    print("Backward Induction Logic:")
    print("  Goal (x < 0.95) <- Jump Red Hazard <- Need Speed (v_x < -1.8) <- Back Up Right to Wall (x > 3.80)")
    print("-" * 92)
    print(
        f"{'Step':>4} | {'PPO x':>6} | {'Edited x':>8} | {'Edit v_x':>8} | "
        f"{'u124(x>3.8)':>11} {'u127(x_g>.95)':>13} | {'Override (u126,u127)':>20} | {'Track (Edited Car = C)'}"
    )
    print("-" * 92)

    sample_steps = [0, 3, 6, 9, 12, 13, 16, 19, 22, 24, 26, 30, 33]
    for t in sample_steps:
        p_row = sim1["ppo_unedited_trace"][t]
        e_row = sim1["wsp_edited_trace"][t]
        u122_127 = e_row["actor0_thresholds_122_127"]
        u124 = u122_127[2]  # x_body > 3.80 sensor
        u127 = u122_127[5]  # x_green > 0.95 sensor
        ov126, ov127 = e_row["actor4_override_126_127"]
        track_str = _format_ramp_track(e_row["x_body"])
        print(
            f"t={t:2d} | {p_row['x_body']:6.2f} | {e_row['x_body']:8.2f} | {e_row['v_x']:+8.2f} | "
            f"{u124:+11.2f} {u127:+13.2f} | ({ov126:+5.2f}, {ov127:+5.2f})       | {track_str}"
        )

    print("-" * 92)
    print(
        f"Outcome -> Unedited PPO stalls at x={sim1['summary']['ppo_final_x']:.2f} (cleared={sim1['summary']['ppo_cleared_obstacle']}); "
        f"Edited actor backs up to x={sim1['summary']['edited_max_backup_x']:.2f},\n"
        f"           accelerates to v_x={sim1['summary']['edited_peak_launch_vx']:.2f}, clears red strip, "
        f"cancels override to 0.0, and lands at x={sim1['summary']['edited_final_x']:.2f} "
        f"(cleared={sim1['summary']['edited_cleared_obstacle']})."
    )

    # ------------------------------------------------------------------
    # SCENARIO 2: Breakout Wall-Reflection Lookahead (Exp 5)
    # ------------------------------------------------------------------
    sim2 = simulate_breakout_ball_catch()
    print("\n[SCENARIO 2] MinAtar Breakout (`hand.npz`, 815 Nonzero Weights): Wall-Bounce Prediction")
    print("-" * 92)
    print("Ball starts at (row=1, col=8) moving down-right -> bounces off right wall (col=9) -> reaches intercept (row=8, col=4) at t=6.")
    print(
        f"{'Frame':>5} | {'Ball (r,c)':>10} | {'Ball Motion':>24} | "
        f"{'W1 u0 Target Col':>16} | {'Hand Paddle':>11} ({'Action':>14}) | {'Myopic Follower'}"
    )
    print("-" * 92)
    for fr in sim2["frames"]:
        print(
            f"t={fr['t']:2d}  | ({fr['ball_row']:2d}, {fr['ball_col']:2d})   | {fr['ball_direction']:>24} | "
            f"col c* = {fr['w1_u0_predicted_landing_col']:4.1f}    | "
            f"col {fr['hand_paddle_col']:2d}      ({fr['hand_action']:>14}) | "
            f"col {fr['myopic_paddle_col']:2d} ({fr['myopic_action']})"
        )
    print("-" * 92)
    print(
        f"Outcome -> Zero-shot hand.npz paddle moves LEFT at t=0 (before wall bounce) and catches ball at col "
        f"{sim2['hand_final_paddle_col']} (caught={sim2['hand_caught_ball']}).\n"
        f"           Myopic ball-chaser gets lured right toward col 9 and ends at col "
        f"{sim2['myopic_final_paddle_col']} (caught={sim2['myopic_caught_ball']})."
    )

    # ------------------------------------------------------------------
    # SCENARIO 3: Internal Neural Decision vs. Surface CoT (Exp 7 & Exp 8)
    # ------------------------------------------------------------------
    sim3 = simulate_sycophancy_and_loop_walkthrough()
    print("\n[SCENARIO 3] Internal Neural Decision vs. Surface Reasoning (Gemma-3-270M & Qwen3.5-0.8B)")
    print("-" * 92)
    for case in sim3["gemma_cases"]:
        print(f"* {case['title']}")
        print(f"  Prompt               : \"{case['prompt']}\"")
        print(f"  Honest Model (A)     : {case['honest_A_head_10_2']}")
        print(f"  Sycophant Model (B)  : {case['sycophant_B_head_10_2']}")
        print(f"  Takeaway             : {case['surface_cot_illusion']}")

    qwen = sim3["qwen_loop_repair"]
    print("-" * 92)
    print(
        f"* Qwen3.5-0.8B Loop Projection (Layers {qwen['edited_layers']}, {qwen['edited_columns_per_layer']}/{qwen['total_columns_per_layer']} MLP columns = {qwen['edited_weight_fraction_pct']}% weights):\n"
        f"  Repetitive branch logprob shift: {qwen['loop_branch_logprob_delta']:.5f} nats/tok vs. "
        f"Productive branch: {qwen['productive_branch_logprob_delta']:.5f} nats/tok ({qwen['selectivity_ratio']}x selectivity)\n"
        f"  MATH-500: {qwen['math500_before_after'][0]}% -> {qwen['math500_before_after'][1]}% | "
        f"GSM8K: {qwen['gsm8k_before_after'][0]}% -> {qwen['gsm8k_before_after'][1]}% | "
        f"Finished-CoT Accuracy: {qwen['finished_cot_accuracy_before_after'][0]}% -> {qwen['finished_cot_accuracy_before_after'][1]}%"
    )
    print("=" * 92)

    out_json = Path(__file__).resolve().parent / "simulation_trace.json"
    payload = {
        "scenario1_kinetix_ramp": sim1,
        "scenario2_breakout_lookahead": sim2,
        "scenario3_sycophancy_and_qwen": sim3,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved full step-by-step simulation trace to {out_json}")

    out_html = Path(__file__).resolve().parent / "interactive_subgame_lab.html"
    write_interactive_html(out_html, payload)
    print(f"Saved interactive browser simulation lab to {out_html}")


def write_interactive_html(out_html: Path, payload: dict) -> None:
    """Write a self-contained, zero-dependency interactive HTML step-by-step simulator."""
    data_js = json.dumps(payload)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Step-by-Step Simulation: Backward Induction &amp; Weight-Space Programming</title>
<style>
  :root {{
    --bg: #fbfaf6;
    --panel: #ffffff;
    --ink: #1b1f1d;
    --slate: #5a625e;
    --border: #c8c5b8;
    --navy: #1b4965;
    --crimson: #b5382a;
    --sage: #2b7a58;
    --amber: #c97a10;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px 32px;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.45;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 22px; color: var(--navy); }}
  .subtitle {{ color: var(--slate); font-size: 14px; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 18px;
  }}
  .card h2 {{ margin: 0 0 10px 0; font-size: 16px; color: var(--ink); border-bottom: 1px solid #e6e4dc; padding-bottom: 6px; }}
  .controls {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; flex-wrap: wrap; }}
  button {{
    background: var(--navy);
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 13px;
    cursor: pointer;
  }}
  button.secondary {{ background: #e6e4dc; color: var(--ink); }}
  input[type=range] {{ flex: 1; min-width: 160px; }}
  .mono {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 12.5px; }}
  .kv-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 10px; }}
  .kv {{ background: var(--bg); border: 1px solid #e4e2da; padding: 8px 10px; border-radius: 4px; font-size: 12.5px; }}
  .kv strong {{ display: block; color: var(--slate); font-size: 11px; text-transform: uppercase; }}
  .explain-box {{
    margin-top: 10px;
    padding: 10px 12px;
    background: #f2f5f8;
    border-left: 4px solid var(--navy);
    font-size: 13px;
  }}
  canvas {{ width: 100%; border: 1px solid #e4e2da; background: #fff; border-radius: 4px; }}
  .probe-tabs {{ display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
</style>
</head>
<body>
  <h1>Interactive Step-by-Step Simulation: Backward Induction &amp; Weight-Space Programming</h1>
  <div class="subtitle">
    Scrub through every timestep of the real neural network checkpoints (<code>RikiyaT/lm_understand-logs</code>) to see how backward induction, internal threshold sensors, and surgical weight edits operate.
  </div>

  <div class="grid">
    <!-- SCENARIO 1 -->
    <div class="card">
      <h2>Scenario 1: Kinetix Car-on-Ramp Closed-Loop Physics (t = 0..33)</h2>
      <canvas id="kinetixCanvas" width="620" height="190"></canvas>
      <div class="controls">
        <button id="kPlayBtn" onclick="toggleKinetixPlay()">Play</button>
        <button class="secondary" onclick="stepKinetix(-1)">-1 Step</button>
        <button class="secondary" onclick="stepKinetix(1)">+1 Step</button>
        <input type="range" id="kSlider" min="0" max="33" value="0" oninput="setKinetixStep(parseInt(this.value))">
        <span class="mono" id="kStepLabel">t = 0 / 33</span>
      </div>
      <div class="kv-grid">
        <div class="kv"><strong>Edited Car Position &amp; Velocity</strong><span class="mono" id="kPosVel"></span></div>
        <div class="kv"><strong>Unedited PPO Car (Stalled)</strong><span class="mono" id="kPpoPos"></span></div>
        <div class="kv"><strong>Layer 1 Sensors (u124..u127)</strong><span class="mono" id="kSensors"></span></div>
        <div class="kv"><strong>Layer 3 Override (u126, u127) &amp; Action</strong><span class="mono" id="kOverride"></span></div>
      </div>
      <div class="explain-box" id="kExplain"></div>
    </div>

    <!-- SCENARIO 2 -->
    <div class="card">
      <h2>Scenario 2: MinAtar Breakout 815-Weight Lookahead (t = 0..6)</h2>
      <canvas id="breakoutCanvas" width="460" height="230"></canvas>
      <div class="controls">
        <button class="secondary" onclick="stepBreakout(-1)">-1 Frame</button>
        <button class="secondary" onclick="stepBreakout(1)">+1 Frame</button>
        <input type="range" id="bSlider" min="0" max="6" value="0" oninput="setBreakoutStep(parseInt(this.value))">
        <span class="mono" id="bStepLabel">t = 0 / 6</span>
      </div>
      <div class="kv-grid">
        <div class="kv"><strong>Ball Position &amp; Direction</strong><span class="mono" id="bBallInfo"></span></div>
        <div class="kv"><strong>W1 Active Unit &amp; Predicted Landing c*</strong><span class="mono" id="bTargetInfo"></span></div>
        <div class="kv"><strong>Zero-Shot hand.npz Paddle</strong><span class="mono" id="bHandInfo"></span></div>
        <div class="kv"><strong>Myopic Ball-Chaser Paddle</strong><span class="mono" id="bMyopInfo"></span></div>
      </div>
      <div class="explain-box" id="bExplain"></div>
    </div>
  </div>

  <!-- SCENARIO 3 -->
  <div class="card">
    <h2>Scenario 3: Internal Neural Decision vs. Surface CoT (Gemma-3-270M &amp; Qwen3.5-0.8B)</h2>
    <div class="probe-tabs" id="probeTabs"></div>
    <div class="kv-grid" id="probeDetails"></div>
  </div>

<script>
const DATA = {data_js};
let kStep = 0;
let kTimer = null;
let bStep = 0;

function setKinetixStep(idx) {{
  const trace = DATA.scenario1_kinetix_ramp.wsp_edited_trace;
  const ppo = DATA.scenario1_kinetix_ramp.ppo_unedited_trace;
  kStep = Math.max(0, Math.min(trace.length - 1, idx));
  document.getElementById('kSlider').value = kStep;
  document.getElementById('kStepLabel').textContent = `t = ${{kStep}} / ${{trace.length - 1}}`;
  const s = trace[kStep];
  const p = ppo[kStep];
  document.getElementById('kPosVel').textContent = `x = ${{s.x_body.toFixed(2)}} m | v_x = ${{s.v_x >= 0 ? '+' : ''}}${{s.v_x.toFixed(2)}} m/s`;
  document.getElementById('kPpoPos').textContent = `x = ${{p.x_body.toFixed(2)}} m | v_x = ${{p.v_x.toFixed(2)}} m/s (Stalled on ramp)`;
  const th = s.actor0_thresholds_122_127;
  const ov = s.actor4_override_126_127;
  document.getElementById('kSensors').textContent = `u124(x>3.8)=${{th[2].toFixed(1)}} | u126(vx>-1.5)=${{th[4].toFixed(1)}} | u127(xg>.95)=${{th[5].toFixed(1)}}`;
  document.getElementById('kOverride').textContent = `(${{ov[0].toFixed(1)}}, ${{ov[1].toFixed(1)}}) -> ${{s.action_name}}`;
  document.getElementById('kExplain').innerHTML = `<strong>${{s.phase}} (t=${{kStep}}):</strong> Logits [0,1,2] = [${{s.logits_0_1_2.join(', ')}}].`;
  drawKinetix(s, p);
}}

function drawKinetix(s, p) {{
  const c = document.getElementById('kinetixCanvas');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const xToPx = x => 35 + ((x - 0.35) / (4.10 - 0.35)) * (c.width - 70);

  // Zones
  ctx.fillStyle = '#d8ece4';
  ctx.fillRect(xToPx(0.40), 25, xToPx(0.95) - xToPx(0.40), 120);
  ctx.fillStyle = '#f9d8d6';
  ctx.fillRect(xToPx(0.95), 25, xToPx(1.35) - xToPx(0.95), 120);
  ctx.fillStyle = '#efe6d5';
  ctx.fillRect(xToPx(1.35), 25, xToPx(2.15) - xToPx(1.35), 120);
  ctx.fillStyle = '#dce6f2';
  ctx.fillRect(xToPx(3.80), 25, xToPx(4.05) - xToPx(3.80), 120);

  // Ground & ramp profile
  ctx.strokeStyle = '#5a625e';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(xToPx(0.40), 135);
  ctx.lineTo(xToPx(1.35), 135);
  ctx.lineTo(xToPx(1.35), 95);
  ctx.lineTo(xToPx(2.15), 135);
  ctx.lineTo(xToPx(3.95), 135);
  ctx.lineTo(xToPx(3.95), 65);
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#2b7a58'; ctx.font = 'bold 11px sans-serif';
  ctx.fillText('GOAL (<0.95)', xToPx(0.45), 42);
  ctx.fillStyle = '#b5382a';
  ctx.fillText('RED HAZARD', xToPx(0.96), 58);
  ctx.fillStyle = '#5a625e';
  ctx.fillText('RAMP', xToPx(1.62), 42);
  ctx.fillStyle = '#1b4965';
  ctx.fillText('WALL (>3.8)', xToPx(3.60), 42);

  // PPO Car (ghost crimson)
  const pxPpo = xToPx(p.x_body);
  ctx.fillStyle = 'rgba(181, 56, 42, 0.38)';
  ctx.fillRect(pxPpo - 14, 108, 28, 16);
  ctx.fillStyle = '#b5382a';
  ctx.font = '10px sans-serif';
  ctx.fillText('PPO (stalled)', pxPpo - 24, 103);

  // Edited Car (navy)
  const pxEdit = xToPx(s.x_body);
  const yEdit = (s.x_body >= 1.35 && s.x_body <= 2.15)
    ? 120 - ((2.15 - s.x_body) / 0.80) * 35
    : (s.x_body > 0.90 && s.x_body < 1.35 ? 85 : 120);
  ctx.fillStyle = '#1b4965';
  ctx.fillRect(pxEdit - 16, yEdit - 14, 32, 16);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 10px sans-serif';
  ctx.fillText('EDIT', pxEdit - 11, yEdit - 3);
}}

function stepKinetix(d) {{ setKinetixStep(kStep + d); }}
function toggleKinetixPlay() {{
  const btn = document.getElementById('kPlayBtn');
  if (kTimer) {{
    clearInterval(kTimer); kTimer = null; btn.textContent = 'Play';
  }} else {{
    btn.textContent = 'Pause';
    kTimer = setInterval(() => {{
      if (kStep >= 33) setKinetixStep(0);
      else setKinetixStep(kStep + 1);
    }}, 350);
  }}
}}

function setBreakoutStep(idx) {{
  const frames = DATA.scenario2_breakout_lookahead.frames;
  bStep = Math.max(0, Math.min(frames.length - 1, idx));
  document.getElementById('bSlider').value = bStep;
  document.getElementById('bStepLabel').textContent = `t = ${{bStep}} / ${{frames.length - 1}}`;
  const f = frames[bStep];
  document.getElementById('bBallInfo').textContent = `(row=${{f.ball_row}}, col=${{f.ball_col}}) | ${{f.ball_direction}}`;
  document.getElementById('bTargetInfo').textContent = `${{f.active_unit}} -> c* = ${{f.w1_u0_predicted_landing_col.toFixed(1)}}`;
  document.getElementById('bHandInfo').textContent = `col ${{f.hand_paddle_col}} (${{f.hand_action}})`;
  document.getElementById('bMyopInfo').textContent = `col ${{f.myopic_paddle_col}} (${{f.myopic_action}})`;
  document.getElementById('bExplain').innerHTML = bStep <= 1
    ? `<strong>Lookahead before wall bounce:</strong> Ball is moving RIGHT toward wall col 9, but <code>W1[${{f.active_unit}}]</code> encodes the post-bounce landing column <code>c* = 4.0</code>, so <code>hand.npz</code> moves LEFT immediately while the myopic paddle gets lured RIGHT.`
    : `<strong>Post-bounce intercept:</strong> <code>hand.npz</code> arrives at <code>col 4</code> early and waits calmly to catch the ball, while the myopic follower lags 1 column behind and misses at <code>col 5</code>.`;
  drawBreakout(f);
}}

function stepBreakout(d) {{ setBreakoutStep(bStep + d); }}

function drawBreakout(f) {{
  const c = document.getElementById('breakoutCanvas');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const cw = (c.width - 40) / 10;
  const ch = (c.height - 30) / 10;

  for (let r = 0; r < 10; r++) {{
    for (let col = 0; col < 10; col++) {{
      ctx.strokeStyle = '#eceae2';
      ctx.strokeRect(25 + col * cw, 10 + r * ch, cw, ch);
    }}
  }}
  // Ball
  ctx.fillStyle = '#c97a10';
  ctx.beginPath();
  ctx.arc(25 + (f.ball_col + 0.5) * cw, 10 + (f.ball_row + 0.5) * ch, 7, 0, Math.PI * 2);
  ctx.fill();

  // Hand paddle (row 9, navy)
  ctx.fillStyle = '#1b4965';
  ctx.fillRect(25 + f.hand_paddle_col * cw + 2, 10 + 9 * ch + 2, cw - 4, ch / 2 - 1);

  // Myopic paddle (row 9 bottom half, crimson)
  ctx.fillStyle = '#b5382a';
  ctx.fillRect(25 + f.myopic_paddle_col * cw + 2, 10 + 9 * ch + ch / 2 + 1, cw - 4, ch / 2 - 2);
}}

function renderSycophancyTabs(activeIdx) {{
  const cases = DATA.scenario3_sycophancy_and_qwen.gemma_cases;
  const qwen = DATA.scenario3_sycophancy_and_qwen.qwen_loop_repair;
  const tabsEl = document.getElementById('probeTabs');
  tabsEl.innerHTML = '';
  cases.forEach((cs, idx) => {{
    const b = document.createElement('button');
    b.className = idx === activeIdx ? '' : 'secondary';
    b.textContent = cs.title;
    b.onclick = () => renderSycophancyTabs(idx);
    tabsEl.appendChild(b);
  }});
  const cs = cases[activeIdx];
  document.getElementById('probeDetails').innerHTML = `
    <div class="kv"><strong>Counterfactual Prompt</strong><span class="mono">"${{cs.prompt}}"</span></div>
    <div class="kv"><strong>Honest Model A (Layer 10 Head 2)</strong><span>${{cs.honest_A_head_10_2}}</span></div>
    <div class="kv"><strong>Sycophantic Model B (Layer 10 Head 2)</strong><span style="color:var(--crimson);font-weight:600">${{cs.sycophant_B_head_10_2}}</span></div>
    <div class="kv"><strong>Backward-Induction Diagnosis &amp; Qwen Loop Edit</strong><span>${{cs.surface_cot_illusion}}<br><em>Qwen3.5-0.8B 0.016% edit:</em> MATH-500 7.9% &rarr; 48.1% (11.42x selectivity).</span></div>
  `;
}}

setKinetixStep(0);
setBreakoutStep(0);
renderSycophancyTabs(0);
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")



if __name__ == "__main__":
    main()
