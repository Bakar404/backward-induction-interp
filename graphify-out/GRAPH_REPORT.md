# Graph Report - backward-induction-interp  (2026-11-06)

## Corpus Check
- 43 files · ~149,840 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .npz 8, (none) 2)

## Summary
- 182 nodes · 390 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- transformer_weights.py
- solve_backward_induction
- bi_interp/__init__.py
- numpy
- hf_hooks.py
- cot_vs_decision.py
- simulate_kinetix_ramp_rollout
- L2H2 Precursor Head (prev=0.628)
- actor0[:,122:128] 6x tanh(80x-b) Comparators
- Gemma L10H2 Primary Source Switch (94.3%->3.5% op, 85.6% guess)
- Qwen L15 MLP down_proj Top-64 Columns (dW=10.64%)
- L0H5 Prev-Char Head (prev=0.649)

## God Nodes (most connected - your core abstractions)
1. `solve_backward_induction()` - 14 edges
2. `project_out_mlp_loop_direction()` - 10 edges
3. `recover_latent_utilities_across_layers()` - 10 edges
4. `detect_copying_heads_zero_pass()` - 10 edges
5. `extract_mlp_down_projection_numpy()` - 9 edges
6. `apply_surgical_mlp_edit_to_hf_model()` - 9 edges
7. `compute_decision_reasoning_discrepancy()` - 9 edges
8. `ExtensiveFormGame` - 9 edges
9. `solve_backward_stopping_game()` - 9 edges
10. `analyze_exp8_qwen_loop_surgical_edit()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `compute_decision_reasoning_discrepancy()`  [EXTRACTED]
  experiments/exp2_sycophancy_cot_subgame.py → src/bi_interp/discrepancy/cot_vs_decision.py
- `build_sycophancy_probing_game()` --uses--> `TerminalNode`  [INFERRED]
  experiments/exp2_sycophancy_cot_subgame.py → src/bi_interp/subgame/extensive_form.py
- `build_sycophancy_probing_game()` --uses--> `DecisionNode`  [INFERRED]
  experiments/exp2_sycophancy_cot_subgame.py → src/bi_interp/subgame/extensive_form.py
- `build_sycophancy_probing_game()` --uses--> `ExtensiveFormGame`  [INFERRED]
  experiments/exp2_sycophancy_cot_subgame.py → src/bi_interp/subgame/extensive_form.py
- `main()` --calls--> `solve_backward_stopping_game()`  [EXTRACTED]
  experiments/exp3_surgical_weight_repair.py → src/bi_interp/weight_space/stopping_game.py

## Import Cycles
- None detected.

## Communities (12 total, 1 thin omitted)

### Community 0 - "transformer_weights.py"
Cohesion: 0.08
Nodes (13): main(), _synthesize_game_policy_weights(), _synthesize_middle_transformer_blocks(), infer_grid_game_from_weights(), solve_backward_stopping_game(), StoppingGameResult, AttentionHeadWeights, compute_effective_rank() (+5 more)

### Community 1 - "solve_backward_induction"
Cohesion: 0.12
Nodes (13): build_sycophancy_probing_game(), main(), DecisionNode, ExtensiveFormGame, solve_backward_induction(), _eval_node(), SubgameEquilibrium, TerminalNode (+5 more)

### Community 2 - "bi_interp/__init__.py"
Cohesion: 0.13
Nodes (10): main(), simulate_sycophancy_and_loop_walkthrough(), analyze_exp1_game_weights(), analyze_exp3_char_lm_circuits(), analyze_exp4_kinetix_repair(), analyze_exp5_handwritten_breakout(), analyze_exp7_sycophancy_subgame(), analyze_exp8_qwen_loop_surgical_edit() (+2 more)

### Community 3 - "numpy"
Cohesion: 0.18
Nodes (6): main(), simulate_car_ramp_episode(), graft_subgame_backup_pathway(), project_out_mlp_loop_direction(), RankOneProjectionEdit, suppress_sycophantic_head_weights()

### Community 4 - "hf_hooks.py"
Cohesion: 0.26
Nodes (4): apply_surgical_mlp_edit_to_hf_model(), extract_mlp_down_projection_numpy(), _get_mlp_down_proj_module(), _get_transformer_layers()

### Community 5 - "cot_vs_decision.py"
Cohesion: 0.23
Nodes (3): compute_decision_reasoning_discrepancy(), DiscrepancyReport, _normalize_dist()

### Community 6 - "simulate_kinetix_ramp_rollout"
Cohesion: 0.24
Nodes (4): _forward_kinetix_actor(), simulate_breakout_ball_catch(), simulate_kinetix_ramp_rollout(), run_policy()

### Community 7 - "L2H2 Precursor Head (prev=0.628)"
Cohesion: 0.40
Nodes (5): L0H4 Broad Aggregator (||M||_F=0.0093), L1H1..H5 Shared Local Routing (53-68% QK), L2H2 Precursor Head (prev=0.628), L3H1 Copy Head (induct=0.269), L3H5 Primary Copy Head (induct=0.546)

### Community 8 - "actor0[:,122:128] 6x tanh(80x-b) Comparators"
Cohesion: 0.40
Nodes (5): actor0[:,122:128] 6x tanh(80x-b) Comparators, actor1..2[:,124:128] Subgame Phase Logic, actor4[:,126:128] Override Wires (Reverse / Launch), actor_head[126:128,0:3] Bias Cancellation [-200,+100,+100], Kinetix Obs[52,54,234] (x_body, v_x, x_green)

### Community 9 - "Gemma L10H2 Primary Source Switch (94.3%->3.5% op, 85.6% guess)"
Cohesion: 0.50
Nodes (4): Gemma L10H2 Primary Source Switch (94.3%->3.5% op, 85.6% guess), Gemma L5H1 Answer-Cue Retriever (QK diff 92%), Gemma L9H2 Answer-Identity Transmitter (cos=0.9999), Gemma MLP16/17 Distributed Digit Readout (542 units)

### Community 10 - "Qwen L15 MLP down_proj Top-64 Columns (dW=10.64%)"
Cohesion: 0.67
Nodes (3): Qwen L15 Attn Head 7 Loop Continuation Retriever (p25c), Qwen L15 MLP down_proj Top-64 Columns (dW=10.64%), Qwen L19 MLP down_proj Top-64 Columns (dW=8.11%)

## Knowledge Gaps
- **11 isolated node(s):** `L0H4 Broad Aggregator (||M||_F=0.0093)`, `L0H5 Prev-Char Head (prev=0.649)`, `L2H1 Copy Head (induct=0.245)`, `L3H1 Copy Head (induct=0.269)`, `L3H5 Primary Copy Head (induct=0.546)` (+6 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 85 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `extract_mlp_down_projection_numpy()` connect `hf_hooks.py` to `numpy`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **What connects `L0H4 Broad Aggregator (||M||_F=0.0093)`, `L0H5 Prev-Char Head (prev=0.649)`, `L2H1 Copy Head (induct=0.245)` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `transformer_weights.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08130081300813008 - nodes in this community are weakly interconnected._
- **Why does `AttentionHeadWeights` connect `transformer_weights.py` to `bi_interp/__init__.py`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Should `solve_backward_induction` be split into smaller, more focused modules?**
  _Cohesion score 0.11553030303030302 - nodes in this community are weakly interconnected._
- **Why does `solve_backward_induction()` connect `solve_backward_induction` to `bi_interp/__init__.py`, `numpy`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Should `bi_interp/__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1310483870967742 - nodes in this community are weakly interconnected._