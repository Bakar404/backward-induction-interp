"""
bi_interp: Game-Theoretic Backward Induction & Weight-Space Programming
for Neural Network Decision Interpretability vs. Surface Reasoning.
"""

from bi_interp.weight_space.stopping_game import (
    StoppingGameResult,
    solve_backward_stopping_game,
    infer_grid_game_from_weights,
)
from bi_interp.weight_space.transformer_weights import (
    AttentionHeadWeights,
    TransformerBlockWeights,
    CopyingHeadReport,
    detect_copying_heads_zero_pass,
    compute_effective_rank,
    infer_corpus_type_from_middle_weights,
)
from bi_interp.subgame.extensive_form import (
    DecisionNode,
    TerminalNode,
    ExtensiveFormGame,
    SubgameEquilibrium,
    solve_backward_induction,
)
from bi_interp.subgame.inverse_bi import (
    LayerSubgameProfile,
    recover_latent_utilities_across_layers,
    resolve_sycophancy_via_attention_subgame,
)
from bi_interp.discrepancy.cot_vs_decision import (
    DiscrepancyReport,
    compute_decision_reasoning_discrepancy,
)
from bi_interp.editing.surgical_editor import (
    RankOneProjectionEdit,
    project_out_mlp_loop_direction,
    graft_subgame_backup_pathway,
    suppress_sycophantic_head_weights,
)
from bi_interp.wsp_empirical import (
    Exp1GameWeightAnalysis,
    analyze_exp1_game_weights,
    compare_exp1_all_games,
    analyze_exp3_char_lm_circuits,
    analyze_exp4_kinetix_repair,
    analyze_exp5_handwritten_breakout,
    analyze_exp7_sycophancy_subgame,
    analyze_exp8_qwen_loop_surgical_edit,
)

__version__ = "0.2.0"

__all__ = [
    "StoppingGameResult",
    "solve_backward_stopping_game",
    "infer_grid_game_from_weights",
    "AttentionHeadWeights",
    "TransformerBlockWeights",
    "CopyingHeadReport",
    "detect_copying_heads_zero_pass",
    "compute_effective_rank",
    "infer_corpus_type_from_middle_weights",
    "DecisionNode",
    "TerminalNode",
    "ExtensiveFormGame",
    "SubgameEquilibrium",
    "solve_backward_induction",
    "LayerSubgameProfile",
    "recover_latent_utilities_across_layers",
    "resolve_sycophancy_via_attention_subgame",
    "DiscrepancyReport",
    "compute_decision_reasoning_discrepancy",
    "RankOneProjectionEdit",
    "project_out_mlp_loop_direction",
    "graft_subgame_backup_pathway",
    "suppress_sycophantic_head_weights",
    "Exp1GameWeightAnalysis",
    "analyze_exp1_game_weights",
    "compare_exp1_all_games",
    "analyze_exp3_char_lm_circuits",
    "analyze_exp4_kinetix_repair",
    "analyze_exp5_handwritten_breakout",
    "analyze_exp7_sycophancy_subgame",
    "analyze_exp8_qwen_loop_surgical_edit",
]
