"""Stage 1: Zero-Pass Static Weight-Space Backward Induction."""

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
]
