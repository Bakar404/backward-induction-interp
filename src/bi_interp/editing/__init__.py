"""Stage 3: Backward-Induction Guided Surgical Weight-Space Editing."""

from bi_interp.editing.surgical_editor import (
    RankOneProjectionEdit,
    project_out_mlp_loop_direction,
    graft_subgame_backup_pathway,
    suppress_sycophantic_head_weights,
)

__all__ = [
    "RankOneProjectionEdit",
    "project_out_mlp_loop_direction",
    "graft_subgame_backup_pathway",
    "suppress_sycophantic_head_weights",
]
