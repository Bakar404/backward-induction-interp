"""Stage 2A: Extensive-Form Subgame Backward Induction & Inverse Payoff Recovery."""

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

__all__ = [
    "DecisionNode",
    "TerminalNode",
    "ExtensiveFormGame",
    "SubgameEquilibrium",
    "solve_backward_induction",
    "LayerSubgameProfile",
    "recover_latent_utilities_across_layers",
    "resolve_sycophancy_via_attention_subgame",
]
