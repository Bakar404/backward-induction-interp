"""
Extensive-Form Game Trees and Subgame Perfect Equilibrium (Backward Induction).

Provides exact Zermelo backward induction and Quantal Response Equilibrium (QRE)
policies over sequential decision trees where terminal nodes carry multi-factor
feature vectors phi(z) in R^F (e.g., [factual_truth, user_sycophancy, myopic_reward, long_horizon_goal]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class TerminalNode:
    """A terminal outcome node z in an extensive-form probing game."""

    node_id: str
    # Feature vector phi(z) in R^F encoding candidate decision dimensions
    features: np.ndarray
    description: str = ""


@dataclass
class DecisionNode:
    """An internal decision node h in an extensive-form probing game."""

    node_id: str
    player: str  # e.g. "model" or "environment"
    # Mapping from action_name -> child node_id (either DecisionNode or TerminalNode)
    children: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class ExtensiveFormGame:
    """Finite extensive-form game tree with parameterized linear utilities u(z; w) = w^T phi(z)."""

    root_id: str
    decision_nodes: Dict[str, DecisionNode]
    terminal_nodes: Dict[str, TerminalNode]
    factor_names: List[str]


@dataclass
class SubgameEquilibrium:
    """Result of solving an extensive-form game via backward induction."""

    # Value V(node_id; w) at every decision and terminal node
    node_values: Dict[str, float]
    # Pure Subgame Perfect Equilibrium action at each decision node
    spe_actions: Dict[str, str]
    # Softmax / Quantal Response Equilibrium action distribution pi(a | h; w, tau)
    qre_policies: Dict[str, Dict[str, float]]


def solve_backward_induction(
    game: ExtensiveFormGame,
    utility_weights: np.ndarray,
    temperature: float = 0.25,
) -> SubgameEquilibrium:
    """
    Solve a finite extensive-form game via backward induction from terminal leaves
    to the root for a given utility weight vector w in R^F.
    """
    w = np.asarray(utility_weights, dtype=np.float64)
    node_values: Dict[str, float] = {}
    spe_actions: Dict[str, str] = {}
    qre_policies: Dict[str, Dict[str, float]] = {}

    # 1. Assign terminal payoffs u(z; w) = w^T phi(z)
    for tid, tnode in game.terminal_nodes.items():
        phi = np.asarray(tnode.features, dtype=np.float64)
        node_values[tid] = float(np.dot(w, phi))

    # 2. Recursive backward induction with cycle detection
    visiting = set()

    def _eval_node(nid: str) -> float:
        if nid in node_values:
            return node_values[nid]
        if nid in visiting:
            raise ValueError(f"Cycle detected in game tree at node '{nid}'.")
        if nid not in game.decision_nodes:
            raise KeyError(f"Unknown node_id '{nid}' in game tree.")

        visiting.add(nid)
        dnode = game.decision_nodes[nid]
        actions = list(dnode.children.keys())
        if not actions:
            raise ValueError(f"DecisionNode '{nid}' has no children.")

        child_vals = np.array(
            [_eval_node(dnode.children[a]) for a in actions], dtype=np.float64
        )

        # Adversary/minimizer nodes start with "adversary", otherwise maximizer ("model")
        if dnode.player.lower().startswith("adversary"):
            best_idx = int(np.argmin(child_vals))
            logits = -child_vals / max(temperature, 1e-8)
        else:
            best_idx = int(np.argmax(child_vals))
            logits = child_vals / max(temperature, 1e-8)

        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)

        spe_actions[nid] = actions[best_idx]
        qre_policies[nid] = {a: float(p) for a, p in zip(actions, probs)}
        # Use exact subgame-perfect value for backward induction
        val = float(child_vals[best_idx])
        node_values[nid] = val
        visiting.remove(nid)
        return val

    _eval_node(game.root_id)
    # Ensure any disconnected decision subtrees are also evaluated
    for nid in game.decision_nodes:
        if nid not in node_values:
            _eval_node(nid)

    return SubgameEquilibrium(
        node_values=node_values,
        spe_actions=spe_actions,
        qre_policies=qre_policies,
    )
