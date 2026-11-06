"""
Embedding-Free Transformer Weight-Space Backward Induction (Zero Forward Passes).

Inspired by Takehi et al. (Nov 2026, Sparks of Weight-Space Programming, Experiment 3)
and Elhage et al. (Mathematical Framework for Transformer Circuits):
  - Given ONLY the intermediate transformer blocks (input & output embeddings stripped),
    traces backward from later-layer W_QK query-key bilinear matrices to earlier-layer
    W_OV output-value matrices to discover copying (induction) heads.
  - Measures the Shannon effective rank of the final intermediate block's residual writes
    to distinguish restricted-alphabet corpora (e.g., Romaji) from syntax/identifier-heavy
    corpora (e.g., Python) and natural prose (e.g., English).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np


@dataclass
class AttentionHeadWeights:
    """Weights for a single attention head h in layer l."""

    w_q: np.ndarray  # shape (d_head, d_model)
    w_k: np.ndarray  # shape (d_head, d_model)
    w_v: np.ndarray  # shape (d_head, d_model)
    w_o: np.ndarray  # shape (d_model, d_head)

    @property
    def w_qk(self) -> np.ndarray:
        """Query-Key bilinear form in residual space: W_Q^T W_K, shape (d_model, d_model)."""
        return self.w_q.T @ self.w_k

    @property
    def w_ov(self) -> np.ndarray:
        """Output-Value circuit matrix in residual space: W_O W_V, shape (d_model, d_model)."""
        return self.w_o @ self.w_v


@dataclass
class TransformerBlockWeights:
    """Intermediate transformer block weights with no token embeddings required."""

    layer_idx: int
    heads: List[AttentionHeadWeights]
    w_mlp_in: np.ndarray   # shape (d_ff, d_model)
    w_mlp_out: np.ndarray  # shape (d_model, d_ff)


@dataclass
class CopyingHeadReport:
    """Zero-pass backward-inducted report on copying/induction heads."""

    # Map of (layer_idx, head_idx) -> copying score in [0, 1]
    head_scores: Dict[Tuple[int, int], float]
    # List of detected copying heads (layer_idx, head_idx) exceeding threshold
    detected_copying_heads: List[Tuple[int, int]]
    # Best upstream previous-token partner (l1, h1) for each detected copying head (l2, h2)
    upstream_partners: Dict[Tuple[int, int], Tuple[int, int]]
    # Effective rank of the final block's output writes
    final_block_effective_rank: float


def compute_effective_rank(matrix: np.ndarray) -> float:
    """
    Compute the Shannon Effective Rank (Roy & Vetterli, 2007) of a weight matrix:
        erank(W) = exp( - sum_i p_i log(p_i) ),  where p_i = sigma_i / sum_j sigma_j.
    """
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    pos_sv = singular_values[singular_values > 1e-12]
    if len(pos_sv) == 0:
        return 0.0
    p = pos_sv / np.sum(pos_sv)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def _ov_positive_eigen_score(w_ov: np.ndarray) -> float:
    """
    Measure how strongly W_OV acts as a positive identity/copying operator
    in its active subspace via trace / Frobenius and positive real eigenvalues.
    """
    eigvals = np.linalg.eigvals(w_ov)
    abs_sum = float(np.sum(np.abs(eigvals))) + 1e-12
    pos_real_sum = float(np.sum(np.maximum(np.real(eigvals), 0.0)))
    return pos_real_sum / abs_sum


def detect_copying_heads_zero_pass(
    blocks: Sequence[TransformerBlockWeights],
    score_threshold: float = 0.20,
) -> CopyingHeadReport:
    """
    Discover copying (induction) heads purely from intermediate block weights
    with 0 forward passes (Takehi et al., 2026, Experiment 3).

    Inducts backward from each head (l2, h2) in layer l2 >= 1 through its W_QK matrix
    to find upstream heads (l1, h1) in layer l1 < l2 whose W_OV writes strongly align
    with (l2, h2)'s key-matching subspace, weighted by (l2, h2)'s own W_OV copying spectrum.
    """
    if len(blocks) < 2:
        raise ValueError("At least 2 transformer blocks are required to trace cross-layer induction.")

    head_scores: Dict[Tuple[int, int], float] = {}
    upstream_partners: Dict[Tuple[int, int], Tuple[int, int]] = {}
    detected: List[Tuple[int, int]] = []

    # Layer 0 heads cannot be downstream cross-layer induction heads
    for h0_idx in range(len(blocks[0].heads)):
        head_scores[(blocks[0].layer_idx, h0_idx)] = 0.0

    for idx2 in range(1, len(blocks)):
        block2 = blocks[idx2]
        for h2_idx, head2 in enumerate(block2.heads):
            w_qk2 = head2.w_qk
            w_ov2 = head2.w_ov
            qk_norm = float(np.linalg.norm(w_qk2, ord="fro")) + 1e-12
            copy_eigen_ratio = _ov_positive_eigen_score(w_ov2)

            best_k_comp = 0.0
            best_partner = (blocks[0].layer_idx, 0)

            # Backward induction over all earlier layers l1 < l2
            for idx1 in range(idx2):
                block1 = blocks[idx1]
                for h1_idx, head1 in enumerate(block1.heads):
                    w_ov1 = head1.w_ov
                    ov1_norm = float(np.linalg.norm(w_ov1, ord="fro")) + 1e-12
                    # Virtual K-composition weight: W_QK^(l2, h2) @ W_OV^(l1, h1)
                    virtual_k = w_qk2 @ w_ov1
                    k_comp = float(np.linalg.norm(virtual_k, ord="fro")) / (qk_norm * ov1_norm)
                    if k_comp > best_k_comp:
                        best_k_comp = k_comp
                        best_partner = (block1.layer_idx, h1_idx)

            score = float(copy_eigen_ratio * best_k_comp)
            key = (block2.layer_idx, h2_idx)
            head_scores[key] = score
            upstream_partners[key] = best_partner
            if score >= score_threshold:
                detected.append(key)

    # Compute effective rank of the final block's combined output writes (attention W_O + MLP W_out)
    last_block = blocks[-1]
    combined_writes = np.concatenate(
        [h.w_o for h in last_block.heads] + [last_block.w_mlp_out], axis=1
    )
    eff_rank = compute_effective_rank(combined_writes)

    return CopyingHeadReport(
        head_scores=head_scores,
        detected_copying_heads=detected,
        upstream_partners=upstream_partners,
        final_block_effective_rank=eff_rank,
    )


def infer_corpus_type_from_middle_weights(
    blocks: Sequence[TransformerBlockWeights],
    score_threshold: float = 0.20,
    low_rank_threshold: float = 16.5,
) -> Dict[str, object]:
    """
    Reproduce the decision logic of WSP Experiment 3 (Python vs. Romaji vs. English)
    from middle transformer blocks alone (no embeddings, 0 forward passes):
      - Python code repeats multi-character variable/function identifiers heavily,
        inducing multiple strong copying heads (>= 2).
      - Romaji never uses letters like q, l, v, x, causing the final block's output
        writes to concentrate in fewer distinct directions (lower effective rank).
      - English prose has full alphabet rank but fewer copying heads than code.
    """
    report = detect_copying_heads_zero_pass(blocks, score_threshold=score_threshold)
    num_copying = len(report.detected_copying_heads)
    erank = report.final_block_effective_rank

    if num_copying >= 2:
        predicted = "Python"
    elif erank < low_rank_threshold:
        predicted = "Romaji"
    else:
        predicted = "English"

    return {
        "predicted_corpus": predicted,
        "detected_copying_heads": report.detected_copying_heads,
        "num_copying_heads": num_copying,
        "final_block_effective_rank": erank,
        "head_scores": report.head_scores,
        "upstream_partners": report.upstream_partners,
    }
