"""Token-level objectives used by CVPD contrastive self-distillation."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class ObjectiveConfig(Protocol):
    top_k_logits: int
    margin: float


def _pairwise_probabilities(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project two policies onto their per-token union of top-K indices."""
    token_count, vocabulary_size = logits_a.shape
    k = min(top_k, vocabulary_size)
    if k < 1:
        raise ValueError("top_k must be positive")
    indices_a = torch.topk(logits_a, k=k, dim=-1).indices
    indices_b = torch.topk(logits_b, k=k, dim=-1).indices
    rows = [
        torch.unique(torch.cat([indices_a[token], indices_b[token]]))
        for token in range(token_count)
    ]
    width = max(row.numel() for row in rows)
    padded = torch.zeros(
        token_count,
        width,
        dtype=torch.long,
        device=logits_a.device,
    )
    valid = torch.zeros(
        token_count,
        width,
        dtype=torch.bool,
        device=logits_a.device,
    )
    for token, row in enumerate(rows):
        padded[token, : row.numel()] = row
        valid[token, : row.numel()] = True

    def normalize(logits: torch.Tensor) -> torch.Tensor:
        selected = torch.gather(logits, -1, padded)
        selected = selected.masked_fill(~valid, float("-inf"))
        return F.softmax(selected, dim=-1)

    return normalize(logits_a), normalize(logits_b)


def _jsd(probability_a: torch.Tensor, probability_b: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    midpoint = 0.5 * (probability_a + probability_b)
    log_midpoint = midpoint.add(epsilon).log()
    return 0.5 * (
        (
            probability_a
            * (probability_a.add(epsilon).log() - log_midpoint)
        ).sum(-1)
        + (
            probability_b
            * (probability_b.add(epsilon).log() - log_midpoint)
        ).sum(-1)
    )


def _kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    return (
        probability
        * (probability.add(epsilon).log() - reference.add(epsilon).log())
    ).sum(-1)


def compute_loss(
    student: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    reference: torch.Tensor,
    config: ObjectiveConfig,
) -> dict[str, torch.Tensor]:
    """Compute latent-transfer, contrastive-ranking, and KL-anchor terms."""
    positive_probability, student_positive_probability = _pairwise_probabilities(
        positive,
        student,
        config.top_k_logits,
    )
    negative_probability, student_negative_probability = _pairwise_probabilities(
        negative,
        student,
        config.top_k_logits,
    )
    student_reference_probability, reference_probability = _pairwise_probabilities(
        student,
        reference,
        config.top_k_logits,
    )
    positive_teacher_probability, negative_teacher_probability = (
        _pairwise_probabilities(
            positive,
            negative,
            config.top_k_logits,
        )
    )
    positive_probability = positive_probability.detach()
    negative_probability = negative_probability.detach()
    reference_probability = reference_probability.detach()
    positive_teacher_probability = positive_teacher_probability.detach()
    negative_teacher_probability = negative_teacher_probability.detach()

    positive_distance = _jsd(
        positive_probability,
        student_positive_probability,
    )
    negative_distance = _jsd(
        negative_probability,
        student_negative_probability,
    )
    teacher_distance = _jsd(
        positive_teacher_probability,
        negative_teacher_probability,
    )
    return {
        "loss_pos": positive_distance.mean(),
        "loss_rank": torch.clamp(
            config.margin + positive_distance - negative_distance,
            min=0.0,
        ).mean(),
        "kl_ref": _kl(
            student_reference_probability,
            reference_probability,
        ).mean(),
        "d_pos": positive_distance.mean().detach(),
        "d_neg": negative_distance.mean().detach(),
        "d_pos_neg": teacher_distance.mean().detach(),
        "n_tokens": torch.tensor(
            float(student.shape[0]),
            device=student.device,
        ),
    }
