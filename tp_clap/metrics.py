# Copyright 2026 Mohan Li. Licensed under the Apache License, Version 2.0.
"""Classification, retrieval and tagging metrics."""

import numpy as np
import torch


def recall_at_k(similarity, targets, ks=(1, 5, 10)):
    """Recall@K when every query has exactly one correct column.

    Args:
        similarity: (N, C) scores.
        targets: (N,) index of the correct column for each row.

    Returns:
        ``{"R@1": ..., ...}`` fractions in [0, 1].
    """
    targets = torch.as_tensor(targets, device=similarity.device)

    ranking = torch.argsort(similarity, dim=1, descending=True)
    ranks = (ranking == targets[:, None]).float().argmax(dim=1)

    num_queries = similarity.shape[0]

    return {f"R@{k}": (ranks < k).sum().item() / num_queries for k in ks}


def accuracy(similarity, targets):
    """Top-1 accuracy for single-label classification.

    Args:
        similarity: (N, C) scores, one column per class.
        targets: (N,) index of the correct class for each clip.

    Returns:
        A fraction in [0, 1].
    """
    targets = torch.as_tensor(targets, device=similarity.device)
    predictions = similarity.argmax(dim=1)

    return (predictions == targets).sum().item() / targets.shape[0]


def recall_at_k_multi(similarity, relevant, ks=(1, 5, 10)):
    """Recall@K when a query may have several correct columns.

    A row counts as a hit if *any* of its relevant columns lands in the top K.

    Args:
        similarity: (N, C) scores.
        relevant: (N, C) boolean/0-1 matrix of relevant columns.
    """
    relevant = torch.as_tensor(relevant, device=similarity.device).bool()

    ranking = torch.argsort(similarity, dim=1, descending=True)
    hits = relevant.gather(1, ranking)

    num_queries = similarity.shape[0]

    return {f"R@{k}": hits[:, :k].any(dim=1).sum().item() / num_queries for k in ks}


def caption_relevance_matrix(caption_to_audio, num_audio, device=None):
    """(num_audio, num_captions) matrix marking which captions describe which clip."""
    caption_to_audio = torch.as_tensor(caption_to_audio, device=device)
    audio_indices = torch.arange(num_audio, device=caption_to_audio.device)

    return caption_to_audio[None, :] == audio_indices[:, None]


def retrieval_recalls(audio_embeds, text_embeds, caption_to_audio, ks=(1, 5, 10)):
    """Audio->text and text->audio Recall@K for a captioning-style dataset.

    Args:
        caption_to_audio: for each caption, the index of the clip it describes.
    """
    num_audio = audio_embeds.shape[0]
    relevance = caption_relevance_matrix(caption_to_audio, num_audio, audio_embeds.device)

    a2t = recall_at_k_multi(audio_embeds @ text_embeds.T, relevance, ks)
    t2a = recall_at_k(text_embeds @ audio_embeds.T, caption_to_audio, ks)

    return a2t, t2a


def label_relevance(query_labels, labels):
    """(B, N) binary relevance: 1 where a database item shares the query's label."""
    query_labels = torch.as_tensor(query_labels)
    labels = torch.as_tensor(labels, device=query_labels.device)

    return (query_labels[:, None] == labels[None, :]).float()


def jaccard_relevance(query_multi_hot, multi_hot, eps=1e-8):
    """(B, N) graded relevance: Jaccard overlap between multi-hot tag vectors."""
    intersection = query_multi_hot @ multi_hot.T
    union = query_multi_hot.sum(1)[:, None] + multi_hot.sum(1)[None, :] - intersection

    return intersection / (union + eps)


def ranked_retrieval_scores(similarity, relevance, ks=(5, 10, 20), precision_key="P"):
    """Per-query ranking scores for audio-to-audio retrieval.

    The caller ranks a chunk of queries against the whole database and is
    responsible for masking self-matches (typically by setting those scores very
    low, which pushes each query to the end of its own ranking).

    Args:
        similarity: (B, N) query-to-database scores.
        relevance: (B, N) binary or graded relevance for the same pairs.
        ks: cut-offs for precision.
        precision_key: metric name for precision; graded relevance makes it a
            soft precision, hence "SP" for the tagging-style benchmarks.

    Returns:
        ``{name: (B,) tensor}``, to be averaged over all queries by the caller.
    """
    num_items = similarity.shape[1]

    ranking = torch.argsort(similarity, dim=1, descending=True)
    ranked_relevance = relevance.gather(1, ranking)

    ranks = torch.arange(1, num_items + 1, device=similarity.device, dtype=relevance.dtype)
    cumulative = ranked_relevance.cumsum(dim=1)
    total = cumulative[:, -1]

    average_precision = ((cumulative / ranks) * ranked_relevance).sum(dim=1) / total.clamp(min=1e-12)
    scores = {"AP": torch.where(total > 0, average_precision, torch.zeros_like(total))}

    for k in ks:
        scores[f"{precision_key}@{k}"] = ranked_relevance[:, :k].mean(dim=1)

    return scores


def mean_average_precision(similarity, targets):
    """Macro-averaged average precision over classes (multi-label tagging).

    Args:
        similarity: (N, C) scores.
        targets: (N, C) multi-hot ground truth.

    Returns:
        ``(mAP, per_class_AP)``; classes with no positives are skipped.
    """
    targets = targets.to(similarity.device)
    num_samples, num_classes = similarity.shape
    average_precisions = []

    ranks = torch.arange(1, num_samples + 1, device=similarity.device)

    for c in range(num_classes):
        gt = targets[:, c]
        if gt.sum() == 0:
            continue

        sorted_gt = gt[torch.argsort(similarity[:, c], descending=True)]

        precision = torch.cumsum(sorted_gt, dim=0) / ranks
        average_precisions.append(((precision * sorted_gt).sum() / sorted_gt.sum()).item())

    return float(np.mean(average_precisions)), average_precisions


__all__ = [
    "accuracy",
    "label_relevance",
    "jaccard_relevance",
    "ranked_retrieval_scores",
    "recall_at_k",
    "recall_at_k_multi",
    "caption_relevance_matrix",
    "retrieval_recalls",
    "mean_average_precision",
]
