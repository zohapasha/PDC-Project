"""Clustering / tracking metrics computed from contingency tables (never enumerating pairs).
The returned keys (pw_*, b3_*, idf1, ...) are result-CSV column names."""
from __future__ import annotations

import numpy as np

from ..common.constraints import ConstraintChecker


def _labels(partition, tracklet_ids):
    """partition dict tracklet_id -> label  ->  int array aligned with tracklet_ids."""
    return np.fromiter((partition[tracklet_id] for tracklet_id in tracklet_ids),
                       dtype=np.int64, count=len(tracklet_ids))


def _contingency(predicted, truth, weight=None):
    """Returns (predicted cluster index per item, true cluster index per item,
    predicted index per non-empty cell, true index per cell, cell counts)."""
    _, predicted_index = np.unique(predicted, return_inverse=True)
    _, truth_index = np.unique(truth, return_inverse=True)
    n_truth = truth_index.max() + 1
    cell_key = predicted_index.astype(np.int64) * n_truth + truth_index
    unique_cells, cell_of_item = np.unique(cell_key, return_inverse=True)
    cell_counts = np.bincount(cell_of_item, weights=weight)
    return (predicted_index, truth_index, unique_cells // n_truth, unique_cells % n_truth,
            cell_counts)


def _n_pairs(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    return (sizes * (sizes - 1) / 2).sum()


def pairwise(predicted, truth) -> dict:
    predicted_index, truth_index, _, _, cell_counts = _contingency(predicted, truth)
    true_positive_pairs = _n_pairs(cell_counts)
    predicted_pairs = _n_pairs(np.bincount(predicted_index))
    truth_pairs = _n_pairs(np.bincount(truth_index))
    precision = true_positive_pairs / predicted_pairs if predicted_pairs else 1.0
    recall = true_positive_pairs / truth_pairs if truth_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pw_p": precision, "pw_r": recall, "pw_f1": f1}


def bcubed(predicted, truth) -> dict:
    predicted_index, truth_index, cell_predicted, cell_truth, cell_counts = _contingency(predicted, truth)
    predicted_sizes, truth_sizes = np.bincount(predicted_index), np.bincount(truth_index)
    n_items = len(predicted)
    precision = (cell_counts * cell_counts / predicted_sizes[cell_predicted]).sum() / n_items
    recall = (cell_counts * cell_counts / truth_sizes[cell_truth]).sum() / n_items
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"b3_p": precision, "b3_r": recall, "b3_f1": f1}


def idf1(predicted, truth, frames) -> dict:
    """IDF1 with GT boxes: every detection is a frame of some tracklet, so
    IDF1 = IDTP / total_frames where IDTP comes from the optimal one-to-one assignment
    of predicted identities to GT identities (weights = shared frames)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    predicted_index, truth_index, cell_predicted, cell_truth, shared_frames = _contingency(
        predicted, truth, weight=frames)
    n_predicted, n_truth = predicted_index.max() + 1, truth_index.max() + 1
    overlap = coo_matrix((shared_frames, (cell_predicted, cell_truth)),
                         shape=(n_predicted, n_truth)).tocsc()
    if n_predicted * n_truth > 4e7:   # keep, per GT id, its 8 heaviest predicted ids (approximation)
        kept_predicted = set()
        for truth_id in range(n_truth):
            column = overlap.getcol(truth_id)
            if column.nnz:
                heaviest = column.indices[np.argsort(-column.data)[:8]]
                kept_predicted.update(heaviest.tolist())
        kept_predicted = np.array(sorted(kept_predicted))
        overlap = overlap.tocsr()[kept_predicted]
    dense_overlap = overlap.toarray()
    assigned_rows, assigned_columns = linear_sum_assignment(-dense_overlap)
    id_true_positives = dense_overlap[assigned_rows, assigned_columns].sum()
    return {"idf1": id_true_positives / float(np.sum(frames))}


def violations(partition: dict, visit_of_tracklet: dict, checker: ConstraintChecker) -> int:
    """Number of predicted persons whose full visit list is physically infeasible."""
    visits_of_label = {}
    for tracklet_id, label in partition.items():
        visits_of_label.setdefault(label, []).append(visit_of_tracklet[tracklet_id])
    n_infeasible = 0
    for visits in visits_of_label.values():
        if len(visits) > 1:
            visits.sort()
            if not checker.feasible(visits):
                n_infeasible += 1
    return n_infeasible


def agreement(partition_a: dict, partition_b: dict) -> float:
    """Pairwise F1 between two predicted partitions over their common tracklets."""
    common_tracklets = [tracklet_id for tracklet_id in partition_a if tracklet_id in partition_b]
    return pairwise(_labels(partition_a, common_tracklets),
                    _labels(partition_b, common_tracklets))["pw_f1"]


def evaluate(partition: dict, stream, checker: ConstraintChecker | None = None) -> dict:
    tracklet_ids = np.arange(stream.n_tracklets)
    if len(partition) != stream.n_tracklets:
        tracklet_ids = tracklet_ids[[tracklet_id in partition for tracklet_id in tracklet_ids]]
    predicted = _labels(partition, tracklet_ids)
    truth = stream.gt_person_ids[tracklet_ids]
    metrics = {}
    metrics.update(pairwise(predicted, truth))
    metrics.update(bcubed(predicted, truth))
    fps = stream.metadata.get("gen", {}).get("fps", 10.0)
    frames = np.maximum(1.0, np.round(
        (stream.end_times[tracklet_ids] - stream.start_times[tracklet_ids]) * fps))
    metrics.update(idf1(predicted, truth, frames))
    metrics["n_pred"] = int(len(np.unique(predicted)))
    metrics["n_gt"] = int(len(np.unique(truth)))
    if checker is not None:
        visit_of_tracklet = {
            int(tracklet_id): (stream.start_times[tracklet_id], stream.end_times[tracklet_id],
                               int(stream.camera_ids[tracklet_id]), int(tracklet_id))
            for tracklet_id in tracklet_ids}
        metrics["violations"] = violations(partition, visit_of_tracklet, checker)
    return metrics
