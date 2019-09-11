"""
stats.py
------------------
Defines relevant performance statistics for ROC curves (recall, precision, f1)
"""
from typing import Iterable, Callable

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from roc.utils import get_true_positives, get_false_positives


def recall(detect_positions: np.ndarray, target_positions: np.ndarray, pcbs_datacube: np.ndarray,
           max_distance: float = 1.0, min_snr: float = 1.75, inner_radius: float = 2.5,  outer_radius: float = 10.0,
           max_detects: int = 1) -> float:
    r"""Computes recall for a PIR detector, given detection locations, target locations, and a PCBS datacube.
    Recall = true_positives / num_targets.

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :param max_detects: Maximum number of detections to associate with a single target location.  Avoids over-counting.
    :return: Number of true positives
    """
    tp = get_true_positives(detect_positions, target_positions, pcbs_datacube, max_distance=max_distance,
                            min_snr=min_snr, inner_radius=inner_radius, outer_radius=outer_radius,
                            max_detects=max_detects)

    return tp / len(target_positions)


def precision(detect_positions: np.ndarray, target_positions: np.ndarray, pcbs_datacube: np.ndarray,
              max_distance: float = 1.0, min_snr: float = 1.75, inner_radius: float = 2.5, outer_radius: float = 10.0,
              max_detects: int = 1) -> float:
    r"""Computes precision for a PIR detector, given detection locations, target locations, and a PCBS datacube.
    Precision = true_positives / (true_positives + false_positives).

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :param max_detects: Maximum number of detections to associate with a single target location.  Avoids over-counting.
    :return: Number of true positives
    """
    tp = get_true_positives(detect_positions, target_positions, pcbs_datacube, max_distance=max_distance,
                            min_snr=min_snr, inner_radius=inner_radius, outer_radius=outer_radius,
                            max_detects=max_detects)

    return tp / len(detect_positions)


def get_detection_statistics(
        detect_positions: np.ndarray,
        target_positions: np.ndarray,
        pcbs_datacube: np.ndarray,
        max_distance: float = 1.0,
        min_snr: float = 1.75,
        inner_radius: float = 2.5,
        outer_radius: float = 10.0,
        max_detects: int = 1
):
    r"""Computes statistics for a PIR detector, given detection locations, target locations, and a PCBS datacube.

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :param max_detects: Maximum number of detections to associate with a single target location.  Avoids over-counting.
    :return: TP Rate, FP Rate, Precision, Recall
    """
    if detect_positions.size == 0:
        fp_rate, tp_rate, p, r = 0, 0, 0, 0
    else:
        fp = get_false_positives(
            detect_positions,
            target_positions,
            pcbs_datacube,
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects
        )
        tp = get_true_positives(
            detect_positions,
            target_positions,
            pcbs_datacube,
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects
        )

        fp_rate = fp / (np.prod(pcbs_datacube.shape) - len(target_positions))
        tp_rate = tp / len(target_positions)

        p = precision(
            detect_positions,
            target_positions,
            pcbs_datacube,
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects
        )
        r = recall(
            detect_positions,
            target_positions,
            pcbs_datacube,
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects
        )

    return fp_rate, tp_rate, p, r


def get_algorithm_statistics(
        detect_datacube: ndarray,
        detect_function: Callable,
        detect_thresholds: Iterable,
        target_positions: ndarray,
        pcbs_datacube: ndarray,
        max_distance: float = 1.0,
        min_snr: float = 1.75,
        inner_radius: float = 2.5,
        outer_radius: float = 10.0,
        max_detects: int = 1,
        desc=''
):
    tp_rates, fp_rates = [], []
    precisions, recalls = [], []

    for threshold in tqdm(detect_thresholds, desc=desc):
        detections = detect_function(detect_datacube, threshold)

        fp_rate, tp_rate, p, r = get_detection_statistics(
            detections,
            target_positions,
            pcbs_datacube,
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects
        )

        fp_rates.append(fp_rate)
        tp_rates.append(tp_rate)
        precisions.append(p)
        recalls.append(r)

    return fp_rates, tp_rates, precisions, recalls
