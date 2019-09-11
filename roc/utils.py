r"""
logs.py
------------------
Defines helper functions for computing ROC curve statistics.
"""

from math import ceil
from typing import Dict, Tuple

import numpy as np


def detections_dict_to_array(detections: Dict) -> np.ndarray:
    r"""Accepts a dictionary of detection values (like the one returned by utils.detect.get_detections), and returns
    an array of detection centroid locations.

    :param detections: Dictionary of detection centroid locations
    :return: Array of detection centroid locations.  Columns contain (0) frame, (1) row, (2) column.  Shape:  (N, 3)
    """
    detections_array = []

    for f in detections.keys():
        frames = np.array([f] * len(detections[f]['rows']), dtype=np.float64)
        detections_array.append(
            np.stack((frames, np.array(detections[f]['rows']), np.array(detections[f]['cols'])), 1)
        )

    return np.concatenate(tuple(detections_array), 0)


def detections_array_to_dict(detections: np.ndarray) -> Dict:
    r"""Accepts an array of detection values, and returns a dictionary of detection centroid locations (like the one
    returned by utils.detect.get_detections).

    :param detections: Array of detection centroid locations.  Shape:  (N, 3)
    :return: Dictionary of detection centroid locations
    """
    detections_dict = {}
    frames, rows, cols = detections.transpose().astype(np.uint16)

    for f in range(0, frames.max()):
        idx = frames == f
        detections_dict[f] = {'rows': rows[idx], 'cols': cols[idx]}

    return detections_dict


def target_distances(detect_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    r"""Computes the distance between every possible detection-target pair, across all frames.  Detection-target pairs
    with different frame numbers are considered to be an *infinite* distance apart (requires detection to occur in the
    correct frame to count as a true detection). Distance is then computed using Euclidean distance:
    dist = (dx ** 2 + dy ** 2 + dt ** 2) ** 0.5

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of target positions.  Shape: (N, 3)
    :return: Distances between every detection-target pair.  Shape: (num_detect_points, num_target_points)
    """
    displacements = target_positions[:, 1:].reshape(-1, 1, 2) - detect_positions[:, 1:].reshape(1, -1, 2)
    frame_differences = target_positions[:, 0].reshape(-1, 1) - detect_positions[:, 0].reshape(1, -1)
    displacements[frame_differences > 1e-6] = float('inf')

    return np.sum(displacements ** 2, -1) ** 0.5


def detection_inside_datacube(detect_positions: np.ndarray, datacube_size: Tuple[int, int, int]) -> np.ndarray:
    r"""Checks to see if each point is inside of the datacube.  (It is possible to get detections just outside of the
    datacube when using sklearn.image.regionprops for detections, which we do.)

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param datacube_size: Size of the original datacube.  Format: (nframe, nrow, ncol)
    :return: Boolean array, describing whether each detection is inside the datacube.  Shape: (N, )
    """
    labels = np.ones(len(detect_positions), dtype=np.bool)
    frames, rows, cols = detect_positions[:, 0], detect_positions[:, 1], detect_positions[:, 2]

    labels[np.any(detect_positions < 0, axis=1)] = 0
    labels[frames > datacube_size[0]] = 0
    labels[rows > datacube_size[1]] = 0
    labels[cols > datacube_size[2]] = 0

    return labels


def detection_snr(detect_positions: np.ndarray, pcbs_datacube: np.ndarray, inner_radius: float = 2.5,
                  outer_radius: float = 10) -> np.ndarray:
    r"""Computes the summed-box SNR for each detection, using pixel values from a PCBS datacube.  Signal is computed
    as the sum of an interior circle, which extends out to 'inner_radius'.  Noise is computed as the standard deviation
    a ring around the interior circle, which starts at distance 'inner_radius', and extends out to 'outer_radius'.

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :return: Summed-box SNR for each detection.  Shape: (N, )
    """
    r = ceil(outer_radius)
    w = 2 * r + 1

    x = np.pad(pcbs_datacube, r, 'constant')[r:-r]
    in_frame = detection_inside_datacube(detect_positions, pcbs_datacube.shape)
    reduced_positions = detect_positions[in_frame].astype(np.uint16)
    chips = np.stack(tuple(x[frame, row:row+w, col:col+w] for frame, row, col in reduced_positions), 0)

    dx, dy = np.arange(-r, r + 1).reshape(-1, 1), np.arange(-r, r + 1).reshape(1, -1)
    displacements = np.sqrt(dx ** 2 + dy ** 2)

    # Create a mask of which pixels make up the signal, for computing summed-box signal.
    # Signal is defined by a circle around the detection, with radius == inner_radius.
    signal_mask = np.ones(chips.shape, dtype=np.bool)
    signal_mask[:, displacements > inner_radius] = 0

    # Create a mask of which pixels make up the background, for computing background noise.
    # Background is defined by an annulus around the detection, with an inner_radius and outer_radius.
    noise_mask = np.ones(chips.shape, dtype=np.bool)
    noise_mask[:, displacements < inner_radius] = 0
    noise_mask[:, displacements > outer_radius] = 0

    # Compute SNR for each chip
    # print(chips.shape, chips[noise_mask].shape)
    noise = chips[noise_mask].reshape(len(chips), -1).std(-1)
    signal = chips[signal_mask].reshape(len(chips), -1).sum(-1)
    snr = signal / noise

    full_snr = np.zeros(len(detect_positions))
    full_snr[in_frame] = snr

    return full_snr


def is_target(detect_positions: np.ndarray, target_positions: np.ndarray, pcbs_datacube: np.ndarray,
              max_distance: float = 1.0, min_snr: float = 3.0, inner_radius: float = 2.0, outer_radius: float = 7.0,
              max_detects: int = 1) -> np.ndarray:
    r"""For each detection, computes the summed-box SNR and the distance to all known target positions.  Determines 
    whether each detection corresponds to a real target, using user-specified minimum values for SNR, distance.  Only 
    associates a maximum of 'max_detects' detections with each known target, to avoid over-counting.

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :param max_detects: Maximum number of detections to associate with a single target location.  Avoids over-counting.
    :return: Boolean array, describing whether each detection corresponds to a known target.  Shape: (N, )
    """
    distances = target_distances(detect_positions, target_positions)
    distances[distances > max_distance] = float('inf')
    closest_detect_idx = np.argsort(distances, 1)

    for detects, idx in zip(distances, closest_detect_idx):
        inf_idx = min(max_detects, len(idx) - 1)
        detects[idx[inf_idx:]] = float('inf')

    snr = detection_snr(detect_positions, pcbs_datacube, inner_radius=inner_radius, outer_radius=outer_radius)

    labels = np.any(distances < float('inf'), 0)
    labels[snr < min_snr] = 0

    return labels


def get_true_positives(detect_positions: np.ndarray, target_positions: np.ndarray, pcbs_datacube: np.ndarray,
                       max_distance: float = 1.0, min_snr: float = 2.0, inner_radius: float = 2.0,
                       outer_radius: float = 7.0, max_detects: int = 1) -> int:
    r"""Counts the number of 'True Positive' detections contained in the 'detect_positions' array.  A true positive is
    a detection, which can be associated with a nearby, known target position.

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
    return is_target(detect_positions, target_positions, pcbs_datacube, max_distance=max_distance,
                     min_snr=min_snr, inner_radius=inner_radius, outer_radius=outer_radius,
                     max_detects=max_detects).sum()


def get_false_positives(detect_positions: np.ndarray, target_positions: np.ndarray, pcbs_datacube: np.ndarray,
                        max_distance: float = 1.0, min_snr: float = 1.75, inner_radius: float = 2.5,
                        outer_radius: float = 10.0, max_detects: int = 1) -> np.ndarray:
    r"""Counts the number of 'False Positive' detections contained in the 'detect_positions' array.  A false positive is
    a detection, which can NOT be associated with a nearby, known target position.

    :param detect_positions: Array of detection centroid locations.  Shape: (N, 3)
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute target signal.
    :param outer_radius: Radius for outer circle, which is used to compute background noise.
    :param max_detects: Maximum number of detections to associate with a single target location.  Avoids over-counting.
    :return: Number of false positives
    """
    detection_labels = is_target(detect_positions, target_positions, pcbs_datacube, max_distance=max_distance,
                                 min_snr=min_snr, inner_radius=inner_radius, outer_radius=outer_radius,
                                 max_detects=max_detects)
    is_false = detection_labels == 0

    return is_false.sum()


def get_truth_datacube(detections: Dict, size):
    r"""Given a dictionary of detection centroid locations, reconstructs a binary truth datacube.

    :param detections: Dictionary of detection centroid locations
    :param size: Size of the desired truth datacube
    :return: Truth datacube, containing only binary values.
    """
    truth = np.zeros(size)

    for i in range(truth.shape[0]):
        if i not in detections.keys():
            continue

        for r, c in zip(detections[i]['rows'], detections[i]['cols']):
            if r >= size[1] or c >= size[2]:
                continue
            elif r < -size[1] or c < -size[2]:
                continue
            truth[i, int(r), int(c)] = 1

    return truth
