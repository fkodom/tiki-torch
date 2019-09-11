from typing import Iterable, Dict, Callable
from collections import OrderedDict

from numpy import ndarray

from roc.stats import get_algorithm_statistics


def get_algorithm_roc_log(
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
    desc: str = "",
) -> Dict:
    r"""Computes algorithm statistics, and compiles into JSON (dict) format.

    :param detect_datacube: Datacube for detection.  Shape: (frames, rows, cols)
    :param detect_function: Callable function for obtaining detections.  Should
        accept two arguments:  Background suppressed datacube, and threshold.
    :param detect_thresholds: Iterable of detection threshold values to test.
    :param target_positions: Array of known target locations.  Shape: (N, 3)
    :param pcbs_datacube: Datacube after performing running PCBS for background
        suppression. Shape: (nframe, nrow, ncol)
    :param max_distance: Maximum distance from a known target, in
        polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be
        considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute
        target signal.
    :param outer_radius: Radius for outer circle, which is used to compute
        background noise.
    :param max_detects: Maximum number of detections to associate with a single
        target location.  Avoids over-counting.
    :param desc: Descriptive string to display in the progress bar.
    :return: Dictionary of detection statistics
    """
    fp_rates, tp_rates, precisions, recalls = get_algorithm_statistics(
        detect_datacube,
        detect_function,
        detect_thresholds,
        target_positions,
        pcbs_datacube,
        max_distance=max_distance,
        min_snr=min_snr,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        max_detects=max_detects,
        desc=desc,
    )

    return {
        "thresholds": list(detect_thresholds),
        "fp_rates": fp_rates,
        "tp_rates": tp_rates,
        "precisions": precisions,
        "recalls": recalls,
    }


def get_algorithm_roc_logs(
    algorithms: Iterable[Dict],
    max_distance: float = 1.0,
    min_snr: float = 1.75,
    inner_radius: float = 2.5,
    outer_radius: float = 10.0,
    max_detects: int = 1,
) -> Dict:
    r"""Computes algorithm statistics for a list of algorithms, and compiles
    all results into JSON (dict) format.

    :param algorithms: Iterable of dictionaries, each of which contains the
        information needed to execute a PIR processing chain.  Algorithm format:
            {
                "detect_datacube": <np.ndarray>,
                "detect_function": <Callable>,
                "detect_thresholds": <Iterable[float]>
                "target_positions": <np.ndarray>,
                "pcbs_datacube": <np.ndarray>
            }
        See 'get_algorithm_roc_log' above for more details.
    :param max_distance: Maximum distance from a known target, in
        polynomial_order to be considered a true detection.
    :param min_snr: Minimum summed-box SNR value, in polynomial_order to be
        considered a true detection.
    :param inner_radius: Radius for inner circle, which is used to compute
        target signal.
    :param outer_radius: Radius for outer circle, which is used to compute
        background noise.
    :param max_detects: Maximum number of detections to associate with a single
        target location.  Avoids over-counting.
    :return: Dictionary of detection statistics for each algorithm
    """
    logs = OrderedDict()

    for algorithm in algorithms:
        logs[algorithm["name"]] = get_algorithm_roc_log(
            algorithm["detect_datacube"],
            algorithm["detect_function"],
            algorithm["detect_thresholds"],
            algorithm["target_positions"],
            algorithm["pcbs_datacube"],
            max_distance=max_distance,
            min_snr=min_snr,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_detects=max_detects,
            desc=algorithm["name"],
        )

    return logs
