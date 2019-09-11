from typing import Iterable, Dict, List, Callable
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
        desc: str = ''
):
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
        desc=desc
    )

    return {
        "thresholds": list(detect_thresholds),
        "fp_rates": fp_rates,
        "tp_rates": tp_rates,
        "precisions": precisions,
        "recalls": recalls
    }


def get_algorithm_roc_logs(
        algorithms: List[Dict],
        max_distance: float = 1.0,
        min_snr: float = 1.75,
        inner_radius: float = 2.5,
        outer_radius: float = 10.0,
        max_detects: int = 1,
):
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
            desc=algorithm["name"]
        )

    return logs
