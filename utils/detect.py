import numpy as np
import skimage.measure
import csv


def get_detections(denoised, original, threshold=0.3):
    """Gets the detections from a 'denoised' datacube, defined by pixels that
    exceed the threshold value.  Makes use of the skimage.measure.regionprops()
    method to handle detections larger than one pixel in area.

    :param denoised: (np.array) the denoised framestack
    :param original: (np.array) the original, background suppressed framestack
    :param threshold: (float) threshold for detections
    :return: Dictionary of detections. {frame_num: {'rows': [..], 'cols': [..]}}
    """
    detections = {}
    for i, (den, org) in enumerate(zip(denoised, original)):
        predetects = np.array(den >= threshold).astype("int")
        unfiltered_detects = skimage.measure.regionprops(
            skimage.measure.label(predetects), intensity_image=org
        )

        detects = {"rows": [], "cols": []}
        for detect in unfiltered_detects:
            position = detect.weighted_centroid
            if position[0] < 0 or position[0] >= original.shape[-2]:
                continue
            if position[1] < 0 or position[1] >= original.shape[-1]:
                continue
            detects["rows"].append(position[0])
            detects["cols"].append(position[1])

        detections[i] = detects

    return detections


def export_detections(detections, path):
    """Export the detections from a processed datacube.  The detection number,
    frame, row, and column are exported to a .csv file.

    :param detections: (dict) Dictionary of detections, from save_detections()
    :param path: (str) path to the output .csv file
    :return: None
    """
    i = 0
    fp = open(path, "w")
    writer = csv.writer(fp, lineterminator="\n")

    for f, coords in detections.items():
        for x, y in zip(coords["rows"], coords["cols"]):
            writer.writerow([i, f, x, y])
            i += 1

    fp.close()
