#!/usr/bin/env python3
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.stats import skew, kurtosis
import csv, json, joblib, sys


def compute_stats(metrics_dir, roi_dir, metric_list):
    """
    Compute multiple descriptive statistics for each metric within each ROI bundle.
    Returns header (list of column names) and features (list of values).
    """

    def percentile(data, q):
        return np.percentile(data, q)

    stat_funcs = [
        ("mean", lambda x: np.mean(x)),
        ("median", lambda x: np.median(x)),
        ("std", lambda x: np.std(x)),
        ("skew", lambda x: skew(x) if x.size > 2 else 0.0),
        ("kurtosis", lambda x: kurtosis(x) if x.size > 3 else 0.0),
        ("p10", lambda x: percentile(x, 10)),
        ("p25", lambda x: percentile(x, 25)),
        ("p75", lambda x: percentile(x, 75)),
        ("p90", lambda x: percentile(x, 90)),
    ]

    metrics = {
        m: nib.load(Path(metrics_dir) / f"{m}.nii.gz").get_fdata() for m in metric_list
    }

    header = []
    features = []

    roi_paths = sorted(Path(roi_dir).glob("*.nii.gz"))
    for roi_path in roi_paths:
        bundle = roi_path.name.removesuffix(".nii.gz")
        mask = nib.load(roi_path).get_fdata() > 0
        for m in metric_list:
            data = metrics[m][mask]
            for stat_name, func in stat_funcs:
                header.append(f"{bundle}_{m}_{stat_name}")
                if data.size:
                    features.append(float(func(data)))
                else:
                    features.append(0.0)
    return header, features


def main():
    if len(sys.argv) != 7:
        print(
            "Usage: extract_descriptive_stats_pca.py <metrics_dir> <roi_dir> <metrics_csv> "
            "<scaler_pkl> <pca_pkl> <output_json>"
        )
        sys.exit(1)

    metrics_dir, roi_dir, metrics_csv, scaler_path, pca_path, output_json = sys.argv[
        1:7
    ]
    metric_list = metrics_csv.split(",")

    _, features = compute_stats(metrics_dir, roi_dir, metric_list)

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    z = pca.transform(scaler.transform(np.asarray(features).reshape(1, -1))).flatten()

    vec128 = z.tolist()
    if len(vec128) < 128:
        vec128.extend([0.0] * (128 - len(vec128)))
    else:
        vec128 = vec128[:128]

    with open(output_json, "w") as f:
        json.dump(vec128, f, indent=4)

    print(f"Saved 128D PCA vector -> {output_json}")


if __name__ == "__main__":
    main()
