#!/usr/bin/env python3
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.stats import skew, kurtosis
import csv, json, joblib, sys, os


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
    print("[DEBUG] Current working dir  :", os.getcwd())
    print("[DEBUG] scaler_pkl           :", os.path.abspath(scaler_path))

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    print(f"ok")


if __name__ == "__main__":
    main()
