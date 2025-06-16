#!/usr/bin/env bash
# Pre-compute DTI metrics & bundle statistics with PEAKS-based TractSeg

set -euo pipefail
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MRTRIX_TMPFILE_DIR=/tmp

metric_list="fa md ad rd"
stat_list="mean p50 iqr p95"

dwi_mha_files=$(find /input/images/dwi-4d-brain-mri -name "*.mha")

for dwi_mha in $dwi_mha_files; do
    # Set up file names
    json_file="/input/dwi-4d-acquisition-metadata.json"
    subj=$(basename "${dwi_mha%.*}")

    bval=/tmp/${subj}.bval
    bvec=/tmp/${subj}.bvec
    nifti=/tmp/${subj}.nii.gz

    out_dir=/tmp/${subj}
    mkdir -p "$out_dir"

    echo "Converting $dwi_mha to $nifti..."
    python convert_mha_to_nifti.py "$dwi_mha" "$nifti"

    echo "Converting $json_file to $bval and $bvec..."
    python convert_json_to_bvalbvec.py "$json_file" "$bval" "$bvec"

    dwi2mask "$nifti" "$out_dir/mask.nii.gz" \
        -fslgrad "$bvec" "$bval" -nthreads $OMP_NUM_THREADS

    dwi2tensor "$nifti" "$out_dir/tensor.nii.gz" \
        -mask "$out_dir/mask.nii.gz" \
        -fslgrad "$bvec" "$bval" -nthreads $OMP_NUM_THREADS

    scil_dti_metrics.py \
        --not_all --mask "$out_dir/mask.nii.gz" \
        --fa "$out_dir/fa.nii.gz" \
        --md "$out_dir/md.nii.gz" \
        --ad "$out_dir/ad.nii.gz" \
        --rd "$out_dir/rd.nii.gz" \
        "$nifti" "$bval" "$bvec" -f

    echo "Estimating FOD & peaks (single-shell)…"
    dwi2response tournier "$nifti" "$out_dir/response.txt" \
        -mask "$out_dir/mask.nii.gz" \
        -fslgrad "$bvec" "$bval" -nthreads $OMP_NUM_THREADS

    dwi2fod csd "$nifti" "$out_dir/response.txt" "$out_dir/fod.nii.gz" \
        -mask "$out_dir/mask.nii.gz" \
        -fslgrad "$bvec" "$bval" -nthreads $OMP_NUM_THREADS

    sh2peaks "$out_dir/fod.nii.gz" "$out_dir/peaks.nii.gz" \
        -mask "$out_dir/mask.nii.gz" -nthreads $OMP_NUM_THREADS

    echo "Running TractSeg on peaks..."
    TractSeg -i "$out_dir/peaks.nii.gz" \
        -o "$out_dir" \
        --bvals "$bval" \
        --bvecs "$bvec" \
        --brain_mask "$out_dir/mask.nii.gz" \
        --keep_intermediate_files \
        --output_type tract_segmentation \
        --nr_cpus $OMP_NUM_THREADS

    python stats_to_vector.py \
        "$out_dir" \
        "$out_dir/bundle_segmentations" \
        "fa,md,ad,rd" \
        "model/scaler_pca.joblib" \
        "model/pca_model.joblib" \
        "/output/features-128.json"

    # stats_csv="/output/${subj}_features.csv"
    # python extract_stats.py \
    #     "$out_dir" \
    #     "$out_dir/bundle_segmentations" \
    #     "fa,md,ad,rd" \
    #     "$stats_csv"
    # echo "Saved full descriptive stats -> $stats_csv"

    rm -rf "$out_dir" "$bval" "$bvec" "$nifti"
done
