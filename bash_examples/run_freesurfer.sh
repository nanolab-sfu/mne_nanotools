#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
NIFTI_ROOT="/Users/isaant/Documents/PosDoc/Projects/BRHRS/MRI/nifti"
export SUBJECTS_DIR="/Users/isaant/Documents/PosDoc/Projects/BRHRS/MRI/freesurfer"

# Number of cores for recon-all -parallel
NCORES=$(sysctl -n hw.ncpu)   # macOS
# NCORES=$(nproc)             # Linux alternative

mkdir -p "$SUBJECTS_DIR"

# -------- LOOP --------
find "$NIFTI_ROOT" -type f -name "*_T1w.nii.gz" | while read -r T1; do

    # Example path parts:
    # sub-BRS0035/ses-20250206/anat/file.nii.gz
    sub=$(basename "$(dirname "$(dirname "$(dirname "$T1")")")")
    ses=$(basename "$(dirname "$(dirname "$T1")")")

    FS_SUBJECT="${sub}_${ses}"

    echo "========================================"
    echo "Running recon-all for: $FS_SUBJECT"
    echo "T1: $T1"
    echo "========================================"

    recon-all \
        -subject "$FS_SUBJECT" \
        -i "$T1" \
        -all \
        -parallel \
        -openmp "$NCORES" \
        -no-isrunning
    break

done