#!/bin/env python

"""
Created on Wed Dec 25 22:23:53 2024

Module name: parcellate_and_connectivity.py
Description: Parcellate MEG time series and PSD using DK and Schaefer200.
             Optionally compute oAEC, wPLI, and PLV for each atlas using
             the corresponding terminal flags.
Author: Santiago Flores <sfloresa@sfu.ca>
"""

#%% Import necessary libraries
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Parcellate MEG estimates with DK and Schaefer200; optionally compute connectivity.")
    p.add_argument("--subject", required=True, help="Subject ID or combined subject/session/run name.")
    p.add_argument("--data-dir", required=True,
                   type=lambda path: os.path.abspath(os.path.expanduser(path)),
                   help="Dataset root containing MEG, MRI, and derivatives.")
    p.add_argument("--session", help="Session label, e.g. 1 or ses-1 (default: ses-1).")
    p.add_argument("--run", help="Run label, e.g. 1 or run-1; inferred from the name if present.")
    p.add_argument("--inv_method", default="beamformer",
                   choices=["MNE", "dSPM", "sLORETA", "beamformer"], help="Inverse method.")
    p.add_argument("-oAEC", "--oAEC", action="store_true", help="Compute oAEC for both atlases.")
    p.add_argument("-wPLI", "--wPLI", action="store_true", help="Compute debiased wPLI for both atlases.")
    p.add_argument("-PLV", "--PLV", action="store_true", help="Compute PLV for both atlases.")
    args = p.parse_args(argv)
    args.subject_raw = args.subject
    # Accept a combined identifier or the MEG filename/path containing it.
    name = Path(args.subject_raw).name
    subject_match = re.match(r"(sub-[^_]+)", name)
    if not subject_match:
        p.error("--subject must start with a sub- identifier.")
    args.subject = subject_match.group(1)
    for entity, prefix in (("session", "ses-"), ("run", "run-")):
        match = re.search(rf"(?:^|_)({prefix}[^_]+)", name)
        inferred = match.group(1) if match else None
        explicit = getattr(args, entity)
        if explicit is not None:
            if not explicit or explicit == prefix:
                p.error(f"--{entity} requires a nonempty label.")
            explicit = explicit if explicit.startswith(prefix) else prefix + explicit
        if explicit and inferred and explicit != inferred:
            p.error(f"--{entity} {explicit} conflicts with {inferred} in the subject name.")
        setattr(args, entity, explicit or inferred)
    args.session = args.session or "ses-1"
    if not args.run:
        p.error("Provide --run or include run-<label> in --subject.")
    if not os.path.isdir(args.data_dir):
        p.error(f"Data directory does not exist: {args.data_dir}")
    return args


def find_stc_path(stc_dir, subject, session, run, inv_method):
    """Find the rest STC without assuming a fixed preprocessing suffix."""
    entities = {subject, session, "task-rest", run}
    matches = [
        path for path in sorted(Path(stc_dir).glob(f"*_{inv_method}_stc-lh.stc"))
        if entities.issubset(path.name.split("_"))
    ]
    if not matches:
        raise FileNotFoundError(
            f"No {inv_method} source estimate for {subject}, {session}, {run} in {stc_dir}. "
            "Run source estimation first; morphing cannot read the MEG FIF directly."
        )
    if len(matches) > 1:
        raise ValueError("Multiple matching source estimates:\n" + "\n".join(map(str, matches)))
    return str(matches[0])


def ensure_annotations(fs_dir, subject, annotation_files):
    """Map missing subject annotations from fsaverage using FreeSurfer."""
    subjects_dir = Path(fs_dir).resolve()
    annot_dir = subjects_dir / "fsaverage" / "label"
    label_dir = subjects_dir / subject / "label"
    missing = [name for name in annotation_files if not (label_dir / name).is_file()]
    if not missing:
        return

    for directory in (subjects_dir, subjects_dir / subject, subjects_dir / "fsaverage"):
        if not directory.is_dir():
            raise FileNotFoundError(f"FreeSurfer directory not found: {directory}")
    for name in missing:
        if not (annot_dir / name).is_file():
            raise FileNotFoundError(f"Reference annotation not found: {annot_dir / name}")

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)
    executable = shutil.which("mri_surf2surf")
    if executable:
        command_prefix = [executable]
    elif env.get("SLURM_JOB_ID"):
        # Modules are shell functions. Pass paths as positional arguments,
        # keeping them out of the shell program itself.
        command_prefix = [
            "bash", "-lc",
            'set -e\n'
            'module load freesurfer/7.4.1\n'
            ': "${EBROOTFREESURFER:?FreeSurfer module did not set EBROOTFREESURFER}"\n'
            'source "$EBROOTFREESURFER/FreeSurferEnv.sh"\n'
            'export SUBJECTS_DIR="$1"\n'
            'shift\n'
            'command -v mri_surf2surf >/dev/null\n'
            'exec mri_surf2surf "$@"',
            "freesurfer-annotation", str(subjects_dir),
        ]
    else:
        raise RuntimeError(
            "mri_surf2surf is unavailable. Activate FreeSurfer before running this script."
        )

    label_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        target = label_dir / name
        print(f"Mapping annotation from {annot_dir / name} to {target}", flush=True)
        subprocess.run(
            command_prefix + [
                "--srcsubject", "fsaverage", "--trgsubject", subject,
                "--hemi", name.split(".", 1)[0],
                "--sval-annot", str(annot_dir / name), "--tval", str(target),
            ],
            env=env, check=True,
        )
        if not target.is_file():
            raise RuntimeError(f"FreeSurfer finished without creating {target}")


args = parse_args()
subject_raw = args.subject_raw
subject = args.subject
data_dir = args.data_dir
session = args.session
inv_method = args.inv_method
run = args.run

#Get tje number of cores from the slurm job submission
num_cpu = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import sys
import mne
import numpy as np
import matplotlib
import nibabel
import picard
import sklearn
import pickle
from scipy.signal import welch
import pandas as pd
import resource
import gc
import faulthandler
import traceback
sys.path.append(os.path.expanduser("~/scratch"))
from mne_nanotools.postprocessing import *
from mne_nanotools.io_handlers import *


faulthandler.enable()
def print_memory(label):
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[MEMORY] {label}: peak RSS = {mem_kb / 1024**2:.2f} GB")

# Set number of cores for parallel processing (must be done before running any linear algebra functions)
#num_cpu = '16'
#os.environ['OMP_NUM_THREADS'] = num_cpu




#%% Directories
fs_dir = os.path.join(data_dir,'MRI','freesurfer') # Path to FreeSurfer subjects
meg_dir = os.path.join(data_dir,'MEG',subject,session,'meg') # Path to MEG data
log_dir = os.path.join(data_dir, 'logfiles')
derivatives_dir = os.path.join(data_dir, 'derivatives', subject, session, inv_method)
output_basename = f"{subject}_{session}_task-rest_{run}_meg"
stc_dir = os.path.join(data_dir, 'derivatives', subject, session, inv_method, 'stc')
stc_path = find_stc_path(stc_dir, subject, session, run, inv_method)

os.makedirs(log_dir, exist_ok=True)


#%% Immutable Variables

#%% List of required files
required_MRIfiles = ["lh.sphere.reg", "lh.pial", "lh.white", "lh.inflated",
                     "rh.sphere.reg", "rh.pial", "rh.white", "rh.inflated"]

required_MEGfiles = [stc_path, stc_path.removesuffix("-lh.stc") + "-rh.stc"]

required_MRIannot = ['lh.Schaefer2018_200Parcels_17Networks_order.annot',
                     'rh.Schaefer2018_200Parcels_17Networks_order.annot',
                     'lh.aparc.annot',
                     'rh.aparc.annot']

# Check if all required files exist
missing_MRIfiles = [ file for file in required_MRIfiles if not os.path.exists(os.path.join(fs_dir, subject, 'surf', file))]
missing_MEGfiles = [file for file in required_MEGfiles if not os.path.isfile(file)]
if not missing_MRIfiles and not missing_MEGfiles:
    ensure_annotations(fs_dir, subject, required_MRIannot)
missing_MRIannot = [ file for file in required_MRIannot if not os.path.exists(os.path.join(fs_dir, subject, 'label', file))]


#%% Immutable Variables

# Define frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (15, 29),
    'g_low': (30, 59),
    'g_high': (60, 90)
}

with open(os.path.join(log_dir,"missing_files_log.txt"), "a") as log_file:  # "a" mode appends to the file

    # Skip subject if files are missing
    if missing_MRIfiles or missing_MRIannot or missing_MEGfiles:
        missing_info = []
        if missing_MRIfiles:
            missing_info.append(f"MRI files: {', '.join(missing_MRIfiles)}")
        if missing_MRIannot:
            missing_info.append(f"MRI annot: {', '.join(missing_MRIannot)}")
        if missing_MEGfiles:
            missing_info.append(f"MEG files: {', '.join(missing_MEGfiles)}")

        # Prepare the message
        message = f"Skipping subject {subject} due to missing files: {', '.join(missing_info)}"

        # Print to console and save to log file
        log_file.write(message + "\n")  # Add a newline for each entry
        raise ValueError(message)


#%% Main steps
try:
    # Load the SourceEstimate for the subject
    raw_stem = Path(stc_path).name.removesuffix(f"_{inv_method}_stc-lh.stc")
    src_path = os.path.join(data_dir, 'derivatives', subject, session, f"{raw_stem}_src.fif")
    stc = mne.read_source_estimate(stc_path)


    stc.subject = subject
    src = mne.read_source_spaces(src_path)
    sfreq = stc.sfreq

    # Compute vertex-wise PSD once and reuse it for both atlases.
    _, _, psd_normalized, band_powers = PSD_per_vertex_parallel(stc, bands)
    stc_psd = mne.SourceEstimate(
        data=psd_normalized, vertices=stc.vertices,
        tmin=0, tstep=0.25, subject=subject,
    )
    del psd_normalized, band_powers

    atlases = {
        "DK": ("aparc", "desikan_killiany_parc"),
        "Schaefer200_17Networks": (
            "Schaefer2018_200Parcels_17Networks_order",
            "schaefer2018_200parcels_17networks_parc",
        ),
    }
    for atlas_name, (parc, atlas_directory) in atlases.items():
        output_dir = os.path.join(derivatives_dir, atlas_directory, run)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Parcellating {atlas_name}", flush=True)
        labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=fs_dir)
        # Exclude non-cortical annotation entries from the parcel matrices.
        labels = [label for label in labels
                  if not any(token in label.name.lower()
                             for token in ("unknown", "medial_wall", "medialwall", "corpuscallosum"))]
        centroids = calculate_parcellation_centroids(labels, src)
        parc_ts_pca = mne.extract_label_time_course(
            stc, labels, src, mode="pca_flip",
        ).astype("float64")
        with open(os.path.join(output_dir, f"parc_ts-pca_{atlas_name}.pkl"), "wb") as f:
            pickle.dump(parc_ts_pca, f)

        parc_psd = mne.extract_label_time_course(
            stc_psd, labels, src, mode="mean",
        ).astype("float64")
        parc_psd_df = pd.DataFrame({
            "parc_psd": [parc_psd], "labels": [labels], "centroids": [centroids],
        })
        with open(os.path.join(output_dir, f"parc_psd_{atlas_name}.pkl"), "wb") as f:
            pickle.dump(parc_psd_df, f)
        del parc_psd, parc_psd_df

        for metric in ("oAEC", "wPLI", "PLV"):
            if not getattr(args, metric):
                continue
            print(f"Computing {metric} for {atlas_name}", flush=True)
            if metric == "oAEC":
                correlations = compute_band_oAEC(parc_ts_pca, sfreq, bands)
            elif metric == "wPLI":
                correlations = compute_band_wPLI(
                    parc_ts_pca, sfreq, bands, n_jobs=num_cpu, debiased=True,
                )
            else:
                correlations = compute_band_PLV(
                    parc_ts=parc_ts_pca, sfreq=sfreq, bands=bands,
                    n_jobs=num_cpu, epoch_duration=10.0,
                )
            connectivity_df = pd.DataFrame({
                "corr_matrix": [correlations], "labels": [labels], "centroids": [centroids],
            })
            filename = f"{metric}_parc_ts-pca_{atlas_name}.pkl"
            with open(os.path.join(output_dir, filename), "wb") as f:
                pickle.dump(connectivity_df, f)
            del correlations, connectivity_df
            gc.collect()
            print_memory(f"after {metric} ({atlas_name})")

        del parc_ts_pca, labels, centroids
        gc.collect()

    del stc_psd
    gc.collect()


except Exception as e:
    print("\n========== PIPELINE FAILED ==========", flush=True)
    print(f"Exception type: {type(e).__name__}", flush=True)
    print(f"Exception: {repr(e)}", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
