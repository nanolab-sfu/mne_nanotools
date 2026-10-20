#!/bin/env python

"""
Created on Wed Dec 25 22:23:53 2024

Module name: morphing2fsaverage.py
Description: Morph MEG source estimates to the fsaverage cortical surface.
             Save morphed time-series and normalized power spectral density
             (PSD) estimates as left- and right-hemisphere STC files, along
             with morphed frequency-band powers as a NumPy array.
             Accept subject, session, and run as separate terminal arguments
             or extract session and run from a combined subject identifier.
Author: Santiago Flores <sfloresa@sfu.ca>
"""

#%% Import necessary libraries
import argparse
import os
import re
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Morph a subject's MEG source estimates to fsaverage.")
    p.add_argument("--subject", required=True, help="Subject ID or combined subject/session/run name.")
    p.add_argument("--data-dir", required=True,
                   type=lambda path: os.path.abspath(os.path.expanduser(path)),
                   help="Dataset root containing MEG, MRI, and derivatives.")
    p.add_argument("--session", help="Session label, e.g. 1 or ses-1 (default: ses-1).")
    p.add_argument("--run", help="Run label, e.g. 1 or run-1; inferred from the name if present.")
    p.add_argument("--inv_method", default="beamformer",
                   choices=["MNE", "dSPM", "sLORETA", "beamformer"], help="Inverse method.")
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
output_dir = os.path.join(data_dir, 'derivatives', subject, session, inv_method, 'fsaverage-morphed_stc')
output_dir = os.path.join(output_dir, run)
output_basename = f"{subject}_{session}_task-rest_{run}_meg"
stc_dir = os.path.join(data_dir, 'derivatives', subject, session, inv_method, 'stc')
stc_path = find_stc_path(stc_dir, subject, session, run, inv_method)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


#%% Immutable Variables

#%% List of required files
required_MRIfiles = ["lh.sphere.reg", "lh.pial", "lh.white", "lh.inflated",
                     "rh.sphere.reg", "rh.pial", "rh.white", "rh.inflated"]

required_MEGfiles = [stc_path, stc_path.removesuffix("-lh.stc") + "-rh.stc"]

# Check if all required files exist
missing_MRIfiles = [ file for file in required_MRIfiles if not os.path.exists(os.path.join(fs_dir, subject, 'surf', file))]
missing_MEGfiles = [file for file in required_MEGfiles if not os.path.isfile(file)]

# Open a file to save the logs
with open(os.path.join(log_dir,"missing_files_log.txt"), "a") as log_file:  # "a" mode appends to the file

    # Skip subject if files are missing
    if missing_MRIfiles or missing_MEGfiles:
        missing_info = []
        if missing_MRIfiles:
            missing_info.append(f"MRI files: {', '.join(missing_MRIfiles)}")
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
    stc = mne.read_source_estimate(stc_path)
    stc.subject = subject
    print_memory("after loading STC")
    # Compute the morph to fsaverage
    morph = mne.compute_source_morph(stc,subject_from=subject,subject_to='fsaverage',subjects_dir=fs_dir)

    # Save the morphed time series; MNE appends -lh.stc and -rh.stc.
    stc_morph = morph.apply(stc)
    stc_morph.save(os.path.join(output_dir, f"{output_basename}_stc_fsaverage"),ftype="stc",overwrite=True)
    del stc_morph
    gc.collect()

    # Compute normalized PSD and band powers
    _, _, psd_normalized, band_powers = PSD_per_vertex_parallel(stc,bands)
    print_memory("after PSD_per_vertex")

    # Initialize storage for morphed band powers
    band_powers_morph = None

    # Loop through each frequency band
    for i, (band, power) in enumerate(band_powers.items()):
        # Morph the band power to fsaverage
        stc_band_morph = stc_per_band(morph, power, stc, subject)

        # Initialize band power matrix on the first iteration
        if band_powers_morph is None:
            lh_vertices, rh_vertices = stc_band_morph.vertices
            total_vertices = len(lh_vertices) + len(rh_vertices)
            band_powers_morph = np.zeros((total_vertices, len(band_powers)))

        # Store the morphed data for the current band
        band_powers_morph[:, i] = stc_band_morph.data.flatten()

    np.save(os.path.join(output_dir, "bands_power.npy"), band_powers_morph)

    # No longer needed
    del stc_band_morph
    del band_powers_morph
    del band_powers
    gc.collect()

    # Create and save spectrally resolved stc
    stc_psd_morph = stc_per_band(morph, psd_normalized, stc, subject)
    stc_psd_morph.tmin = 0
    stc_psd_morph.tstep = .25
    stc_psd_morph.save(os.path.join(output_dir, f"{output_basename}_psd_fsaverage"),ftype="stc",overwrite=True)
    del stc_psd_morph
    gc.collect()



except Exception as e:
    print("\n========== PIPELINE FAILED ==========", flush=True)
    print(f"Exception type: {type(e).__name__}", flush=True)
    print(f"Exception: {repr(e)}", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
