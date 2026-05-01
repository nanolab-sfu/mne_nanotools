#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:21:38 2026

@author: isaant
"""

import os
import mne
import sys
sys.path.append(os.path.expanduser("~/python_modules"))
from nanotools import preprocessing, postprocessing, io_handlers
# ----------------------------
# EDIT THESE
# ----------------------------
root_dir = "/Volumes/A_Flores/temp_san/some_PREVENT-AD"
subject_id = "sub-CONP0008" 
suffix = "ses-01_run-2"
fs_subject = subject_id+"_"+suffix if suffix else subject_id# /MRI/freesurfer/sub-XX_ses-XX_run-X
session = "ses-02"          # your MEG session
subjects_dir_name = "MRI/freesurfer"  # relative inside root_dir (or absolute)
system = "CTF"              # "CTF" or "MEGIN"


# ----------------------------
# DERIVED PATHS
# ----------------------------
fs_dir = os.path.join(root_dir, subjects_dir_name)
meg_dir = os.path.join(root_dir, "MEG", subject_id, session,'meg')

# Update these if your filenames differ
trans_file = os.path.join(meg_dir,f"{subject_id}_{session}-corr_trans.fif")
fids_file  = os.path.join(f"/Volumes/A_Flores/temp_san/some_PREVENT-AD/MRI/freesurfer/{fs_subject}/bem/sub-CONP0008-fiducials.fif")

# Pick a file that exists for your system
# CTF: a .ds folder ; MEGIN: a .fif file
if system.upper() == "CTF":
    # choose any ds folder inside meg_dir (adjust pattern if needed)
    candidates = [os.path.join(meg_dir, x) for x in os.listdir(meg_dir) if x.endswith(".ds")]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .ds found in {meg_dir}")
    raw_path = candidates[0]
else:
    candidates = [os.path.join(meg_dir, x) for x in os.listdir(meg_dir) if x.endswith(".fif")]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .fif found in {meg_dir}")
    raw_path = candidates[0]

print("Using raw:", raw_path)
print("Using trans:", trans_file)
print("Using fs_dir:", fs_dir)

# ----------------------------
# LOAD
# ----------------------------
raw = mne.io.read_raw(raw_path, preload=False, verbose="error")
path2hsp=os.path.join(meg_dir,f"{subject_id}_{session}_hsp_ready.fif")
print(path2hsp)
src_raw = mne.io.read_raw_fif(path2hsp, preload=False)
fids, hsp, hpi = io_handlers.extract_dig_points(src_raw.info)
print("→ Adding missing fiducials:", fids.keys(), "HSP:", hsp.shape, "HPI:", hpi.shape)
raw = io_handlers.inject_dig_into_raw(raw, fids=fids, hsp=hsp, hpi=hpi)


# If your dig points (montage) were saved separately, load them and set them.
# Otherwise raw.info['dig'] should already contain them.
if os.path.exists(fids_file):
    fids, coord_frame = mne.io.read_fiducials(fids_file)
    # Not strictly necessary for plot_alignment if raw already has dig,
    # but kept here if you want to inspect.
    print("Loaded fiducials:", len(fids), "coord_frame:", coord_frame)

trans = mne.read_trans(trans_file)

# ----------------------------
# PLOT
# ----------------------------
mne.viz.plot_alignment(
    info=raw.info,
    trans=trans,
    subject=fs_subject,
    subjects_dir=fs_dir,
    dig=True,
    surfaces={"head-dense":.5},
    coord_frame="meg",   # you asked for 'meg'
    mri_fiducials=True,
    verbose=True,
    meg={"helmet": 0.1, "sensors": 0.1},
)

# IMPORTANT for Spyder: keep the figure open
#mne.viz.set_3d_view(fig, azimuth=90, elevation=80, distance=0.6, )