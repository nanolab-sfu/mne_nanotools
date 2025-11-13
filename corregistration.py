#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch the MNE coregistration GUI to manually create or adjust the MEG-MRI transform (.trans.fif).

Version 0.1.0 - Last modified 11/11/2025

Usage example:
    python generate_trans.py \
        --root_dir /Users/isaant/Documents/PosDoc/Projects/BRHRS \
        --subject_id sub-BRS0035 \
        --subjects_dir MRI/freesurfer
"""

import os
import argparse
import mne

def open_coregistration_gui(root_dir, subject_id, subjects_dir_name="MRI/freesurfer", compute_bem_if_missing: bool = True,):
    """
    Open the MNE coregistration GUI for a given subject.

    Parameters
    ----------
    root_dir : str
        Root directory containing MEG/ and MRI/ folders.
    subject_id : str
        Subject identifier (e.g., 'sub-BRS0035').
    subjects_dir_name : str
        Relative path to the FreeSurfer subjects directory inside root_dir.
    """
    root_dir = os.path.abspath(root_dir)
    fs_dir = os.path.join(root_dir, subjects_dir_name)
    meg_dir = os.path.join(root_dir, "MEG", subject_id)
    meg_files = [f for f in os.listdir(meg_dir) if f.endswith(".fif") and "raw" in f]
    trans_path = os.path.join(meg_dir, f"{subject_id}-trans_corr.fif")
    bem_path = os.path.join(fs_dir, subject_id, "bem", f"{subject_id}-5120-5120-5120-bem-sol.fif")
    bem_dir = os.path.join(fs_dir, "bem")

    if compute_bem_if_missing and not os.path.exists(bem_path):
        os.makedirs(bem_dir, exist_ok=True)
        conductivity = (0.3,)   # Single layer for MEG
        model = mne.make_bem_model(subject=subject_id, ico=4, #The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280. If None, no subsampling is applied.
                            conductivity=conductivity, 
                            subjects_dir=fs_dir) #bem conductivity model
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_path, bem_sol)
        print("→ Creating watershed BEM (if missing)...")
        try:
            mne.bem.make_watershed_bem(subject=subject_id, subjects_dir=fs_dir, overwrite=True)
            mne.bem.make_scalp_surfaces(subject=subject_id, subjects_dir=fs_dir, overwrite=True) #Creates the high resolution -head-dense.fif

        except Exception as e:
            print(f"⚠️ Watershed BEM failed: {e}")



    print(f"\n🧠 Launching MNE coregistration for {subject_id}")
    print(f"Subjects dir: {fs_dir}")
    if meg_files:
        raw_path = os.path.join(meg_dir, meg_files[0])
        print(f"→ Using {raw_path} for head shape / fiducials.")
    else:
        raw_path = None
        print("⚠️ No MEG raw file found — GUI will open without head points.")

  # Launch GUI
    mne.gui.coregistration(
        subject=subject_id,
        subjects_dir=fs_dir,
        inst=raw_path,
        fullscreen=True,
        head_high_res=True,
        show=True,
        block=True,
    )

    print("\n✅ When you finish aligning in the GUI, click “Save” to write:")
    print(f"   {meg_dir}/{subject_id}-corr_trans.fif")
    print("Then simply close the GUI to end this script.")

def _parse_args():
    p = argparse.ArgumentParser(description="Open MNE coregistration GUI for a subject.")
    p.add_argument("--root_dir", required=True, help="Root directory with MEG/ and MRI/ folders")
    p.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-BRS0035)")
    p.add_argument("--subjects_dir_name", default="MRI/freesurfer",
                   help="Relative path to the FreeSurfer subjects dir inside root_dir")
    p.add_argument("--compute_bem_if_missing", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    open_coregistration_gui(args.root_dir, args.subject_id, args.subjects_dir_name, args.compute_bem_if_missing)