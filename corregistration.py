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

def open_coregistration_gui(root_dir, subject_id, subjects_dir_name="MRI/freesurfer"):
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
    subjects_dir = os.path.join(root_dir, subjects_dir_name)
    meg_dir = os.path.join(root_dir, "MEG", subject_id)
    meg_files = [f for f in os.listdir(meg_dir) if f.endswith(".fif") and "raw" in f]

    print(f"\n🧠 Launching MNE coregistration for {subject_id}")
    print(f"Subjects dir: {subjects_dir}")
    if meg_files:
        raw_path = os.path.join(meg_dir, meg_files[0])
        print(f"→ Using {raw_path} for head shape / fiducials.")
    else:
        raw_path = None
        print("⚠️ No MEG raw file found — GUI will open without head points.")

  # Launch GUI
    mne.gui.coregistration(
        subject=subject_id,
        subjects_dir=subjects_dir,
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
    p.add_argument("--subjects_dir", default="MRI/freesurfer",
                   help="Relative path to the FreeSurfer subjects dir inside root_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    open_coregistration_gui(args.root_dir, args.subject_id, args.subjects_dir)