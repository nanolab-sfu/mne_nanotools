#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch the MNE coregistration GUI to manually create or adjust the MEG-MRI transform (.trans.fif).

Version 0.1.1 - Last modified 21/01/2026

Usage example:
    python corregistration.py \
        --root_dir /Users/isaant/Documents/PosDoc/Projects/BRHRS \
        --subject_id sub-BRS0035 \
        --session 251016 \
        --subjects_dir MRI/freesurfer
"""

import os
import argparse
import mne
import sys
import numpy as np
from mne.bem import make_scalp_surfaces, make_watershed_bem

sys.path.append(os.path.expanduser("~"))
from nanotools import preprocessing, postprocessing

def open_coregistration_gui(root_dir, subject_id, subjects_dir_name="MRI/freesurfer", session=None, compute_bem_if_missing = True, system="MEGIN"):
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
    session : str
        Specified only if there is more than one sessions
    system : str
        MEG system ("MEGIN" uses .fif and allows TSSS; "CTF" uses .ds and no TSSS).
    """
    

    root_dir = os.path.abspath(root_dir)
    fs_dir = os.path.join(root_dir, subjects_dir_name)
    system_upper = system.upper()
    if system_upper not in {"MEGIN", "CTF"}:
        raise ValueError("system must be 'MEGIN' or 'CTF'")
    meg_ext = ".fif" if system_upper == "MEGIN" else ".ds"
    if session is None:
        meg_dir = os.path.join(root_dir, "MEG", subject_id)
    else:
        meg_dir = os.path.join(root_dir, "MEG", subject_id, session)

    meg_nested = os.path.join(meg_dir, "meg")
    meg_dir = meg_nested if os.path.isdir(meg_nested) else meg_dir

    meg_files = [
        f for f in os.listdir(meg_dir)
        if f.endswith(meg_ext) and ("raw" in f or "meg" in f)
    ]
    trans_found = [f for f in os.listdir(meg_dir) if f.endswith("-corr_trans.fif")]
    bem_path = os.path.join(fs_dir, subject_id, "bem", f"{subject_id}-5120-5120-5120-bem-sol.fif")
    bem_dir = os.path.join(fs_dir,subject_id, "bem")
    
    if trans_found:
        print(f"Found existing file: {trans_found[0]}. Skipping... \n")
        sys.exit()
    
    if compute_bem_if_missing and not os.path.exists(bem_path):
        print("→ Creating watershed BEM (if missing)...")
        try:   
            bem_surfaces_path = os.path.join(fs_dir, subject_id, "bem", "outer_skin.surf")
            if not os.path.exists(bem_surfaces_path):
                print ('→ Making BEM')
                make_watershed_bem(subject=subject_id, subjects_dir=fs_dir, overwrite=True) #if not found, create BEM surfaces using the FreeSurfer watershed algorithm (T1w images)
                make_scalp_surfaces(subject=subject_id, subjects_dir=fs_dir, overwrite=True) #Creates the high resolution -head-dense.fif
        except Exception as e:
            print(f"⚠️ BEM failed: {e}")

        try:    
            os.makedirs(bem_dir, exist_ok=True)
            conductivity = (0.3,)   # Single layer for MEG
            model = mne.make_bem_model(subject=subject_id, ico=4, #The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280. If None, no subsampling is applied.
                                conductivity=conductivity, 
                                subjects_dir=fs_dir) #bem conductivity model
            bem_sol = mne.make_bem_solution(model)
            mne.write_bem_solution(bem_path, bem_sol)
            
       
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
    
    if session is None:
        print("\n✅ When you finish aligning in the GUI, click “Save” to write:")
        print(f"   {meg_dir}/{subject_id}-corr_trans.fif")
        print("Then simply close the GUI to end this script.")
    else:
        print("\n✅ When you finish aligning in the GUI, click “Save” to write:")
        print(f"   {meg_dir}/{subject_id}-{session}-corr_trans.fif")
        print("Then simply close the GUI to end this script.")
        
    if system_upper == "CTF":
        raw = preprocessing.read_data(raw_path)
        pos_file = os.path.join(meg_dir, f"{subject_id}_{session}_headshape.pos")

        # ----  Read CTF .pos (skip first line, ignore index column) ----
        pts = np.loadtxt(pos_file, skiprows=1, usecols=(1, 2, 3))  # shape (N, 3)

        # ----  Unit heuristic: CTF .pos is commonly in cm ----
        mx = np.max(np.abs(pts))
        if mx > 1:          # e.g., values around 8–10 -> cm
            pts = pts / 100.0

        pts[:,1]*=-1 #based on https://www.fieldtriptoolbox.org/faq/source/coordsys/ Neruomeg VS CTF
        pts = pts[:, [1, 0, 2]]
        pts_dig=pts[:-12,:]
        pts_fids=pts[-12:-9,:]
            
        print("Loaded headshape points:", pts.shape[0], "points. Max abs (m):", np.max(np.abs(pts)))

        # ----  Build a DigMontage containing BOTH fiducials + headshape ----
        montage = mne.channels.make_dig_montage(
            nasion=pts_fids[0],
            lpa=pts_fids[1],
            rpa=pts_fids[2],
            hsp=pts_dig,
            coord_frame="head",
        )

        # ----  Inject into raw (this will replace dig with the montage's dig) ----
        raw.set_montage(montage, on_missing="ignore")

        print("n_dig after:", len(raw.info["dig"]))


        out_fif = os.path.join(meg_dir, f"{subject_id}_{session}_hsp_ready.fif")
        raw.save(out_fif, overwrite=True)
        raw_path = out_fif
        
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

   

def _parse_args():
    p = argparse.ArgumentParser(description="Open MNE coregistration GUI for a subject.")
    p.add_argument("--root_dir", required=True, help="Root directory with MEG/ and MRI/ folders")
    p.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-BRS0035)")
    p.add_argument("--subjects_dir_name", default="MRI/freesurfer",
                   help="Relative path to the FreeSurfer subjects dir inside root_dir")
    p.add_argument("--session", default=None,
                   help="Session by date or order (e.g., 01012020 or ses-1)")
    p.add_argument("--compute_bem_if_missing", action="store_true", default=True)
    p.add_argument("--system", default="MEGIN", choices=["MEGIN", "CTF"],
                   help="MEG system; sets expected raw extension (.fif for MEGIN, .ds for CTF) and TSSS handling.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    open_coregistration_gui(args.root_dir, args.subject_id, args.subjects_dir_name, args.session, args.compute_bem_if_missing, args.system)
