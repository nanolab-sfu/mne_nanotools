#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch the MNE coregistration GUI to manually create or adjust the MEG-MRI transform (.trans.fif).

Version 1.1.0 - Last modified 04/02/2026

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
import nibabel as nib
import traceback
from mne.bem import make_scalp_surfaces, make_watershed_bem
from mne.io.constants import FIFF
from datetime import datetime


sys.path.append(os.path.expanduser("~"))
from mne_nanotools import preprocessing, io_handlers

def open_coregistration_gui(root_dir, 
                            subject_id,
                            subjects_dir_name,
                            session,
                            suffix,
                            compute_bem_if_missing,
                            system,
                            mri_fiducials,
                            json,
                            automated,
                            overwrite):
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
    suffix : str
        Specified only if there a suffix in the freesurfer naming convention
    system : str
        MEG system ("MEGIN" uses .fif and allows TSSS; "CTF" uses .ds and no TSSS).
    mri_fiducials: dictionary,
        Dictionary with the landmark coordinates in the freesurfer (tkrRAS(mm)) mri space.
    json : bool
        If True, will look for _coordsystem.json and _T1w.json for mri (world (mm)) and head landmark coordinates. Default False
    automated: bool
        If True, coregistration refinement will be done automatically (-fiducials.fif must exist). 
    """
    

    root_dir = os.path.abspath(root_dir)
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # initialize log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"corregistration_{subject_id}_{timestamp}.txt")
    
    fs_dir = os.path.join(root_dir, subjects_dir_name)
    system_upper = system.upper()
    
    if automated == True:
        if mri_fiducials==None and json==False:
            print("\n⚠️ Automatic corregistration cannot be done, MRI fuducials not provided.")
            print("\n⚠️ Coregistration terminated.")
            sys.exit()
    
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
    if (
        f.endswith(meg_ext)
        and ("raw" in f or "meg" in f)
        and "noise" not in f.lower()
        and "erm" not in f.lower()
    )
    ]
    
    trans_found = [f for f in os.listdir(meg_dir) if f.endswith("-corr_trans.fif")]
    if trans_found and overwrite==False:
        print(f"Found existing file: {trans_found[0]}. Skipping... \n")
        sys.exit()

    head_fid = None
    if json:
        try:
            head_coordinate_file = os.path.join(meg_dir, f"{subject_id}_{session}_coordsystem.json")
            head_coordinates = io_handlers.load_json(head_coordinate_file)
            head_fid = head_coordinates['AnatomicalLandmarkCoordinates']
            intended = io_handlers.strip_bids_prefix(head_coordinates["IntendedFor"])
            mri_basename = io_handlers.strip_nii_suffix(os.path.basename(intended))
            fs_subject = io_handlers.extract_bids_id(mri_basename)
            while not os.path.isdir(os.path.join(fs_dir, fs_subject)) and re.search(r'_', fs_subject): #will look for the file that matches inside the freesufer output folder (no suffix).
                fs_subject = fs_subject.rsplit('_', 1)[0]
            bem_path = os.path.join(fs_dir, fs_subject, "bem", f"{fs_subject}-5120-5120-5120-bem-sol.fif")
            bem_dir = os.path.join(fs_dir, fs_subject, "bem")
            fiducials_file = os.path.join(bem_dir,f'{fs_subject}-fiducials.fif')
            ses = io_handlers.extract_bids_session(intended, session)
    
            anat_folder = os.path.join(root_dir, "MEG", subject_id, ses,"anat")
            t1_nii = os.path.join(anat_folder, f"{mri_basename}.nii.gz")
            t1_json = os.path.join(anat_folder, f"{mri_basename}.json")
    
            lm = io_handlers.load_json(t1_json)["AnatomicalLandmarkCoordinates"]
    
            img = nib.load(t1_nii)
            scannerRAS_to_tkrRAS = io_handlers.scannerRAS_to_tkrRAS_matrix(os.path.join(fs_dir, fs_subject, "mri", "T1.mgz"))
    
            pts = []
            for fid in ("NAS", "LPA", "RPA"):
                ident = io_handlers.fiff_ident_from_label(fid)
                if ident is None or fid not in lm:
                    continue
    
                ijk = np.array(lm[fid], float)
                ras_tkr_mm = io_handlers.nifti_ijk_to_tkrRAS(ijk, img.affine, scannerRAS_to_tkrRAS)/1000
    
                pts.append(dict(r=ras_tkr_mm, ident=ident,kind=FIFF.FIFFV_POINT_CARDINAL,))
                
            mne.io.write_fiducials(fiducials_file, pts, coord_frame="mri", overwrite=True)
    
        except Exception as e:
            print(f"⚠️ Fiducial construction failed: {e} \n")
            # ---- Write error to txt ----
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(log_file, "w") as f:
                f.write("⚠️ Fiducial construction failed\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
             
            sys.exit(1)
    
    else:
        fs_subject = f"{subject_id}_{suffix}" if suffix else subject_id
        bem_path = os.path.join(fs_dir, fs_subject, "bem", f"{fs_subject}-5120-5120-5120-bem-sol.fif")
        bem_dir = os.path.join(fs_dir, fs_subject, "bem")
        fiducials_file = os.path.join(bem_dir,f'{fs_subject}-fiducials.fif')

    
    if compute_bem_if_missing and not os.path.exists(bem_path):
        print("→ Creating watershed BEM (if missing)...")
        try:   
            bem_surfaces_path = os.path.join(fs_dir, fs_subject, "bem", "outer_skin.surf")
            if not os.path.exists(bem_surfaces_path):
                print ('→ Making BEM')
                make_watershed_bem(subject=fs_subject, subjects_dir=fs_dir, overwrite=True) #if not found, create BEM surfaces using the FreeSurfer watershed algorithm (T1w images)
                make_scalp_surfaces(subject=fs_subject, subjects_dir=fs_dir, overwrite=True) #Creates the high resolution -head-dense.fif
        except Exception as e:
            print(f"⚠️ BEM failed: {e} \n")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(log_file, "w") as f:
                print(f"⚠️ BEM failed: \n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
             
            print(f"Error log written to: {log_file}")
            

        try:    
            os.makedirs(bem_dir, exist_ok=True)
            conductivity = (0.3,)   # Single layer for MEG
            model = mne.make_bem_model(subject=fs_subject, ico=4, #The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280. If None, no subsampling is applied.
                                conductivity=conductivity, 
                                subjects_dir=fs_dir) #bem conductivity model
            bem_sol = mne.make_bem_solution(model)
            mne.write_bem_solution(bem_path, bem_sol)
            
       
        except Exception as e:
            print(f"⚠️ Watershed BEM failed: {e} \n")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(log_file, "w") as f:
                print(f"⚠️ Watershed BEM failed: \n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
             
            print(f"Error log written to: {log_file}")
    
    


    print(f"\n🧠 Launching MNE coregistration for {subject_id}")
    print(f"Subjects dir: {fs_dir}")
    if meg_files:
        raw_path = os.path.join(meg_dir, meg_files[0])
        print(f"→ Using {raw_path} for head shape / fiducials.")
    else:
        raw_path = None
        print("⚠️ No MEG raw file found — GUI will open without head points.")
    
    if automated == False:
        if session is None:
            print("\n✅ When you finish aligning in the GUI, click “Save” to write:")
            print(f"   {meg_dir}/{subject_id}-corr_trans.fif")
            print("Then simply close the GUI to end this script.")
        else:
            print("\n✅ When you finish aligning in the GUI, click “Save” to write:")
            print(f"   {meg_dir}/{subject_id}-{session}-corr_trans.fif")
            print("Then simply close the GUI to end this script.")
    else: 
        if mri_fiducials or json:
            print("\n✅ Coregistration will be done automatically using the provided landmarks.")
    
        
    if mri_fiducials: 
        pts = []
        for fid, coord in mri_fiducials.items():
            if fid.upper() == "LPA":
                ident = FIFF.FIFFV_POINT_LPA
            elif fid.upper() in ("NAS", "NASION"):
                ident = FIFF.FIFFV_POINT_NASION
            elif fid.upper() == "RPA":
                ident = FIFF.FIFFV_POINT_RPA
            else:
                continue
        
            pts.append(
                dict(
                    r=np.array(coord, dtype=float),
                    ident=ident,
                    kind=FIFF.FIFFV_POINT_CARDINAL,
                )
            )
            
        coord_frame = 'mri'
        mne.io.write_fiducials(fiducials_file , pts, coord_frame=coord_frame, overwrite=True)

        print(f"\n→ Fiducials written to {fiducials_file}")

    # ===============================================================================================

    # The suffix parameter overlaps a lot with the extracted information in this routine when
    # the --json == True. Optimize?

    # ===============================================================================================
    
    
        
    if system_upper == "CTF":
        print("\n✅ Reading landmarks and headpoints.")

        raw = preprocessing.read_data(raw_path)

        pos_file = os.path.join(meg_dir, f"{subject_id}_{session}_headshape.pos")
        
        # ---- Read CTF .pos (skip first line, ignore index column) ----
        pts_ctf = np.loadtxt(pos_file, skiprows=1, usecols=(1, 2, 3))  # (N, 3)
        
        # ---- Prefer fiducials from JSON only if all three are present ----
        has_json_fids = (
            isinstance(head_fid, dict)
            and all(k in head_fid for k in ("NAS", "LPA", "RPA"))
        )
        
        if has_json_fids:
            # Convention: last 12 points → extra points, last 3 of those are fiducials
            print(f"\n✅Using head landmarks in {head_fid}.")
            pts_ctf[-12:-9, :] = np.vstack([
                np.asarray(head_fid["NAS"], float),
                np.asarray(head_fid["LPA"], float),
                np.asarray(head_fid["RPA"], float),
            ])
        
        # ---- Unit heuristic: CTF .pos commonly in cm ----
        
        print("\n✅ Transforming CTF head coords -> Neuromag head coords.")
        
        mx = np.max(np.abs(pts_ctf))
        if mx > 1:  # e.g., values around 8–10 -> cm
            pts_ctf = pts_ctf / 100.0  # cm -> m
        
        # ---- CTF head coords -> Neuromag head coords ----
        # (FieldTrip coordsys faq: Neuromag vs CTF)
        T = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        
        trans = mne.transforms.Transform(fro="ctf_head", to="head", trans=T)
        
        pts_head = mne.transforms.apply_trans(trans, pts_ctf)
        
        # Convention in your file: last 12 are extra points; last 3 of those are fiducials
        pts_dig  = pts_head[:-12, :]
        pts_fids = pts_head[-12:-9, :]
        
        print("\n✅Loaded headshape points:", pts_head.shape[0],
              "points. Max abs (m):", np.max(np.abs(pts_head)))
        
        

        montage = mne.channels.make_dig_montage(
            nasion=pts_fids[0],
            lpa=pts_fids[1],
            rpa=pts_fids[2],
            hsp=pts_dig,
            coord_frame="head",)
            
        # ----  Inject into raw (this will replace dig with the montage's dig) ----
        
        print("\n✅ Injecting landmark into raw structure.")
        raw.set_montage(montage, on_missing="ignore")

        print("n_dig after:", len(raw.info["dig"]))


        out_fif = os.path.join(meg_dir, f"{subject_id}_{session}_hsp_ready.fif")
        raw.save(out_fif, overwrite=True)
        raw_path = out_fif
    
    if automated and os.path.exists(fiducials_file):
        # Launch automated coregistration
        info = mne.io.read_info(raw_path)
        coreg = mne.coreg.Coregistration(info = info,
                                         subject=fs_subject,
                                         subjects_dir=fs_dir,
                                         fiducials='auto',
                                         on_defects='raise')
        coreg.fit_fiducials(verbose=True)
        mne.write_trans(f"{meg_dir}/{subject_id}_{session}-corr_trans.fif",coreg.trans, overwrite=overwrite)

    else:
        # Launch GUI
        mne.gui.coregistration(subject=fs_subject,
                               subjects_dir=fs_dir,
                               inst=raw_path,
                               fullscreen=True,
                               head_high_res=True,
                               show=True,
                               block=True,)

   

def _parse_args():
    p = argparse.ArgumentParser(description="Open MNE coregistration GUI for a subject.")
    p.add_argument("--root_dir", type=str, required=True, help="Root directory with MEG/ and MRI/ folders")
    p.add_argument("--subject_id", type=str, required=True, help="Subject ID (e.g., sub-BRS0035)")
    p.add_argument("--subjects_dir_name", type=str, default="MRI/freesurfer",
                   help="Relative path to the FreeSurfer subjects dir inside root_dir")
    p.add_argument("--session", type=str, default=None,
                   help="Session by date or order (e.g., 01012020 or ses-1)")
    p.add_argument("--suffix", type=str, default=None,
                   help="Optional FreeSurfer subject suffix (e.g., 'ses-01_run-2' for multiple T1w runs.")
    p.add_argument("--compute_bem_if_missing", action="store_true", default=True)
    p.add_argument("--system", type=str, default="MEGIN", choices=["MEGIN", "CTF"],
                   help="MEG system; sets expected raw extension (.fif for MEGIN, .ds for CTF) and TSSS handling.")
    p.add_argument("--mri_fiducials", type=dict, default=None, help="Dictionary with the landmark coordinates in the freesurfer (tkrRAS(mm)) mri space.")
    p.add_argument("--json", action="store_true", help="Commonly shared with open datasets with dephased MRI;  If True, will look for '_coordsystem.json' and '_T1w.json' for mri (world (mm)) and head landmark coordinates.")
    p.add_argument("--automated", action="store_true", help="Coregistration refinement will be done automatically if True; (-fiducials.fif must exist)"),
    p.add_argument("--overwrite", action="store_true", help="Overwrites corr_trans if it exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    open_coregistration_gui(
        root_dir=args.root_dir,
        subject_id=args.subject_id,
        subjects_dir_name=args.subjects_dir_name,
        session=args.session,
        suffix=args.suffix,
        compute_bem_if_missing=args.compute_bem_if_missing,
        system=args.system,
        mri_fiducials=args.mri_fiducials,
        json=args.json,               # <-- THIS
        automated=args.automated,
        overwrite=args.overwrite
    )