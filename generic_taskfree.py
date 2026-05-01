#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic task-free MEGIN preprocessing with ERM-SSP, tSSS, QC report, and bandwise source PSDs.

Version 0.1.3 - Last modified 21/01/2026

Example:
    python generic_taskfree_MEGIN.py \
        --root_dir /PATH/TO/DATASET \
        --subject_id sub-BRS0035 \
        --tsss_dir /PATH/TO/tsss_params/2023 \
        --l_freq 0.5 --h_freq 200 --line_freqs 60 120 180 \
        --downsample 500 --st_duration 10.0
"""

import os
import argparse
from pathlib import Path
import glob
import re
import matplotlib
matplotlib.use("Agg")  # headless mode for servers
import matplotlib.pyplot as plt
import mne
import numpy as np
from importlib import reload
from mne.report import Report
from datetime import datetime
import traceback

# ---- custom user modules ----
import sys
sys.path.append(os.path.expanduser("~"))
from mne_nanotools import preprocessing, postprocessing, io_handlers

# Local file discovery helpers (supports MNE-style + BIDS-style + suffixed variants)
# ----------------------------------------------------------

def parse_ranges(arg):
    """
    Parse string like:
    "42-45,50-53"
    into:
    [(42, 45), (50, 53)]
    """
    ranges = []
    for part in arg.split(","):
        start, end = part.split("-")
        ranges.append((float(start), float(end)))
    return ranges

def find_meg(root_dir: str,
                 subject_id: str,
                 session: str | None = None,
                 task: str | None = None,
                 run: int | str | None = None,
                 prefer: str = "any",
                 system: str = "MEGIN") -> Path:
    """Find a single MEG file for a subject/session/task.

    Supports:
      - MNE-style: sub_NVAR008_rest1_raw*.ext, sub_NVAR008_somatoauditory1_raw_tsss.ext
      - BIDS-style: sub-BRS0034_ses-20241217_task-rest_run-1_meg*.ext
      - BIDS + suffix: ..._meg_digFiltered.ext (or other *_meg_*.ext)

    Notes
    -----
    - Run can be encoded as run-1 or run-01; we try both.
    - Some datasets omit run entirely; we try both with/without run.
    """

    root_in = Path(root_dir).expanduser().resolve()
    if not root_in.exists():
        raise FileNotFoundError(f"root_dir not found: {root_in}")

    system_upper = system.upper()
    if system_upper not in {"MEGIN", "CTF"}:
        raise ValueError("system must be 'MEGIN' or 'CTF'")
    ext = ".fif" if system_upper == "MEGIN" else ".ds"

    # Normalize dataset root vs MEG root
    # - If user passes the dataset root, MEG is usually under root/MEG
    # - If user passes the MEG root directly, keep it
    meg_root = root_in / "MEG" if (root_in / "MEG").exists() else root_in

    # Improved session normalization
    ses_norm = None
    if session:
        # For BIDS: session may be 'ses-20241217' or '20241217'
        if session.startswith("ses-"):
            ses_norm = session
        else:
            # NVAR style often uses numeric folder names like '251017'
            ses_norm = session

    # Constrain search to MEG tree only (avoid derivatives) and to the requested session if provided.
    search_roots: list[Path] = []
    base_roots: list[Path] = []
    if ses_norm is not None:
        # Try both NVAR-style (MEG/sub_X/251017) and BIDS-style (MEG/sub-X/ses-YYYY)
        base_roots.append(meg_root / subject_id / ses_norm)
        if not ses_norm.startswith("ses-"):
            base_roots.append(meg_root / subject_id / f"ses-{ses_norm}")
        else:
            base_roots.append(meg_root / subject_id / ses_norm.replace("ses-", ""))
    else:
        base_roots.append(meg_root / subject_id)

    # Include common nested variant MEG/sub-X/ses-YYYY/meg
    for br in base_roots:
        search_roots.append(br)
        search_roots.append(br / "meg")

    # Deduplicate while preserving order
    search_roots = list(dict.fromkeys(search_roots))

    # Keep only roots that exist
    search_roots = [p for p in search_roots if p.exists()]
    if not search_roots:
        raise FileNotFoundError(
            "No MEG search root found for the provided subject/session.\n"
            f"  meg_root={meg_root}\n  subject_id={subject_id}\n  session={session}\n"
        )

    candidates: list[Path] = []

    # Normalize run so CLI can accept: 1, 01, run-1, run-01
    def _norm_run(r):
        if r is None:
            return None
        if isinstance(r, int):
            return r
        rs = str(r).strip()
        rs = rs.replace("run-", "")
        rs = rs.replace("run_", "")
        rs = rs.lstrip("0") or "0"
        try:
            return int(rs)
        except ValueError:
            raise ValueError(f"Invalid run value: {r}. Use e.g. 1, 01, run-1, run-01")

    run_int = _norm_run(run)

    # --- MNE-style patterns ---
    if task:
        for sr in search_roots:
            candidates.extend(sr.rglob(f"{subject_id}_{task}_raw*{ext}"))

    # --- BIDS-style patterns ---
    if ses_norm and task:
        if run_int is None:
            run_parts = ["", "_run-*", "_run-??"]
        else:
            r = int(run_int)
            run_parts = [f"_run-{r}", f"_run-{r:02d}"]

        for run_part in run_parts:
            patt_bids = f"{subject_id}_{ses_norm}_task-{task}{run_part}_meg*{ext}"
            for sr in search_roots:
                candidates.extend(sr.rglob(patt_bids))

    # De-duplicate and keep files only
    candidates = sorted({c.resolve() for c in candidates if c.is_file() or c.is_dir()})

    # Filter out processed artifacts (we want the true input raw, not cached outputs)
    def _is_valid_input(p: Path) -> bool:
        n = p.name
        bad_tokens = ["_tsss", "_filt", "_proj", "_src", "_stc", "_head_pos", "_QC_report"]
        return not any(tok in n for tok in bad_tokens)

    candidates = [c for c in candidates if _is_valid_input(c)]

    if not candidates:
        raise FileNotFoundError(
            "No MEG file found.\n"
            f"  root_dir={root_in}\n  subject_id={subject_id}\n  session={session}\n  task={task}\n  run={run_int}\n  system={system_upper}\n"
            "Tried MNE-style and BIDS-style patterns.\n"
            "If this is BIDS, check whether run is zero-padded (run-01) or omitted.\n"
        )

    def score(p: Path) -> int:
        name = p.name
        s = 0
        ext_re = re.escape(ext)

        # Strong preference for matching the requested session folder, if provided
        if ses_norm is not None:
            if f"/{ses_norm}/" in p.as_posix():
                s += 100
            # also consider alternate ses- forms
            if ses_norm.startswith("ses-") and f"/{ses_norm.replace('ses-','')}/" in p.as_posix():
                s += 90
            if (not ses_norm.startswith("ses-")) and f"/ses-{ses_norm}/" in p.as_posix():
                s += 90

        # Preference handling:
        #   --prefer any  : no preference
        #   --prefer raw  : prefer files WITHOUT extra processing suffixes like *_meg_<suffix>.fif/.ds
        #   --prefer <token> : prefer files whose filename contains <token> (e.g., digFiltered, channels_removed)
        if prefer and prefer != "any":
            if prefer == "raw":
                # Prefer "clean" BIDS names when possible (exact *_meg.ext) and avoid known processed tokens
                if re.search(rf"_meg{ext_re}$", name):
                    s += 10
                else:
                    s -= 2
            else:
                s += 10 if prefer in name else 0

        # Prefer exact '_meg.ext' over suffixed variants when ties remain (BIDS)
        if re.search(rf"_meg{ext_re}$", name):
            s += 2

        return s

    candidates = sorted(candidates, key=score, reverse=True)

    if len(candidates) > 1:
        print("Multiple candidates found. Using best match:")
        for c in candidates[:10]:
            print(f"  - {c}")
        print(f"Selected: {candidates[0]}")

    return candidates[0]


def find_erm(root_dir: str,
                 subject_id: str,
                 session: str | None = None,
                 system: str = "MEGIN") -> Path:
    """Find ERM file for either naming style:

    Supports:
      - MNE:  sub_NVAR008_erm_raw*.ext
      - BIDS: sub-BRS0276_ses-20250710_task-erm_meg*.ext
      - BIDS noise alias: sub-BRS0276_ses-20250710_task-noise_meg*.ext
    """

    root_in = Path(root_dir).expanduser().resolve()
    if not root_in.exists():
        raise FileNotFoundError(f"root_dir not found: {root_in}")

    system_upper = system.upper()
    if system_upper not in {"MEGIN", "CTF"}:
        raise ValueError("system must be 'MEGIN' or 'CTF'")
    ext = ".fif" if system_upper == "MEGIN" else ".ds"

    meg_root = root_in / "MEG" if (root_in / "MEG").exists() else root_in

    ses_norm = None
    if session:
        # For BIDS: session may be 'ses-20241217' or '20241217'
        if session.startswith("ses-"):
            ses_norm = session
        else:
            ses_norm = session

    # Constrain search to MEG tree only (avoid derivatives) and to the requested session if provided.
    search_roots: list[Path] = []
    base_roots: list[Path] = []
    if ses_norm is not None:
        base_roots.append(meg_root / subject_id / ses_norm)
        if not ses_norm.startswith("ses-"):
            base_roots.append(meg_root / subject_id / f"ses-{ses_norm}")
        else:
            base_roots.append(meg_root / subject_id / ses_norm.replace("ses-", ""))
    else:
        base_roots.append(meg_root / subject_id)

    # Include common nested variant MEG/sub-X/ses-YYYY/meg
    for br in base_roots:
        search_roots.append(br)
        search_roots.append(br / "meg")

    search_roots = list(dict.fromkeys(search_roots))

    search_roots = [p for p in search_roots if p.exists()]
    if not search_roots:
        raise FileNotFoundError(
            "No ERM search root found for the provided subject/session.\n"
            f"  meg_root={meg_root}\n  subject_id={subject_id}\n  session={session}\n"
        )

    candidates: list[Path] = []

    # MNE-style ERM
    for sr in search_roots:
        candidates.extend(sr.rglob(f"{subject_id}_erm_raw*{ext}"))
        # Some centers name empty-room as "noise"
        candidates.extend(sr.rglob(f"{subject_id}_noise_raw*{ext}"))

    # BIDS-style ERM
    if ses_norm:
        for sr in search_roots:
            candidates.extend(sr.rglob(f"{subject_id}_{ses_norm}_task-erm*_meg*{ext}"))
            candidates.extend(sr.rglob(f"{subject_id}_{ses_norm}_task-noise*_meg*{ext}"))

    candidates = sorted({c.resolve() for c in candidates if c.is_file() or c.is_dir()})

    # Remove cached tSSS/filt/proj outputs
    candidates = [c for c in candidates if "_tsss" not in c.name and "_filt" not in c.name and "_proj" not in c.name]

    if not candidates:
        raise FileNotFoundError(
            "No ERM file found.\n"
            f"  root_dir={root_in}\n  subject_id={subject_id}\n  session={session}\n  system={system_upper}\n"
            "Tried MNE-style and BIDS-style ERM patterns.\n"
        )

    if ses_norm is not None:
        candidates = sorted(candidates, key=lambda p: (0 if f"/{ses_norm}/" in p.as_posix() else 1, len(p.name), p.name))
    else:
        candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))

    if len(candidates) > 1:
        print("Multiple ERM candidates found. Using best match:")
        for c in candidates[:10]:
            print(f"  - {c}")
        print(f"Selected ERM: {candidates[0]}")

    return candidates[0]

# ----------------------------------------------------------

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def _already_has_sss(r: mne.io.BaseRaw) -> bool:
    """Return True if Maxwell/SSS/tSSS has already been applied to this Raw."""
    try:
        ph = r.info.get("proc_history", [])
    except Exception:
        ph = []
    if not ph:
        return False
    # MNE stores Maxwell filtering provenance in proc_history entries (typically under 'max_info')
    for entry in ph:
        if isinstance(entry, dict) and ("max_info" in entry):
            return True
    return False


# ----------------------------------------------------------
# Main preprocessing function
# ----------------------------------------------------------

def preprocess_subject(
    root_dir: str,
    subject_id: str,
    session: str | None = None,
    suffix : str | None = None,
    task: str = 'rest1',
    trans: str = '-corr_trans.fif',
    in_file: str | None = None,
    erm_file: str | None = None,
    task_basename: str = "{sub}_{task}_raw",
    erm_basename: str = "{sub}_erm_raw",
    tsss_dir: str = "/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023",
    st_duration: float = 10.0,
    sss_erm_st_duration: float = None,
    l_freq: float = 0.5,
    h_freq: float = 200.0,
    line_freqs: tuple = (60, 120, 180),
    downsample: int = 500,
    crop_tmin: tuple = (10, 10),
    crop_tmax: tuple = (110, 250),
    ecg_ch: str = "ECG003",
    eog_ch: str = ["EOG001", "EOG002"],
    reject_mag: float = 4e-12,
    reject_grad: float = 4000e-13,
    eSSS : str | None = None,
    subjects_dir_name: str = "MRI/freesurfer",
    compute_bem_if_missing: bool = True,
    bem_watershed: bool = True,
    inv_method: str = "dSPM",
    snr: float = 3.0,
    bands: dict = None,
    additional_bads: tuple = (),
    n_jobs: int = 8,
    num_proj: tuple = (1,1), # ECG and EOG proj
    erm_ssp_band: str | list = "broad",
    verbose: bool = False,
    system: str = "MEGIN",
    json: bool = False,
    overwrite: bool = False,
):
    """
    Generic preprocessing pipeline for MEGIN/CTF resting-state data:
    ERM-based SSP -> (tSSS for MEGIN only) -> filtering -> ECG/EOG QC -> BEM/src/forward/inverse -> STC -> bandwise PSDs.

    Saves: report HTML, tSSS and filtered FIF files, head position .pos, and STC files.
    """
    # ---- Verbose control ----
    if not verbose:
        mne.set_log_level("ERROR")
    else:
        mne.set_log_level("INFO")

    system_upper = system.upper()
    if system_upper not in {"MEGIN", "CTF"}:
        raise ValueError("system must be 'MEGIN' or 'CTF'")
    ext = ".fif" if system_upper == "MEGIN" else ".ds"

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta":  (13, 30),
            "g_low": (30, 50),
            "g_high": (50, 90),
        }

    # ---- Directory setup ----
    subject = subject_id
    parent_path = os.path.abspath(root_dir)
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generic_taskfree_{subject_id}_{timestamp}.txt")

    fs_dir = os.path.join(parent_path, subjects_dir_name)

    if session is None:
        meg_dir = os.path.join(root_dir, "MEG", subject_id)
        deriv_dir = os.path.join(parent_path, "derivatives", subject_id)    # /derivatives/sub-XX
    else:
        meg_dir = os.path.join(root_dir, "MEG", subject_id, session)
        deriv_dir = os.path.join(parent_path, "derivatives", subject_id, session)    # /derivatives/sub-XX

    # Support BIDS-style nested layout: .../MEG/sub-XX/ses-YYYY/meg
    meg_nested = os.path.join(meg_dir, "meg")
    meg_dir = meg_nested if os.path.isdir(meg_nested) else meg_dir

    if json:
        try:
            head_coordinate_file = os.path.join(meg_dir, f"{subject_id}_{session}_coordsystem.json")
            head_coordinates = io_handlers.load_json(head_coordinate_file)
            intended = io_handlers.strip_bids_prefix(head_coordinates["IntendedFor"])
            mri_basename = io_handlers.strip_nii_suffix(os.path.basename(intended))
            fs_subject = io_handlers.extract_bids_id(mri_basename)
            while not os.path.isdir(os.path.join(fs_dir, fs_subject)) and re.search(r'_', fs_subject): #will look for the file that matches inside the freesufer output folder (no suffix).
                fs_subject = fs_subject.rsplit('_', 1)[0]
            print("→ .json file was specified. following the path to subjects surface...")

        except Exception as e:
            # ---- Write error to txt ----
            print(f"⚠️ Could not find .json file: {e}. System will exit")
            
            with open(log_file, "w") as f:
                f.write("No .json file found\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())

            sys.exit(1)
    else:
            fs_subject = f"{subject_id}_{suffix}" if suffix else subject_id# /MRI/freesurfer/sub-XX


    os.makedirs(deriv_dir, exist_ok=True)

    # ---- Expected inputs ----

    task_basename = task_basename[:-4] + ext
    erm_basename = erm_basename[:-4] + ext

    if in_file is not None:
        path2raw = str(Path(in_file).expanduser().resolve())
    else:
        path2raw = os.path.join(meg_dir, task_basename.format(sub=subject_id, task=task))


    # ---- Head position path (derived from input FIF filename) ----
    # Examples:
    #   sub_NVAR008_rest1_raw.fif -> sub_NVAR008_rest1_raw_head_pos.pos
    #   sub-BRS0034_ses-20241217_task-rest_run-1_meg_digFiltered.fif -> ..._meg_digFiltered_head_pos.pos
    head_pos_path = str(Path(path2raw).with_suffix("") ) + "_head_pos.pos"

    if erm_file is not None:
        path2raw_erm = str(Path(erm_file).expanduser().resolve())
    else:
        # fallback to legacy template (still works for MNE-style if located in meg_dir)
        path2raw_erm = os.path.join(meg_dir, erm_basename.format(sub=subject_id))


    if not os.path.exists(path2raw):
        raise FileNotFoundError(f"Raw file not found: {path2raw}")
    if not os.path.exists(path2raw_erm):
        raise FileNotFoundError(f"ERM raw file not found: {path2raw_erm}")

    # ---- tSSS calibration files ----
    calibration = os.path.join(tsss_dir, "sss_cal.dat")
    cross_talk = os.path.join(tsss_dir, "ct_sparse.fif")
    if not os.path.exists(calibration) or not os.path.exists(cross_talk):
        print("⚠️ calibration/crosstalk not found, continuing without them (MNE will handle gracefully).")

    # ---- Report initialization ----
    report = Report(title=Path(os.path.basename(path2raw)).stem  + "_QC_report", raw_psd=False)

    # ---- Load data ----
    raw = preprocessing.read_data(path2raw)
    raw.del_proj()
    raw_erm = preprocessing.read_data(path2raw_erm)
    raw_erm.del_proj()
    
    if system_upper == "CTF" and os.path.isfile(os.path.join(meg_dir,f"{subject_id}_{session}_hsp_ready.fif")):
       path2hsp=os.path.join(meg_dir,f"{subject_id}_{session}_hsp_ready.fif")
       print(path2hsp)
       src_raw = mne.io.read_raw_fif(path2hsp, preload=False)
       fids, hsp, hpi = io_handlers.extract_dig_points(src_raw.info)
       print("→ Adding missing fiducials:", fids.keys(), "HSP:", hsp.shape, "HPI:", hpi.shape)
       raw = io_handlers.inject_dig_into_raw(raw, fids=fids, hsp=hsp, hpi=hpi)

    report.add_raw(raw=raw, title="Raw Resting", scalings='auto')

    # ---- Head Movement Report ----

    try: 
        preprocessing.compute_head_movement_report(raw, report, subject_id, deriv_dir, system_upper)
    
    except Exception as e:
        print(f"⚠️ Head movement report failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ Head movement report failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    # ---- PSD before tSSS ----
    fig = raw.compute_psd(fmax=200,
            method="welch", picks=['meg'],
            n_fft=int(4 * raw.info["sfreq"]),     # 4-second window
            n_overlap=int(2 * raw.info["sfreq"]),     # 50% overlap (2-second)
            average='mean',
            window='hann').plot(picks="data", exclude="bads", amplitude=True, show=False)
    report.add_figure(fig=fig, title="Raw PSD")

    # ---- Head position and channel renaming----
    if system_upper == "MEGIN":
        try:
            #head_pos_path = os.path.join(meg_dir,  "head_pos.pos")
            head_pos = mne.chpi.read_head_pos(head_pos_path)
        except Exception as e:
            print(f"⚠️ Could not read head_pos from {head_pos_path}: {e}, it will be computed now")
            head_pos = preprocessing.compute_head_position(raw)
            mne.chpi.write_head_pos(head_pos_path, head_pos)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(log_file, "w") as f:
                f.write(f"⚠️ Could not read head_pos from {head_pos_path}. it was computed\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
    else:
        head_pos = None
        mapping = {"HEOG": "eog", "VEOG": "eog","ECG": "ecg"}
        ch_type_map = {ch: typ for ch, typ in mapping.items() if ch in raw.ch_names}
        if ch_type_map:
            raw.set_channel_types(ch_type_map)
        
        raw.pick(["meg", "stim", "misc", "eog", "ecg"]).load_data()

    # ---- Cached tSSS paths (derived from input FIF filenames) ----
    # Examples:
    #   sub_NVAR008_rest1_raw.fif -> sub_NVAR008_rest1_raw_tsss.fif
    #   sub-BRS0034_ses-20241217_task-rest_run-1_meg_digFiltered.fif -> ..._meg_digFiltered_tsss.fif
    tsss_raw_path = str(Path(path2raw).with_suffix("")) + "_tsss.fif"

    #   sub_NVAR008_erm_raw.fif -> sub_NVAR008_erm_raw_tsss.fif
    #   sub-BRS0034_ses-20241217_task-erm_meg.fif -> ..._task-erm_meg_tsss.fif
    tsss_erm_path = str(Path(path2raw_erm).with_suffix("")) + "_tsss.fif"
    extended_proj = []
    if system_upper == "MEGIN":
        try:
            if os.path.exists(tsss_raw_path) and os.path.exists(tsss_erm_path) and not overwrite:
                print("→ Loading existing tSSS files...")
                raw = mne.io.read_raw_fif(tsss_raw_path, preload=True)
                raw_erm = mne.io.read_raw_fif(tsss_erm_path, preload=True)
                if head_pos is None:
                    try:
                        head_pos = mne.chpi.read_head_pos(head_pos_path)
                    except Exception:
                        head_pos = None
            else:
                # ---------- REST / TASK DATA ----------
                if _already_has_sss(raw):
                    print("→ Input data already has Maxwell/SSS applied; skipping tSSS and caching as-is...")
                    if not os.path.exists(tsss_raw_path):
                        raw.save(tsss_raw_path, overwrite=True)
                else:
                    print("→ Applying tSSS to resting...")

                    # Build extended projections from ERM
                    if eSSS: 

                        for i, (low_freq, high_freq) in enumerate(eSSS):
                            print(f"Computing projections for band {low_freq}-{high_freq} Hz")
                            filt_erm = raw_erm.copy().filter(l_freq=low_freq, h_freq=high_freq)
                            # You can customize number of components per band
                            if i == 0:
                                n_mag, n_grad = 1, 1
                            else:
                                n_mag, n_grad = 3, 3

                            proj = mne.compute_proj_raw(
                                filt_erm,
                                meg="combined",
                                n_mag=n_mag,
                                n_grad=n_grad
                            )
                            extended_proj.extend(proj)

                    raw = preprocessing.max_filter(
                        raw,
                        extended_proj=extended_proj,
                        calibration=calibration if os.path.exists(calibration) else None,
                        cross_talk=cross_talk if os.path.exists(cross_talk) else None,
                        st_duration=st_duration,
                        head_pos=head_pos,
                    )
                    raw.save(tsss_raw_path, overwrite=True)

            # ---------- ERM ----------
            if _already_has_sss(raw_erm):
                print("→ ERM already has Maxwell/SSS applied; skipping SSS and caching as-is...")
                if not os.path.exists(tsss_erm_path):
                    raw_erm.save(tsss_erm_path, overwrite=True)
            else:
                print("→ Applying SSS to ERM...")
                raw_erm = preprocessing.max_filter(
                    raw_erm,
                    extended_proj=extended_proj,
                    calibration=calibration if os.path.exists(calibration) else None,
                    cross_talk=cross_talk if os.path.exists(cross_talk) else None,
                    st_duration=sss_erm_st_duration,
                    head_pos=None,
                )
                raw_erm.save(tsss_erm_path, overwrite=True)
                
        except Exception as e:
            print(f"⚠️ Could not apply Maxwell filter: {e}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(log_file, "w") as f:
                f.write(f"⚠️ Could not apply Maxwell filter: {e}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Error message:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())   
        # Rename path2raw to facilitate naming conventions
        path2raw = tsss_raw_path
        path2raw_erm = tsss_erm_path
        
        # ---- PSD after tSSS ----
        fig = raw.compute_psd(fmax=180,
                method="welch", picks=['meg'],
                n_fft=int(4 * raw.info["sfreq"]),     # 4-second window
                n_overlap=int(2 * raw.info["sfreq"]),     # 50% overlap (2-second)
                average='mean',
                window='hann').plot(picks="data", exclude="bads", amplitude=True, show=False)
        report.add_figure(fig=fig, title=f"PSD after tSSS, eSSS:{eSSS}")
        
    else:
        print("→ System set to CTF: skipping tSSS/Maxwell filtering; using input data directly.")

    

    # ---- Temporal cropping ----
    try:
        raw.crop(tmin=crop_tmin[1], tmax=crop_tmax[1])
    except Exception as e:
        print(f"⚠️ Raw cropping failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ Raw cropping failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        
    try:
        raw_erm.crop(tmin=crop_tmin[0], tmax=crop_tmax[0])
    except Exception as e:
        print(f"⚠️ Empty room cropping failed: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ Empty room cropping failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())



    # ---- Filtering & notch ----
    print(f"→ Filtering {l_freq}-{h_freq} Hz, notch {line_freqs}")
    raw = preprocessing.filter_data(raw, l_freq=l_freq, h_freq=h_freq, line_freqs=line_freqs)
    raw_erm = preprocessing.filter_data(raw_erm, l_freq=l_freq, h_freq=h_freq, line_freqs=line_freqs)

    # ---- Downsample ----
    if downsample:
        print(f"→ Downsampling to {downsample} Hz")
        raw.resample(downsample)
        raw_erm.resample(downsample)
        fig = raw.compute_psd(fmax=180,
            method="welch",
            n_fft=int(4 * raw.info["sfreq"]),     # 4-second window
            n_overlap=int(2 * raw.info["sfreq"]),     # 50% overlap (2-second)
            average='mean',
            window='hann').plot(picks="data", exclude="bads", amplitude=True, show=False)

        report.add_figure(fig, title=f"PSD after filters + downsample ({downsample} Hz)")

    # ---- Additional bad channels ----
    if additional_bads:
        raw.info["bads"].extend(additional_bads)
        raw_erm.info["bads"].extend(additional_bads)

    # ---- ECG/EOG QC ----
    try:
        print("→ EOG/ECG artifact detection")
        ecg_ev = mne.preprocessing.create_ecg_epochs(raw, ch_name=ecg_ch).average()
        fig = ecg_ev.plot_joint(show=False)
        report.add_figure(fig, title="ECG events")
    except Exception as e:
        print(f"⚠️ ECG QC failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ ECQ QC failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
    try:
        eog_ev = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_ch).average()
        fig = eog_ev.plot_joint(show=False)
        report.add_figure(fig, title="EOG events")
    
    except Exception as e:
        print(f"⚠️ EOG QC failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ EOC QC failed \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    
    # ---- SSP ----
    try:
        
        print("→ Applying SSP")
        # ERM SSP can be broadband or computed from a specific band
        erm_for_ssp = raw_erm
        if erm_ssp_band != "broad":
            if not isinstance(erm_ssp_band, (list, tuple)) or len(erm_ssp_band) != 2:
                raise ValueError("erm_ssp_band must be 'broad' or [low, high]")
            low, high = erm_ssp_band
            print(f"→ Computing ERM SSP using filtered band {low}-{high} Hz")
            erm_ssp_caption = f"Band-limited ERM SSP projectors extracted after filtering the ERM from {low} to {high} Hz."
            erm_for_ssp = raw_erm.copy().filter(l_freq=low, h_freq=high)
        else:
            print("→ Computing ERM SSP using broadband ERM (no filtering)")
            erm_ssp_caption = "Broadband ERM SSP projectors; no additional ERM filtering was applied before SSP extraction."
        er_proj = mne.compute_proj_raw(erm_for_ssp, n_grad=3, n_mag=3, verbose=True)
        
        er_exp_var = []
        for proj in er_proj:
            if "explained_var" in proj:
                er_exp_var.append(f"{np.round(proj['explained_var'], 2)}%")

        fig = mne.viz.plot_projs_topomap(er_proj, info=raw_erm.info, show=False)
        fig.suptitle("ERM SSP projectors")
        report.add_figure(
            fig,
            title="ERM Projections",
            caption=f"{erm_ssp_caption} Explained variance: {', '.join(er_exp_var) if er_exp_var else 'not available'}."
        )

        # Create SSP ecg/eog projectors
        ecg_proj, ecg_array = mne.preprocessing.compute_proj_ecg(raw,n_grad=3,n_mag=3, reject=None) # For ECG proj, first pca is always enough
        fig = mne.viz.plot_projs_joint(ecg_proj, ecg_ev, show=False)
        fig.suptitle("ECG projectors")
        exp_var = []
        for i in range(len(ecg_proj)):
            exp_var.append(str(np.round(ecg_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        report.add_figure(fig, title='Ecg Projections', caption = f"{', '.join(exp_var)} — num of proj selected = {num_proj[0]}")
            
        eog_proj, eog_array = mne.preprocessing.compute_proj_eog(raw,n_grad=3,n_mag=3, reject=None) # Default options look fine
        fig = mne.viz.plot_projs_joint(eog_proj, eog_ev, show=False)
        fig.suptitle("EOG projectors")
        exp_var = []
        for i in range(len(eog_proj)):
            exp_var.append(str(np.round(eog_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        report.add_figure(fig, title='Eog Projections', caption = f"{', '.join(exp_var)} — num of proj selected = {num_proj[1]}")

        # EOG/ECG projections are added after ERM SSP projectors
        for i in range(0,num_proj[0]):
            raw.add_proj(ecg_proj[i]) #For ECG proj, first pca is always enough
            raw_erm.add_proj(ecg_proj[i]) 

        for i in range(0,num_proj[1]):
            raw.add_proj(eog_proj[i]) #For EOG proj, first pca seems enough
            raw_erm.add_proj(eog_proj[i])

        for i in range(0,num_proj[0]):
            raw.add_proj(er_proj[i]) #For ECG proj, first pca is always enough
            raw_erm.add_proj(er_proj[i]) 

        raw.apply_proj()
        raw_erm.apply_proj()
    except Exception as e:
        print(f"⚠️ SSP computation failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ SSP computation failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    # --- Amplitude and gradient thresholds ----
    try:
        print("→ Amplitude and gradient thresholding")
        n_windows, bad_windows, metrics, thresholds, bad_times = preprocessing.detect_bad_mad_grads_mags(
        raw,
        win_length=1.0,
        n_mad=3,
        )

        fig = preprocessing.plot_mad_qc(n_windows, bad_windows, metrics, thresholds, subject_name=subject)
        report.add_figure(fig, title='MAD amplitude/gradient QC', caption='P2P + gradient thresholds')
        plt.close("all")
        onset = [t[0]+raw.first_time for t in bad_times]
        duration = [t[1]-t[0] for t in bad_times]
        labels = ["BAD_mad"] * len(bad_times)
        orig_time = raw.info["meas_date"]
        ann = mne.Annotations(onset=onset, duration=duration, description=labels, orig_time=orig_time)
        raw.set_annotations(ann + raw.annotations)
        raw.load_data() # Ensure BAD segments are masked

        fig = raw.compute_psd(fmax=180).plot(picks="data", exclude="bads", amplitude=True, show=False)

        report.add_figure(fig, title="PSD after MAD")
        
        scalings = 'auto'
        if system_upper == "CTF":
            scalings = dict(mag=1e-10, grad=4e-10, eeg=20e-6, eog=150e-6, ecg=5e-4,
                            emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                            resp=1, chpi=1e-4, whitened=1e2)
        
        fig_butterfly = raw.plot(start=0, duration=raw.times[-1],scalings=scalings, show=False)
        report.add_figure(
            fig_butterfly,
            title="Time Series (Bad Windows Range)",
            caption=f"From first to last BAD_mad window ({bad_times[0][0]:.2f}s–{bad_times[-1][1]:.2f}s)"
        )

    
    except Exception as e:
        print(f"⚠️ MAD failed: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ MAD failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    # ---- Data and noise covariance ----
    print("→ Data and noise covariance")
    data_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=300)
    noise_cov = mne.compute_raw_covariance(raw_erm, tmin=0, tmax=300)

    report.add_covariance(data_cov, info=raw.info, title='Data covariance')
    report.add_covariance(noise_cov, info=raw_erm.info, title='Noise covariance')

    # ---- Save filtered raw ----
    if ext == "fif":
        filt_path = os.path.join(str(Path(path2raw).with_suffix("") ) + f"_filt_proj.fif")
        raw.save(filt_path, overwrite=True)

    # ======================================================
    #       SOURCE MODELING (BEM / SRC / FORWARD / INVERSE)
    # ======================================================

    # ---- Coregistration metrics + report visualization ----
    print("→ Coregistration")

    trans_path = os.path.join(meg_dir, f"{subject}-{session}{trans}")
    trans_path = trans_path if os.path.isfile(trans_path) else os.path.join(meg_dir, f"{subject}_{session}{trans}")
       

    if os.path.exists(trans_path):
        # Load the .trans file
        trans = mne.read_trans(trans_path)

        # Compute dig → MRI distances
        # distances = mne.dig_mri_distances( info=raw.info, trans=trans, subject=fs_subject,subjects_dir=fs_dir)

        #mean_distance_mm = np.mean(distances) * 1000
        #std_distance_mm  = np.std(distances)  * 1000

        #note = f"Distance: {mean_distance_mm:.2f} +- {std_distance_mm:.2f} mm"

        report.add_trans(
            trans=trans_path,
            info=raw.info,
            subject=fs_subject,
            subjects_dir=fs_dir,
            plot_kwargs=dict(surfaces='head-dense',
            mri_fiducials=True, meg={"helmet": 0.1, "sensors": 0.1, "ref": 1}),
            title='Coregistration',
            alpha=1
        )

    else:
        print(f"⚠️ Missing trans file: {trans_path}")
    

    # ---- BEM ----
    bem_path = os.path.join(fs_dir, fs_subject, "bem", f"{subject}-5120-5120-5120-bem-sol.fif")
    bem_dir = os.path.join(fs_dir, "bem")
    src_path = os.path.join(deriv_dir, Path(os.path.basename(path2raw)).stem + "_src.fif")

    if compute_bem_if_missing and not os.path.exists(bem_path):
        os.makedirs(bem_dir, exist_ok=True)
        conductivity = (0.3,)   # Single layer for MEG
        model = mne.make_bem_model(subject=fs_subject, ico=4, #The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280. If None, no subsampling is applied.
                            conductivity=conductivity, 
                            subjects_dir=fs_dir) #bem conductivity model
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_path, bem_sol)
        if bem_watershed:
            print("→ Creating watershed BEM (if missing)...")
            try:
                mne.bem.make_watershed_bem(subject=fs_subject, subjects_dir=fs_dir, overwrite=True)
                mne.bem.make_scalp_surfaces(subject=fs_subject, subjects_dir=fs_dir, overwrite=True) #Creates the high resolution -head-dense.fif

            except Exception as e:
                print(f"⚠️ Watershed BEM failed: {e}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(log_file, "w") as f:
                    f.write("⚠️ Watershed BEM failed: \n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                    f.write("Error message:\n")
                    f.write(str(e) + "\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())

    if not os.path.exists(src_path):
        print("→ Setting up source space...")
        src = mne.setup_source_space(subject=fs_subject, subjects_dir=fs_dir, add_dist="patch")
        src.save(src_path, overwrite=True)
    else:
        src = mne.read_source_spaces(src_path)

    # ---- BEM / alignment QC ----
    try:
        # Plot BEM (2D slices)
        fig = mne.viz.plot_bem(subject=fs_subject, subjects_dir=fs_dir, src=src)
        report.add_figure(fig, title="Sources on BEM")

        # Plot 3D alignment (no 'show' kwarg)
        fig = mne.viz.plot_alignment(
            subject=fs_subject,
            subjects_dir=fs_dir,
            surfaces="white", #white becuase mne use white for sourse reconstruction? 
            coord_frame="mri",
            src=src
        )
        mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75, distance=0.40,
                            focalpoint=(-0.03, -0.01, 0.03))
        report.add_figure(fig, title="Alignment (sources over WM)")

        # Safely close 3D figure
        plt.close("all")
        mne.viz.close_3d_figure(fig) # Add sliding bar to rotate the brain

    except Exception as e:
        print(f"⚠️ BEM/alignment plots failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ BEM/alignment plots failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    # ---- Forward ----

    try:
        print('→ Foward Solutions')
        fwd = mne.make_forward_solution(
                    raw.info, trans=trans_path, src=src, bem=bem_path, meg=True, eeg=False, mindist=0.0, n_jobs=n_jobs)
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)

    except Exception as e:
        print(f"⚠️ Foward solution failed: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(log_file, "w") as f:
            f.write("⚠️ Forward solution failed: \n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Error message:\n")
            f.write(str(e) + "\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    # ---- Inverse ----

    # Choose the STC directory based on session

    stc_dir = os.path.join(deriv_dir, inv_method, 'stc')

    # Ensure directory exists
    os.makedirs(stc_dir, exist_ok=True)

    # Base filename for the STC
    stc_base = Path(os.path.basename(path2raw)).stem + f"_{inv_method}_stc"
    stc_path = os.path.join(stc_dir, stc_base)   # <<--- correct final path


    # Missing files check
    if not os.path.exists(trans_path):
        print(f"⚠️ Missing trans file: {trans_path}")
        stc = None

    elif not os.path.exists(bem_path):
        print(f"⚠️ Missing BEM file: {bem_path}")
        stc = None

    else:
        # ------------------ MINIMUM NORM ------------------
        if inv_method != 'beamformer':
            print(f"→ Computing Source Estimation {inv_method}")
            # Silence annoying joblib warnings
            os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
            os.environ["JOBLIB_NO_MPI"] = "1"
            # if STC does not exist: compute it
            if not os.path.exists(stc_path + "-lh.stc") or overwrite:
                print("→ Forward solution...")
                print(f"→ Inverse operator ({inv_method})...")

                inv = mne.minimum_norm.make_inverse_operator(
                    raw.info, fwd, noise_cov, loose=0.2, depth=0.8
                )

                lambda2 = 1.0 / (snr ** 2)

                stc = mne.minimum_norm.apply_inverse_raw(
                    raw, inv, lambda2=lambda2, method=inv_method
                )

                try:
                    stc.save(stc_path, overwrite=True)
                    print(f"→ STC saved at {stc_path}")
                except Exception as e:
                    print(f"⚠️ Could not save STC: {e}")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(log_file, "w") as f:
                        f.write("⚠️ Could not save STC: \n")
                        f.write(f"Timestamp: {timestamp}\n\n")
                        f.write("Error message:\n")
                        f.write(str(e) + "\n\n")
                        f.write("Traceback:\n")
                        f.write(traceback.format_exc())

            else:
                print(f"→ Reading existing STC ({inv_method})...")
                stc = mne.read_source_estimate(stc_path)

        # ------------------ BEAMFORMER ------------------
        else:
            if not os.path.exists(stc_path + "-lh.stc") or overwrite:
                print(f"→ Computing Source Estimation {inv_method}")
                start, stop = raw.time_as_index([0,crop_tmax[1]-crop_tmin[1]])

                #Whats all this hyperparameters?! Make it more clear to you and everyone
                filters = mne.beamformer.make_lcmv(
                    raw.info,
                    fwd,
                    data_cov,
                    reg=0.05, #whats the regularization?
                    noise_cov=noise_cov,
                    pick_ori="max-power",
                    weight_norm="unit-noise-gain",
                    rank='info'
                )

                stc = mne.beamformer.apply_lcmv_raw(raw, filters,
                                                    start=start, stop=stop)
                

                try:
                    stc.save(stc_path, overwrite=True)
                    print(f"→ STC saved at {stc_path}")
                except Exception as e:
                    print(f"⚠️ Could not save STC: {e}")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(log_file, "w") as f:
                        f.write("⚠️ Could not save STC: \n")
                        f.write(f"Timestamp: {timestamp}\n\n")
                        f.write("Error message:\n")
                        f.write(str(e) + "\n\n")
                        f.write("Traceback:\n")
                        f.write(traceback.format_exc())

            else:
                print("→ Reading Beamformer STC...")
                stc = mne.read_source_estimate(stc_path)


    # ======================================================
    #       BANDWISE SOURCE PSDs
    # ======================================================
    if stc is not None:
        print("→ Computing vertex-wise PSDs (parallel)...")
        psd_all, freqs, psd_normalized, band_powers = postprocessing.PSD_per_vertex_parallel(stc,bands)
        #%% Figure hyperparameters
        fig, axes = plt.subplots(6, 3, figsize=(10, 10))  # 5 filas, 2 columnas
        axes = axes.reshape(6, 3)  # Asegurar que tenga forma (5, 2) para indexación clara

        # Agregar títulos superiores para las columnas
        axes[0, 0].set_title("Lateral", fontsize=16)
        axes[0, 1].set_title("Medial", fontsize=16)
        axes[0, 2].set_title("Dorsal", fontsize=16)


        surfer_kwargs = dict(surface='pial',
                        hemi='split',
                        subject=fs_subject, 
                        subjects_dir=fs_dir,
                        #views="medial",
                        colormap='jet',
                        time_unit="s",
                        size=(1100, 500),
                        smoothing_steps=5,
                        colorbar=False)



        # Generar las visualizaciones de las bandas para lateral y medial
        for i, (band, power) in enumerate(band_powers.items()):

            surfer_kwargs['hemi']='split'
            # SourceEstimate per band
            stc_band = mne.SourceEstimate(power, vertices=stc.vertices,
                                          tmin=0, tstep=.25, subject=fs_subject)

            # stc_band_morph = morph.apply(stc_band)
            clim = dict(kind="value", lims=[.0* max(power), 0.4 * max(power), .8 * max(power)])
            # Lateral view
            surfer_kwargs['views'] = 'lateral'
            brain_lateral = stc_band.plot(**surfer_kwargs,
                                          clim=clim)
            img_lateral = brain_lateral.screenshot()
            brain_lateral.close()  # Closing interactive object

            # Visualización medial
            surfer_kwargs['views'] = 'medial'
            brain_medial = stc_band.plot(**surfer_kwargs,
                                         clim=clim)
            img_medial = brain_medial.screenshot()
            brain_medial.close()  # Closing interactive object

            # Medial View
            surfer_kwargs['hemi']='both'
            surfer_kwargs['views'] = 'dorsal'
            brain_dorsal = stc_band.plot(**surfer_kwargs,
                                         clim=clim)
            img_dorsal = brain_dorsal.screenshot()
            brain_dorsal.close()  # Closing interactive object

            # Subplots
            axes[i, 0].imshow(img_lateral)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(img_medial)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(img_dorsal)
            axes[i, 2].axis('off')

            axes[i, 0].text(-0.1, 0.5, band, fontsize=14, va='center', ha='right',
                             transform=axes[i, 0].transAxes, rotation=90)

        plt.suptitle(subject,fontsize=16, fontweight='bold')

        # Adjust layout to remove gaps between subplots
        plt.subplots_adjust(hspace=0, wspace=0)

        report.add_figure(fig, title='Spectrally Resolved Source Estimation')
        report_dir = os.path.join(deriv_dir, inv_method, 'report')
        report_path = os.path.join(report_dir, Path(os.path.basename(path2raw)).stem + "_QC_report.html")
        # Ensure directory exists
        os.makedirs(report_dir, exist_ok=True)
        fig_path = os.path.join(report_dir, "PSD_band_dist" + Path(os.path.basename(path2raw)).stem + ".png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        report.add_figure(plt.figure(), title="Spectrally Resolved Source Estimation (placeholder panel)")

        # ---- Save report ----
        report.save(report_path, overwrite=True)
        print(f"→ Report saved at: {report_path}")
        print(f"✅ Finished preprocessing for {subject}")


# ----------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Preprocess MEGIN/CTF task-free MEG data with ERM-SSP, tSSS (MEGIN only), QC, and source modeling.")
    p.add_argument("--root_dir", required=True, type=str)
    p.add_argument("--subject_id", required=True, type=str)
    p.add_argument("--session", default=None, required=False, help="Session (e.g., 20241217 or ses-20241217). Optional for MNE-style naming.")
    p.add_argument("--suffix", type=str, default=None, help="Optional FreeSurfer subject suffix (e.g., 'ses-01_run-2' for multiple T1w runs.")
    p.add_argument("--resting", default='rest1', required=False, help="Backward-compatible alias for --task (e.g., rest1/rest2).")
    p.add_argument("--task", default=None, required=False, help="Generic task name. Examples: rest, msit, somatoauditory1, rest1.")
    p.add_argument("--trans", type= str, default='-corr_trans.fif', required=False, help="Generic corregistration file sufix. Examples: -corr_trans.fif, _hsp_ready.fif")
    p.add_argument("--run", type=str, default=None, required=False,
                   help="BIDS run identifier. Accepts: 1, 01, run-1, run-01.")
    p.add_argument("--in_file", type=str, default=None, required=False, help="Explicit path to input FIF (overrides auto-discovery).")
    p.add_argument("--prefer", type=str, default="any",
                   help="Preference token when multiple match: 'any', 'raw', or any substring to prefer (e.g., digFiltered, channels_removed).")
    p.add_argument("--erm_file", type=str, default=None, required=False, help="Explicit path to ERM FIF (overrides auto-discovery).")
    p.add_argument("--system", type=str, default="MEGIN", choices=["MEGIN", "CTF"], help="Acquisition system. Controls file extensions and tSSS application.")
    p.add_argument("--task_basename", type=str, default="{sub}_{task}_raw.fif")
    p.add_argument("--erm_basename", type=str, default="{sub}_erm_raw.fif")
    p.add_argument("--tsss_dir", type=str, default="/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023")
    p.add_argument("--st_duration", type=float, default=10.0)
    p.add_argument("--sss_erm_st_duration", type=float, default=None)
    p.add_argument("--l_freq", type=float, default=0.5)
    p.add_argument("--h_freq", type=float, default=200.0)
    p.add_argument("--line_freqs", type=float, nargs="*", default=[60, 120, 180])
    p.add_argument("--downsample", type=int, default=500)
    p.add_argument("--crop_tmin", type=float, nargs=2, default=[10.0, 10.0])
    p.add_argument("--crop_tmax", type=float, nargs=2, default=[110.0, 250.0])
    p.add_argument("--ecg_ch", type=str, default="ECG003")
    p.add_argument("--eog_ch", type=str, default="EOG001")
    p.add_argument("--reject_mag", type=float, default=4e-12)
    p.add_argument("--reject_grad", type=float, default=4000e-13)
    p.add_argument("--eSSS", type=parse_ranges, default=None, required=False, help="Frequency ranges for eSSS as start-end pairs, e.g. 42-45,50-53")
    p.add_argument("--subjects_dir_name", type=str, default="MRI/freesurfer")
    p.add_argument("--compute_bem_if_missing", action="store_true", default=True)
    p.add_argument("--no_compute_bem_if_missing", dest="compute_bem_if_missing", action="store_false")
    p.add_argument("--bem_watershed", action="store_true", default=True)
    p.add_argument("--no_bem_watershed", dest="bem_watershed", action="store_false")
    p.add_argument("--inv_method", type=str, default="beamformer", choices=["MNE", "dSPM", "sLORETA","beamformer"])
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--num_proj", type=int, nargs=2, default=[1, 1], help="Number of ECG and EOG SSP projectors to apply, respectively. Example: --num_proj 1 1")
    p.add_argument("--erm_ssp_band", type=str, default="broad", help="ERM band for SSP: 'broad' or low-high (e.g. 10-20)")
    # additional_bads como lista
    p.add_argument("--additional_bads", type=str, nargs="*", default=[])
    p.add_argument("--verbose", action="store_true", help="Enable verbose MNE output")
    p.add_argument("--json", action = "store_true", help="Only true if .json file with fidutials exists AND was use to generete the coregistation automatically")
    p.add_argument("--overwrite", action = "store_true", help="If called, it will overwrite all the *_tsss.fif, *.stc and report files, allong any other output.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Resolve task label (preferred: --task, fallback: --resting)
    if args.task is not None:
        task_label = args.task
    elif getattr(args, "resting", None) is not None:
        task_label = args.resting
    else:
        raise ValueError("You must provide --task (preferred) or --resting (legacy).")
    # Resolve input FIF
    if args.in_file is not None:
        file_path = Path(args.in_file).expanduser().resolve()
    else:
        file_path = find_meg(
            root_dir=args.root_dir,
            subject_id=args.subject_id,
            session=args.session,
            task=task_label,
            run=args.run,
            prefer=args.prefer,
            system=args.system,
        )

    print(f"Input MEG path: {file_path}")

    # Resolve ERM FIF
    if args.erm_file is not None:
        erm_path = Path(args.erm_file).expanduser().resolve()
    else:
        erm_path = find_erm(
            root_dir=args.root_dir,
            subject_id=args.subject_id,
            session=args.session,
            system=args.system,
        )

    print(f"ERM path: {erm_path}")

    # Parse erm_ssp_band argument
    if args.erm_ssp_band == "broad":
        erm_ssp_band = "broad"
    else:
        low, high = args.erm_ssp_band.split("-")
        erm_ssp_band = [float(low), float(high)]

    preprocess_subject(
        root_dir=args.root_dir,
        subject_id=args.subject_id,
        session=args.session,
        suffix=args.suffix,
        task=task_label,
        trans=args.trans,
        in_file=str(file_path),
        erm_file=str(erm_path),
        task_basename=args.task_basename,
        erm_basename=args.erm_basename,
        tsss_dir=args.tsss_dir,
        st_duration=args.st_duration,
        sss_erm_st_duration=args.sss_erm_st_duration,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        line_freqs=tuple(args.line_freqs) if args.line_freqs else (),
        downsample=args.downsample,
        crop_tmin=tuple(args.crop_tmin),
        crop_tmax=tuple(args.crop_tmax),
        ecg_ch=args.ecg_ch,
        eog_ch=args.eog_ch,
        reject_mag=args.reject_mag,
        reject_grad=args.reject_grad,
        eSSS=args.eSSS,
        subjects_dir_name=args.subjects_dir_name,
        compute_bem_if_missing=args.compute_bem_if_missing,
        bem_watershed=args.bem_watershed,
        inv_method=args.inv_method,
        snr=args.snr,
        additional_bads=tuple(args.additional_bads),
        n_jobs=args.n_jobs,
        num_proj=args.num_proj,
        erm_ssp_band=erm_ssp_band,
        verbose=args.verbose,
        system=args.system,
        json=args.json,
        overwrite=args.overwrite,
    )