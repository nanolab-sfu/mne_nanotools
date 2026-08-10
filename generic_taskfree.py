#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic task-free MEGIN preprocessing with ERM-SSP, tSSS, QC report, and bandwise source PSDs.

Version 0.1.3 - Last modified 04/05/2026

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
from typing import Literal
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
import warnings
import logging

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

# ---- Channel list parser ----
def parse_channel_list(values):
    """
    Parse channel names passed either as space-separated values, comma-separated
    values, or a mixture of both.

    Examples
    --------
    --additional_bads MEG2443 MEG1032
    --additional_bads MEG2443,MEG1032
    --additional_bads MEG2443,MEG1032 MEG0111
    """
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    channels = []
    for item in values:
        channels.extend(ch.strip() for ch in item.split(",") if ch.strip())

    return channels


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
        bad_tokens = ["_tsss","_tsss_mc", "_filt", "_SSP", "_bp", "_proj", "_src", "_stc", "_head_pos", "_QC_report"]
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
    candidates = [c for c in candidates if "_tsss" not in c.name and "_tsss_mc" not in c.name and "_bp" not in c.name and "_notch" not in c.name and "_filt" not in c.name and "_proj" not in c.name]

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
# Helper: Add figure(s) with captions (handles list/tuple of figs)
def _add_figure_with_caption(report, fig, title, caption=None):
    """Add one or more Matplotlib figures to an MNE report with matching captions.

    Some MNE plotting functions, such as Evoked.plot_joint(), can return a list
    of figures. In that case, Report.add_figure() requires one caption per
    figure, not a single caption string.
    """
    if isinstance(fig, (list, tuple)):
        captions = None if caption is None else [caption] * len(fig)
        report.add_figure(fig=fig, title=title, caption=captions)
    else:
        report.add_figure(fig=fig, title=title, caption=caption)




# ----------------------------------------------------------
# Helper to save preprocessing hyperparameters
# ----------------------------------------------------------

def save_hyperparameters(
    report_dir: str,
    args,
    raw_stem: str,
):
    """Save all CLI hyperparameters used for preprocessing."""

    os.makedirs(report_dir, exist_ok=True)

    name_parts = [raw_stem]

    # Add preferred file type if specified, e.g. digFiltered
    if getattr(args, "prefer", None):
        name_parts.append(str(args.prefer))

    out_name = "_".join(name_parts) + "_hyperparameters.txt"
    out_path = os.path.join(report_dir, out_name)

    with open(out_path, "w") as f:
        f.write("Generic MNE preprocessing hyperparameters\n")
        f.write("=" * 60 + "\n\n")

        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")

    print(f"→ Hyperparameters saved at: {out_path}")

# EOG/ECG projections are added after ERM SSP projectors.
# MNE returns SSP projectors as one list containing different MEG sensor types.
# For MEGIN, descriptions usually contain:
#   - "planar" for gradiometers
#   - "axial" for magnetometers
# Therefore, num_proj_* is interpreted as (n_grad, n_mag).
def _select_proj_by_meg_type(projs, n_proj, label, info, system):
    # num_proj_* can arrive as an int, [int] from argparse nargs=1,
    # or occasionally as a tuple/list. In all cases, use one value.

    n_proj = int(n_proj)

    grad_chs = set(info["ch_names"][idx] for idx in mne.pick_types(info, meg="grad"))
    mag_chs = set(info["ch_names"][idx] for idx in mne.pick_types(info, meg="mag"))

    grad_proj = []
    mag_proj = []
    other_meg_proj = []

    for p in projs:
        desc = p.get("desc", "").lower()
        col_names = set(p.get("data", {}).get("col_names", []))

        if col_names & grad_chs or "planar" in desc or "grad" in desc:
            grad_proj.append(p)
        elif col_names & mag_chs or "axial" in desc or "mag" in desc:
            mag_proj.append(p)
        else:
            other_meg_proj.append(p)

    if system == "MEGIN":
        selected = grad_proj[:n_proj] + mag_proj[:n_proj]
        expected = n_proj * 2

        if len(selected) < expected:
            print(
                f"⚠️ {label}: requested {n_proj} grad + {n_proj} mag projectors, "
                f"but found {len(grad_proj)} grad and {len(mag_proj)} mag projectors. "
                f"Applying {len(selected)} projectors."
            )

    else:
        # CTF systems do not have the same planar/axial MEGIN split.
        # Apply projectors from the available MEG type only.
        available_proj = grad_proj + mag_proj + other_meg_proj
        selected = available_proj[:n_proj]

        if len(selected) < n_proj:
            print(
                f"⚠️ {label}: requested {n_proj} CTF MEG projectors, "
                f"but found only {len(available_proj)}. "
                f"Applying {len(selected)} projectors."
            )

    return selected

def _add_selected_projs(raw_obj, raw_erm_obj, projs):
    if projs:
        raw_obj.add_proj(projs)
        raw_erm_obj.add_proj(projs)

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
    num_proj_eog: tuple = (1,1), # ECG and EOG proj
    num_proj_ecg: tuple = (1,1), # ECG and EOG proj
    num_proj_erm: tuple = (1,1), # SSP empty room
    num_proj_raw: tuple = (1,1), # SSP generic raw
    num_proj_bcglike: tuple = (1,1), # SSP bcglike-events
    erm_ssp_band: Literal["broad"] | tuple[float, float] | None = None,
    raw_ssp_band : str | None = None,
    bcglike_ssp: bool = False,
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
        # Suppress MNE INFO messages
        mne.set_log_level("ERROR")

        # Suppress specific scipy/mne PSD warnings
        warnings.filterwarnings(
            "ignore",
            message="nperseg = .* is greater than input length.*",
            category=UserWarning,
        )

        # Optional: suppress matplotlib/font warnings too
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

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
        print("→ Adding head movement report:")
        preprocessing.compute_head_movement_report(raw, report, Path(os.path.basename(path2raw)).stem, deriv_dir, system_upper)
    
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
    tsss_mc_raw_path = str(Path(path2raw).with_suffix("")) + "_tsss_mc.fif"

    #   sub_NVAR008_erm_raw.fif -> sub_NVAR008_erm_raw_tsss.fif
    #   sub-BRS0034_ses-20241217_task-erm_meg.fif -> ..._task-erm_meg_tsss.fif
    tsss_erm_path = str(Path(path2raw_erm).with_suffix("")) + "_tsss.fif"
    tsss_mc_erm_path = str(Path(path2raw_erm).with_suffix("")) + "_tsss_mc.fif"

    # Prefer the movement-compensated cache, but fall back to regular tSSS.
    existing_tsss_raw_path = next(
        (path for path in (tsss_mc_raw_path, tsss_raw_path) if os.path.exists(path)),
        None,
    )
    existing_tsss_erm_path = next(
        (path for path in (tsss_mc_erm_path, tsss_erm_path) if os.path.exists(path)),
        None,
    )
    selected_tsss_raw_path = None
    selected_tsss_erm_path = None
    extended_proj = []
    if system_upper == "MEGIN":
        try:
            if not overwrite and existing_tsss_raw_path and existing_tsss_erm_path:
                print(
                    "→ Loading existing tSSS files:\n"
                    f"  data: {existing_tsss_raw_path}\n"
                    f"  ERM:  {existing_tsss_erm_path}"
                )
                raw = mne.io.read_raw_fif(existing_tsss_raw_path, preload=True)
                raw_erm = mne.io.read_raw_fif(existing_tsss_erm_path, preload=True)
                selected_tsss_raw_path = existing_tsss_raw_path
                selected_tsss_erm_path = existing_tsss_erm_path
                if head_pos is None:
                    try:
                        head_pos = mne.chpi.read_head_pos(head_pos_path)
                    except Exception:
                        head_pos = None
            else:
                # ---------- REST / TASK DATA ----------
                if _already_has_sss(raw):
                    print("→ Input data already has Maxwell/SSS applied; skipping tSSS and caching as-is...")
                    if overwrite or not os.path.exists(tsss_raw_path):
                        raw.save(tsss_raw_path, overwrite=True)
                    selected_tsss_raw_path = tsss_raw_path
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
                                n_grad=n_grad,
                                verbose = False,
                            )
                            extended_proj.extend(proj)

                    raw = preprocessing.max_filter(
                        raw,
                        extended_proj=extended_proj,
                        calibration=calibration if os.path.exists(calibration) else None,
                        cross_talk=cross_talk if os.path.exists(cross_talk) else None,
                        st_duration=st_duration,
                        head_pos=head_pos, #If array, movement compensation will be performed.
                    )
                    raw.save(tsss_mc_raw_path, overwrite=True)
                    selected_tsss_raw_path = tsss_mc_raw_path

            # ---------- ERM ----------
            if _already_has_sss(raw_erm):
                print("→ ERM already has Maxwell/SSS applied; skipping SSS and caching as-is...")
                if selected_tsss_erm_path is None and (
                    overwrite or not os.path.exists(tsss_erm_path)
                ):
                    raw_erm.save(tsss_erm_path, overwrite=True)
                if selected_tsss_erm_path is None:
                    selected_tsss_erm_path = tsss_erm_path
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
                raw_erm.save(tsss_mc_erm_path, overwrite=True)
                selected_tsss_erm_path = tsss_mc_erm_path
                
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
        path2raw = selected_tsss_raw_path or (
            tsss_mc_raw_path if os.path.exists(tsss_mc_raw_path) else tsss_raw_path
        )
        path2raw_erm = selected_tsss_erm_path or (
            tsss_mc_erm_path if os.path.exists(tsss_mc_erm_path) else tsss_erm_path
        )
        
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

        report.add_figure(fig, title=f"PSD after bandpass filter {l_freq}-{h_freq} + downsample ({downsample} Hz)")

    # ---- Additional bad channels ----
    if additional_bads:
        raw.info["bads"].extend(additional_bads)
        raw_erm.info["bads"].extend(additional_bads)
        print(f"→ Additional bad channels were added: {additional_bads} Hz")

    # ---- SSP ----

    try:

        print("→ SSP configuration:")
        if raw_ssp_band:
            raw_ssp_desc = ", ".join([f"{low}-{high} Hz" for low, high in raw_ssp_band])
            print(f"   - Generic raw SSP: enabled | bands={raw_ssp_desc} | n_proj={num_proj_raw} per band/per available MEG type | n_grad=3, n_mag=3")
        else:
            print("   - Generic raw SSP: disabled")

        print(f"   - ECG SSP: enabled | n_proj={num_proj_ecg} per available MEG type | n_grad=3, n_mag=3 | reject=None")
        print(f"   - EOG SSP: enabled | n_proj={num_proj_eog} per available MEG type | n_grad=3, n_mag=3 | reject=None")

        if erm_ssp_band:
            if erm_ssp_band == "broad":
                erm_ssp_desc = "broadband ERM"
            else:
                erm_ssp_desc = f"ERM filtered {erm_ssp_band[0]}-{erm_ssp_band[1]} Hz"
            print(f"   - ERM SSP: enabled | {erm_ssp_desc} | n_proj={num_proj_erm} per available MEG type | n_grad=3, n_mag=3")
        else:
            print("   - ERM SSP: disabled")

        if bcglike_ssp:
            print(f"   - Ballistocardiographic-like SSP: enabled | ECG-locked | filter=1.5-8 Hz | epoch=-0.2 to 0.5 s | n_proj={num_proj_bcglike} per available MEG type | n_grad=3, n_mag=3 | reject=None")
        else:
            print("   - Ballistocardiographic-like SSP: disabled")
        #print("→ Computing ECG SSP — num of proj selected = {num_proj_ecg} ")   

        if raw_ssp_band: 
            generic_proj = []
            for i, (low, high) in enumerate(raw_ssp_band):
                #print(f"→ Computing generic (raw) SSP using filtered band {low}-{high} Hz")
                filt_raw = raw.copy().filter(l_freq=low, h_freq=high)
                # You can customize number of components per band
                proj = mne.compute_proj_raw(filt_raw, n_grad=3, n_mag=3, verbose=False)
                generic_proj.extend(proj)
            
            generic_exp_var = []
            for proj in generic_proj:
                if "explained_var" in proj:
                    generic_exp_var.append(f"{np.round(proj['explained_var'], 2)}%")

            generic_ssp_caption = f"Band-limited SSP projectors extracted after filtering the subjects MEG from {raw_ssp_band} Hz."

            fig = mne.viz.plot_projs_topomap(generic_proj, info=raw.info, show=False)
            fig.suptitle("ERM SSP projectors")
            report.add_figure(
                fig,
                title="Generic Raw Projections",
                caption = (f"{generic_ssp_caption}\n"
                           f"Explained variance: {generic_exp_var}\n"
                           f"Num of projections selected: {num_proj_raw}")
            )
            selected_generic_proj = []
            n_per_band = 6  # n_grad=3 + n_mag=3 in mne.compute_proj_raw above

            for j, (low, high) in enumerate(raw_ssp_band):
                start = j * n_per_band
                stop = start + n_per_band
                band_proj = generic_proj[start:stop]

                selected_band_proj = _select_proj_by_meg_type(
                    band_proj,
                    num_proj_raw,
                    f"Generic raw SSP {low}-{high} Hz",
                    raw.info,
                    system_upper,
                )
                selected_generic_proj.extend(selected_band_proj)

            _add_selected_projs(raw, raw_erm, selected_generic_proj)
            print("→ Applying Generic raw SSP before ECG/EOG event detection")
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

    # ---- ECG/EOG QC ----
    ecg_ev = None
    eog_ev = None
    ecg_event_count = None
    eog_event_count = None
    heart_rate_bpm = None
    blink_rate_bpm = None

    try:
        print("→ EOG/ECG artifact detection")
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, ch_name=ecg_ch)
        ecg_ev = ecg_epochs.average()
        ecg_event_count = len(ecg_epochs.events)
        duration_min = raw.times[-1] / 60.0 if raw.times[-1] > 0 else np.nan
        heart_rate_bpm = ecg_event_count / duration_min if duration_min and not np.isnan(duration_min) else np.nan
        fig = ecg_ev.plot_joint(show=False)
        _add_figure_with_caption(
            report,
            fig,
            title="ECG events",
            caption=(
                f"ECG events detected: {ecg_event_count}; "
                f"estimated heart rate: {heart_rate_bpm:.2f} events/min."
            ),
        )
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
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_ch)
        eog_ev = eog_epochs.average()
        eog_event_count = len(eog_epochs.events)
        duration_min = raw.times[-1] / 60.0 if raw.times[-1] > 0 else np.nan
        blink_rate_bpm = eog_event_count / duration_min if duration_min and not np.isnan(duration_min) else np.nan
        fig = eog_ev.plot_joint(show=False)
        _add_figure_with_caption(
            report,
            fig,
            title="EOG events",
            caption=(
                f"EOG/blink events detected: {eog_event_count}; "
                f"estimated blink rate: {blink_rate_bpm:.2f} events/min."
            ),
        )

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
        # Generic SSP projectors were already added and applied above.
        # We intentionally estimate ECG/EOG SSP on the cleaned data because
        # generic SSP may improve ECG/EOG event detection by removing large
        # low-frequency artifacts that can obscure cardiac/blink activity.
        #
        # Important MNE behavior:
        # compute_proj_ecg() and compute_proj_eog() return ALL projectors,
        # including those already stored in raw.info['projs'], followed by the
        # newly estimated ECG/EOG projectors.
        #
        # If we do not remove the pre-existing projectors, the selection code
        # below may accidentally re-select the generic SSP projectors instead
        # of the newly computed ECG/EOG SSP projectors.
        n_existing_proj = len(raw.info.get("projs", []))

        ecg_proj_all, ecg_array = mne.preprocessing.compute_proj_ecg(raw, n_grad=3, n_mag=3, reject=None) # For ECG proj, first pca is always enough
        ecg_proj = ecg_proj_all[n_existing_proj:]
        # Keep only the newly estimated ECG projectors.
        fig = mne.viz.plot_projs_joint(ecg_proj, ecg_ev, show=False)
        fig.suptitle("ECG projectors")
        exp_var = []
        for i in range(len(ecg_proj)):
            exp_var.append(str(np.round(ecg_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        ecg_caption_parts = [
            f"Explained variance:\n{', '.join(exp_var)}",
            f"Num of projections selected:\n{num_proj_ecg}",
        ]
        if ecg_event_count is not None and heart_rate_bpm is not None:
            ecg_caption_parts.append(
                f"ECG events used for projector estimation:\n{ecg_event_count}\nEstimated heart rate:\n{heart_rate_bpm:.2f} events/min"
            )
        _add_figure_with_caption(report, fig, title='ECG Projections', caption="\n".join(ecg_caption_parts))
        
        #print("→ Computing EOG SSP — num of proj selected = {num_proj_eog} ")   
        eog_proj_all, eog_array = mne.preprocessing.compute_proj_eog(raw, n_grad=3, n_mag=3, reject=None) # Default options look fine
        eog_proj = eog_proj_all[n_existing_proj:]
        # Keep only the newly estimated EOG projectors.
        fig = mne.viz.plot_projs_joint(eog_proj, eog_ev, show=False)
        fig.suptitle("EOG projectors")
        exp_var = []
        for i in range(len(eog_proj)):
            exp_var.append(str(np.round(eog_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        eog_caption_parts = [
            f"Explained variance:\n{', '.join(exp_var)}",
            f"Num of projections selected:\n{num_proj_eog}",
        ]
        if eog_event_count is not None and blink_rate_bpm is not None:
            eog_caption_parts.append(
                f"EOG/blink events used for projector estimation:\n{eog_event_count}\nEstimated blink rate:\n{blink_rate_bpm:.2f} events/min"
            )
        _add_figure_with_caption(report, fig, title='EOG Projections', caption="\n".join(eog_caption_parts))
        
        # ERM SSP can be broadband or computed from a specific band
        if erm_ssp_band: #To avoid TSSS reduncancy, this is only run if called! 
            erm_for_ssp = raw_erm.copy()
            if erm_ssp_band != "broad":
                if not isinstance(erm_ssp_band, (list, tuple)) or len(erm_ssp_band) != 2:
                    raise ValueError("erm_ssp_band must be 'broad' or [low, high]")
                low, high = erm_ssp_band
                #print(f"→ Computing ERM SSP using filtered band {low}-{high} Hz")
                erm_ssp_caption = f"Band-limited ERM SSP projectors extracted after filtering the ERM from {low} to {high} Hz."
                erm_for_ssp = raw_erm.copy().filter(l_freq=low, h_freq=high)
            else:
                #print("→ Computing ERM SSP using broadband ERM (no filtering)")
                erm_ssp_caption = "Broadband ERM SSP projectors; no additional ERM filtering was applied before SSP extraction."
            er_proj = mne.compute_proj_raw(erm_for_ssp, n_grad=3, n_mag=3, verbose=False)
            
            er_exp_var = []
            for proj in er_proj:
                if "explained_var" in proj:
                    er_exp_var.append(f"{np.round(proj['explained_var'], 2)}%")

            fig = mne.viz.plot_projs_topomap(er_proj, info=raw_erm.info, show=False)
            fig.suptitle("ERM SSP projectors")
            report.add_figure(
                fig,
                title="ERM Projections",
                caption = (f"{erm_ssp_caption}\n"
                           f"Explained variance: {er_exp_var}\n"
                           f"Num of projections selected: {num_proj_erm}")
            )

        
        
        if bcglike_ssp: #Ballistocardiographic
            #print("→ Computing SSP for ballistocardiographic-like events: bandpass 1.5-8Hz, -200-500 ms")
            filt_raw = raw.copy().filter(l_freq=1.5, h_freq=8)
            bcglike_ev = mne.preprocessing.create_ecg_epochs(filt_raw, ch_name=ecg_ch, tmin=-0.2, tmax=0.5).average()
            fig = bcglike_ev.plot_joint(show=False)
            report.add_figure(fig, title="Ballistocardiographic-like events")
            bcglike_proj_all, bcglike_array = mne.preprocessing.compute_proj_ecg(raw, n_grad=3, n_mag=3, l_freq=1.5, h_freq=8, reject=None) # For ECG proj, first pca is always enough
            bcglike_proj = bcglike_proj_all[n_existing_proj:]
            # Keep only the newly estimated BCG-like projectors.
            fig = mne.viz.plot_projs_joint(bcglike_proj, bcglike_ev, show=False)
            fig.suptitle("Ballistocardiographic-like SSP")
            exp_var = []
            
            for i in range(len(bcglike_proj)):
                exp_var.append(str(np.round(bcglike_proj[i]['explained_var'],2)))
                exp_var.append('%, ')
            _add_figure_with_caption(report, fig, title='Ballistocardiographic-like Projections', caption=f"{', '.join(exp_var)} — num of proj selected = {num_proj_bcglike}")

        selected_ecg_proj = _select_proj_by_meg_type(ecg_proj, num_proj_ecg, "ECG SSP", raw.info, system_upper)
        _add_selected_projs(raw, raw_erm, selected_ecg_proj)

        selected_eog_proj = _select_proj_by_meg_type(eog_proj, num_proj_eog, "EOG SSP", raw.info, system_upper)
        _add_selected_projs(raw, raw_erm, selected_eog_proj)

        if erm_ssp_band:
            selected_erm_proj = _select_proj_by_meg_type(er_proj, num_proj_erm, "ERM SSP", raw_erm.info, system_upper)
            _add_selected_projs(raw, raw_erm, selected_erm_proj)

        if bcglike_ssp:
            selected_bcglike_proj = _select_proj_by_meg_type( bcglike_proj, num_proj_bcglike, "Ballistocardiographic-like SSP", raw.info, system_upper,)
            _add_selected_projs(raw, raw_erm, selected_bcglike_proj)

        
            
        print("→ Applying SSP ECG,EOG and ERM SSP's")
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
        n_windows, bad_windows, metrics, thresholds, bad_times = preprocessing.detect_bad_mad_grads_mags(raw, win_length=1.0, n_mad=3,)

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

        fig = raw.compute_psd(fmax=180, verbose=False).plot(picks="data", exclude="bads", amplitude=True, show=False)

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
    # MNE Raw.save() writes FIF files. Even if the input is CTF .ds,
    # processed outputs should be cached as FIF.
    processing_suffix = "_notch_bp_SSP.fif"
    filt_path = str(Path(path2raw).with_suffix("")) + processing_suffix
    filt_erm_path = str(Path(path2raw_erm).with_suffix("")) + processing_suffix
    raw.save(filt_path, overwrite=True)
    raw_erm.save(filt_erm_path, overwrite=True)

    print(f"→ Filtered/projected raw saved at: {filt_path}")
    print(f"→ Filtered/projected ERM saved at: {filt_erm_path}")


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
    p.add_argument("--erm_ssp_band", type=str, default=None, help="ERM band for SSP: 'broad' or low-high (e.g. 10-20)")
    p.add_argument("--raw_ssp_band", type=parse_ranges, default=None, required=False, help="Frequency ranges for generic SSP as start-end pairs, e.g. 42-45,50-53")
    p.add_argument("--bcglike_ssp", action="store_true", help="Apply ssp for ballistocardiographic-like events.")
    p.add_argument("--num_proj_eog", type=int, default=1, help="Number of EOG SSP projectors to apply, respectively. Example: --num_proj_eog 1")
    p.add_argument("--num_proj_ecg", type=int, default=1, help="Number of ECG SSP projectors to apply, respectively. Example: --num_proj_ecg 1")
    p.add_argument("--num_proj_erm", type=int, default=1, help="Number of bandlimited ERM SSP projectors to apply, respectively. Example: --num_proj_erm 1")
    p.add_argument("--num_proj_raw", type=int, default=1, help="Number of bandlimited Raw SSP projectors to apply, respectively. Example: --num_proj_raw 1")
    p.add_argument("--num_proj_bcglike", type=int, default=1, help="Number of ballistocardiographic-like SSP projectors to apply, respectively. Example: --num_proj_bcglike 1 1")
    
    # Additional bad channels can be passed as spaces, commas, or both.
    # Examples:
    #   --additional_bads MEG2443 MEG1032
    #   --additional_bads MEG2443,MEG1032
    #   --additional_bads MEG2443,MEG1032 MEG0111
    p.add_argument("--additional_bads", type=str, nargs="*", default=[], help="Add any numbers of bad channels")
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
    if args.erm_ssp_band is None:
        erm_ssp_band = None
    elif args.erm_ssp_band == "broad":
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
        additional_bads=tuple(parse_channel_list(args.additional_bads)),
        n_jobs=args.n_jobs,
        erm_ssp_band=erm_ssp_band,
        raw_ssp_band=args.raw_ssp_band,
        bcglike_ssp=args.bcglike_ssp,
        num_proj_eog=args.num_proj_eog,
        num_proj_ecg=args.num_proj_ecg,
        num_proj_erm=args.num_proj_erm,
        num_proj_raw=args.num_proj_raw,
        num_proj_bcglike=args.num_proj_bcglike,
        verbose=args.verbose,
        system=args.system,
        json=args.json,
        overwrite=args.overwrite,
    )

    # ---- Save preprocessing hyperparameters ----
    if args.session is None:
        report_dir = os.path.join(
            args.root_dir,
            "derivatives",
            args.subject_id,
            args.inv_method,
            "report",
        )
    else:
        report_dir = os.path.join(
            args.root_dir,
            "derivatives",
            args.subject_id,
            args.session,
            args.inv_method,
            "report",
        )

    raw_stem = Path(file_path).stem
    save_hyperparameters(report_dir=report_dir, args=args, raw_stem=raw_stem)