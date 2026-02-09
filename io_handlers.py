#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 19:43:02 2026

@author: isaant
"""

import re
import json
import mne
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from mne.io.constants import FIFF



def extract_dig_points(info):
    """Return dict with fiducials (nasion/lpa/rpa), hsp (Nx3), hpi (Mx3)."""
    dig = info.get("dig", None)
    if dig is None:
        raise RuntimeError("No dig points found in info['dig'].")

    fids = {}
    hsp = []
    hpi = []

    for d in dig:
        kind = d["kind"]
        ident = d["ident"]
        r = np.array(d["r"], float)

        if kind == FIFF.FIFFV_POINT_CARDINAL:
            if ident == FIFF.FIFFV_POINT_NASION:
                fids["nasion"] = r
            elif ident == FIFF.FIFFV_POINT_LPA:
                fids["lpa"] = r
            elif ident == FIFF.FIFFV_POINT_RPA:
                fids["rpa"] = r

        elif kind == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(r)

        elif kind == FIFF.FIFFV_POINT_HPI:
            hpi.append(r)

    hsp = np.array(hsp, float) if len(hsp) else np.zeros((0, 3))
    hpi = np.array(hpi, float) if len(hpi) else np.zeros((0, 3))

    missing = {"nasion", "lpa", "rpa"} - set(fids.keys())
    if missing:
        raise RuntimeError(f"Missing fiducials: {missing}")

    return fids, hsp, hpi




def inject_dig_into_raw(raw, fids, hsp, hpi=None):
    """Inject fiducials + headshape (+ optional HPI) into raw.info via montage."""
    kwargs = dict(
        nasion=fids["nasion"],
        lpa=fids["lpa"],
        rpa=fids["rpa"],
        hsp=hsp,
        coord_frame="head",
    )

    # Only add hpi if you actually have it
    if hpi is not None and len(hpi):
        kwargs["hpi"] = hpi

    montage = mne.channels.make_dig_montage(**kwargs)
    raw.set_montage(montage, on_missing="ignore")
    return raw

def strip_bids_prefix(path: str) -> str:
    return path.replace("bids::", "")


def strip_nii_suffix(fname: str) -> str:
    """Remove .nii or .nii.gz"""
    return re.sub(r"\.nii(\.gz)?$", "", fname)


def extract_bids_session(path: str, fallback=None) -> str:
    m = re.search(r"(ses-[^_/]+)", path)
    return m.group(1) if m else fallback


def fiff_ident_from_label(label: str):
    label = label.upper()
    if label == "LPA":
        return FIFF.FIFFV_POINT_LPA
    if label in ("NAS", "NASION"):
        return FIFF.FIFFV_POINT_NASION
    if label == "RPA":
        return FIFF.FIFFV_POINT_RPA
    return None


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def scannerRAS_to_tkrRAS_matrix(t1_mgz: str) -> np.ndarray:
    """Return scannerRAS(mm) → tkrRAS(mm) transform from FreeSurfer T1.mgz"""
    mgz = nib.load(t1_mgz)
    vox2ras = mgz.header.get_vox2ras()
    vox2tkr = mgz.header.get_vox2ras_tkr()
    return vox2tkr @ np.linalg.inv(vox2ras)


def nifti_ijk_to_tkrRAS(
    ijk: np.ndarray,
    nifti_affine: np.ndarray,
    scannerRAS_to_tkrRAS: np.ndarray,
) -> np.ndarray:
    """NIfTI ijk → tkrRAS (mm)"""
    ras_scanner_mm = apply_affine(nifti_affine, ijk)
    return (scannerRAS_to_tkrRAS @ np.r_[ras_scanner_mm, 1.0])[:3]