#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 19:43:02 2026

@author: isaant
"""

import mne
import numpy as np
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

