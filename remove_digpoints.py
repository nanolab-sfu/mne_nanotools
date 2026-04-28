#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated pipeline to remove digitilized points under the fiducials plane
and generate a report.

Version 0.1.0 - Last modified 09/12/2025

Usage example:
    python remove_digipoints.py \
        --root_dir /Users/isaant/Documents/PosDoc/Projects/Variability_project/NVAR-data \
        --subject_id sub_NVAR008 \
        --session 251016 \
        --task rest1

@author: isaant
"""

import os
import argparse
import mne
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from mne.io.constants import FIFF

import sys
sys.path.append(os.path.expanduser("~"))
from mne_nanotools import preprocessing


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def run_remove_digitilized_points(root_dir, subject_id, session, task, run):

    # ====== BUILD PATHS ======================================================
    root_dir = os.path.abspath(root_dir)
    meg_dir = os.path.join(root_dir, "MEG", subject_id, session)
    meg_nested = os.path.join(meg_dir, "meg")
    print(meg_nested)
    meg_dir = meg_nested if os.path.isdir(meg_nested) else meg_dir

    basename = f"{subject_id}_{session}_task-{task}_{run}_meg.fif"
    path2raw = os.path.join(meg_dir, basename)

    if not os.path.exists(path2raw):
        raise FileNotFoundError(f"Task raw file not found:{path2raw}")

    # ====== LOAD DATA ========================================================
    raw = preprocessing.read_data(path2raw)
    dig = raw.info["dig"]

    # =====================================================================
    # EXTRACT LPA / RPA / NASION FROM DIGITIZATION
    # =====================================================================
    cardinals = [d for d in dig if d["kind"] == FIFF.FIFFV_POINT_CARDINAL]

    lpa = [d for d in cardinals if d["ident"] == FIFF.FIFFV_POINT_LPA][0]["r"]
    rpa = [d for d in cardinals if d["ident"] == FIFF.FIFFV_POINT_RPA][0]["r"]
    nas = [d for d in cardinals if d["ident"] == FIFF.FIFFV_POINT_NASION][0]["r"]

    # =====================================================================
    # BUILD PLANE NORMAL
    # =====================================================================
    normal = np.cross(rpa - lpa, nas - lpa)
    normal /= np.linalg.norm(normal)

    if normal[1] < 0:
        normal = -normal

    # =====================================================================
    # CLASSIFY DIG POINTS
    # =====================================================================
    accepted = []
    rejected = []
    hpi = []

    for d in dig:
        point = d["r"]
        kind = d["kind"]

        if kind == FIFF.FIFFV_POINT_HPI:
            hpi.append(point)
            continue
        if kind == FIFF.FIFFV_POINT_CARDINAL:
            continue

        val = np.dot(normal, point - lpa)

        if val >= 0:
            accepted.append(point)
        else:
            rejected.append(point)

    accepted = np.array(accepted)
    rejected = np.array(rejected)
    hpi = np.array(hpi)

    # =====================================================================
    # BUILD ORTHONORMAL SQUARE PLANE FOR PLOTTING
    # =====================================================================
    u = (rpa - lpa)
    u /= np.linalg.norm(u)

    v_temp = nas - lpa
    v_temp -= np.dot(v_temp, u) * u
    v = v_temp / np.linalg.norm(v_temp)

    plane_size = 0.12
    s = np.linspace(-plane_size, plane_size, 20)
    t = np.linspace(-plane_size, plane_size, 20)
    S, T = np.meshgrid(s, t)
    plane = lpa + S[..., None] * u + T[..., None] * v

    X = plane[..., 0]
    Y = plane[..., 1]
    Z = plane[..., 2]

    # =====================================================================
    # PLOT FIGURE
    # =====================================================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Digitized Headshape and Fiducial Plane")

    # ax.plot_surface(X, Y, Z, alpha=0.2, color="gray")

    if len(accepted):
        ax.scatter(accepted[:, 0], accepted[:, 1], accepted[:, 2], c="green", s=10, label="Accepted")

    if len(rejected):
        ax.scatter(rejected[:, 0], rejected[:, 1], rejected[:, 2], c="red", s=10, label="Rejected")

    ax.scatter(*lpa, c="blue", s=80, marker="o", label="LPA")
    ax.scatter(*rpa, c="blue", s=80, marker="o", label="RPA")
    ax.scatter(*nas, c="blue", s=80, marker="o", label="Nasion")

    if len(hpi):
        ax.scatter(hpi[:, 0], hpi[:, 1], hpi[:, 2], c="yellow", s=60, marker="^", label="HPI")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.view_init(elev=20, azim=60)
    ax.set_box_aspect([1, 1, 1])

    # =====================================================================
    # CREATE REPORT (PDF)
    # =====================================================================
    report_path = os.path.join(
        meg_dir,
        f"{subject_id}_{session}_{task}_digReport.pdf"
    )

    with PdfPages(report_path) as pdf:

        # Page 1: 3D plot
        pdf.savefig(fig)

        # Page 2: Summary text
        fig2 = plt.figure(figsize=(8, 10))
        plt.axis("off")

        text = f"""
        Digitization Report
        -----------------------------

        Subject:      {subject_id}
        Session:      {session}
        Task:      {task}

        Fiducials (meters):
            LPA:  {lpa}
            RPA:  {rpa}
            NAS:  {nas}

        Point Counts:
            HPI coils:           {len(hpi)}
            Accepted headshape:  {len(accepted)}
            Rejected headshape:  {len(rejected)}

        Total points: {len(accepted) + len(rejected) + len(hpi)}

        The figure on page 1 shows:
            - Green: accepted points (kept)
            - Red: rejected points (removed)
            - Yellow: HPI coils
            - Blue: fiducials used to build plane

        """

        plt.text(0.01, 0.99, text, va="top", fontsize=12)
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"\nSaved digitization report:\n{report_path}")

    # =====================================================================
    # BUILD MONTAGE AND SAVE CLEANED FILE
    # =====================================================================
    mont = mne.channels.make_dig_montage(
        nasion=nas,
        lpa=lpa,
        rpa=rpa,
        hsp=accepted if len(accepted) else None,
        hpi=hpi if len(hpi) else None,
        coord_frame="head"
    )

    raw.set_montage(mont, on_missing="ignore")

    output_file = path2raw.replace(".fif", "_digFiltered.fif")
    raw.save(output_file, overwrite=True)

    print(f"Saved cleaned file:\n{output_file}")


# =====================================================================
# COMMAND-LINE INTERFACE
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove digitized points under LPA–RPA–NAS plane")

    parser.add_argument("--root_dir", required=True, help="Root project directory")
    parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-BRS0035)")
    parser.add_argument("--session", required=True, help="Session folder (e.g., 251016)")
    parser.add_argument("--task", default="rest", help="Task run name (rest)")
    parser.add_argument("--run", default="run-1", help="Task iteration number (e.g., run-1)")

    args = parser.parse_args()

    run_remove_digitilized_points(
        root_dir=args.root_dir,
        subject_id=args.subject_id,
        session=args.session,
        task=args.task,
        run=args.run,
    )
