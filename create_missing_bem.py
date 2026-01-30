#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteratively create missing BEMs for all FreeSurfer subjects.

This script checks whether a BEM solution exists for each subject
in the FreeSurfer subjects directory. If not, it creates:
  1) Watershed BEM surfaces
  2) High-resolution scalp surface
  3) Single-layer MEG BEM solution (5120 vertices)

Version 0.1.0 - Last modified 03/01/2026

Usage example:
    python create_missing_bem.py \
        --root_dir /Users/isaant/Documents/PosDoc/Projects/BRHRS \
        --subjects_dir_name MRI/freesurfer
"""

import os
import argparse
import mne
from mne.bem import make_watershed_bem, make_scalp_surfaces


def create_bem_if_missing(subject, subjects_dir, overwrite=False):
    """
    Create BEM surfaces and solution if missing for a given subject.
    """
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    bem_sol_path = os.path.join(
        bem_dir, f"{subject}-5120-5120-5120-bem-sol.fif"
    )

    if os.path.exists(bem_sol_path) and not overwrite:
        print(f"✅ BEM already exists for {subject}, skipping.")
        return

    print(f"\n Processing BEM for {subject}")

    try:
        os.makedirs(bem_dir, exist_ok=True)

        outer_skin = os.path.join(bem_dir, "outer_skin.surf")
        if not os.path.exists(outer_skin) or overwrite:
            print("  → Creating watershed BEM surfaces...")
            make_watershed_bem(
                subject=subject,
                subjects_dir=subjects_dir,
                overwrite=overwrite
            )

        print("  → Creating high-resolution scalp surface...")
        make_scalp_surfaces(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=overwrite
        )

        print("  → Creating BEM model (single-layer MEG)...")
        model = mne.make_bem_model(
            subject=subject,
            subjects_dir=subjects_dir,
            conductivity=(0.3,),
            ico=4  # 5120 vertices
        )

        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_sol_path, bem_sol)

        print(f"  ✅ BEM written: {bem_sol_path}")

    except Exception as e:
        print(f"⚠️  Failed to create BEM for {subject}: {e}")


def run(root_dir, subjects_dir_name, overwrite=False):
    root_dir = os.path.abspath(root_dir)
    subjects_dir = os.path.join(root_dir, subjects_dir_name)

    if not os.path.isdir(subjects_dir):
        raise RuntimeError(f"Subjects dir not found: {subjects_dir}")

    subjects = sorted(
        s for s in os.listdir(subjects_dir)
        if os.path.isdir(os.path.join(subjects_dir, s))
        and not s.startswith(".")
    )

    print(f" Found {len(subjects)} FreeSurfer subjects")

    for subject in subjects:
        create_bem_if_missing(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=overwrite
        )

    print("\n BEM check/creation finished.")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Create missing BEMs for FreeSurfer subjects"
    )
    p.add_argument(
        "--root_dir", required=True,
        help="Root directory containing MRI/freesurfer"
    )
    p.add_argument(
        "--subjects_dir_name", default="MRI/freesurfer",
        help="Relative path to FreeSurfer subjects dir inside root_dir"
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute BEM even if it already exists"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        root_dir=args.root_dir,
        subjects_dir_name=args.subjects_dir_name,
        overwrite=args.overwrite
    )