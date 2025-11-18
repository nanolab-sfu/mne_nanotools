#%%
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic task-free MEGIN preprocessing with ERM-SSP, tSSS, QC report, and bandwise source PSDs.

Version 0.1.0 - Last modified 15/11/2025

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
import matplotlib
matplotlib.use("Agg")  # headless mode for servers
import matplotlib.pyplot as plt
import mne
import numpy as np
from importlib import reload
from mne.report import Report

# ---- custom user modules (as in your original script) ----
import sys
sys.path.append(os.path.expanduser("~"))
from nanotools import preprocessing, postprocessing


# ----------------------------------------------------------
# Main preprocessing function
# ----------------------------------------------------------
def preprocess_subject(
    root_dir: str,
    subject_id: str,
    session:None,
    resting = 'rest1',
    rest_basename: str = "{sub}_{rest}_raw.fif",
    erm_basename: str = "{sub}_erm_raw.fif",
    tsss_dir: str = "/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023",
    st_duration: float = 10.0,
    sss_erm_st_duration: float = None,
    l_freq: float = 0.5,
    h_freq: float = 200.0,
    line_freqs: tuple = (60, 120, 180),
    downsample: int = 500,
    crop_tmin: tuple = (0., 30.),
    crop_tmax: tuple = (300., 300.),
    ecg_ch: str = "ECG003",
    eog_ch: str = ["EOG001", "EOG002"],
    reject_mag: float = 4e-12,
    reject_grad: float = 4000e-13,
    subjects_dir_name: str = "MRI/freesurfer",
    compute_bem_if_missing: bool = True,
    bem_watershed: bool = True,
    inv_method: str = "dSPM",
    snr: float = 3.0,
    bands: dict = None,
    additional_bads: tuple = (),
    n_jobs: int = 8,
    num_proj: tuple = (1,1), # ECG and EOG proj
    verbose: bool = False,
):
    """
    Generic preprocessing pipeline for MEGIN resting-state data:
    ERM-based SSP -> tSSS -> filtering -> ECG/EOG QC -> BEM/src/forward/inverse -> STC -> bandwise PSDs.

    Saves: report HTML, tSSS and filtered FIF files, head position .pos, and STC files.
    """
    # ---- Verbose control ----
    if not verbose:
        mne.set_log_level("ERROR")
    else:
        mne.set_log_level("INFO")

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
    fs_dir = os.path.join(parent_path, subjects_dir_name)   # /MRI/freesurfer/sub-XX
    if session is None:
        meg_dir = os.path.join(root_dir, "MEG", subject_id)
        deriv_dir = os.path.join(parent_path, "derivatives", subject_id)    # /derivatives/sub-XX
        head_pos_path = os.path.join(root_dir, "MEG", subject_id, subject_id +"_"+ resting +"_"+ "raw_head_pos.pos")
    else:
        meg_dir = os.path.join(root_dir, "MEG", subject_id, session)
        deriv_dir = os.path.join(parent_path, "derivatives", subject_id, session)    # /derivatives/sub-XX
        head_pos_path = os.path.join(root_dir, "MEG", subject_id, session, subject +"_"+ resting +"_"+ "raw_head_pos.pos")

    os.makedirs(deriv_dir, exist_ok=True)

    # ---- Expected inputs ----
    path2raw_rest = os.path.join(meg_dir, rest_basename.format(sub=subject_id, rest=resting))
    path2raw_erm = os.path.join(meg_dir, erm_basename.format(sub=subject_id))
    
    if not os.path.exists(path2raw_rest):
        raise FileNotFoundError(f"Resting raw file not found: {path2raw_rest}")
    if not os.path.exists(path2raw_erm):
        raise FileNotFoundError(f"ERM raw file not found: {path2raw_erm}")

    # ---- tSSS calibration files ----
    calibration = os.path.join(tsss_dir, "sss_cal.dat")
    cross_talk = os.path.join(tsss_dir, "ct_sparse.fif")
    if not os.path.exists(calibration) or not os.path.exists(cross_talk):
        print("⚠️ calibration/crosstalk not found, continuing without them (MNE will handle gracefully).")

    # ---- Report initialization ----
    report_path = os.path.join(deriv_dir, f"{subject}_{inv_method}_{resting}_QC_report.html")
    report = Report(title=f"{subject}_{inv_method}_{resting}_QC_report", raw_psd=True)

    # ---- Load data ----
    raw_rest = preprocessing.read_data(path2raw_rest)
    raw_rest.del_proj()
    raw_erm = preprocessing.read_data(path2raw_erm)
    raw_erm.del_proj()

    report.add_raw(raw=raw_rest, title="Raw Resting")

    # ---- Head position ----
    try:
        #head_pos_path = os.path.join(meg_dir,  "head_pos.pos")
        head_pos = mne.chpi.read_head_pos(head_pos_path)
    except Exception as e:
        print(f"⚠️ Could not read head_pos from {head_pos_path}: {e}")
        head_pos = None
    #if head_pos is not None:
    #   mne.chpi.write_head_pos(head_pos_path, head_pos)

    # ---- Cached tSSS paths ----
    tsss_rest_path = os.path.join(meg_dir, f"{subject}_{resting}_raw_tsss.fif")
    tsss_erm_path = os.path.join(meg_dir, f"{subject}_erm_raw_tsss.fif")

    if os.path.exists(tsss_rest_path) and os.path.exists(tsss_erm_path):
        print("→ Loading existing tSSS files...")
        raw_rest = mne.io.read_raw_fif(tsss_rest_path, preload=True)
        raw_erm = mne.io.read_raw_fif(tsss_erm_path, preload=True)
        if head_pos is None:
            try:
                head_pos = mne.chpi.read_head_pos(head_pos_path)
            except Exception:
                head_pos = None
    else:
        print("→ Applying tSSS to resting...")
        raw_rest = preprocessing.max_filter(
            raw_rest,
            calibration=calibration if os.path.exists(calibration) else None,
            cross_talk=cross_talk if os.path.exists(cross_talk) else None,
            st_duration=st_duration,
            head_pos=head_pos,
        )
        raw_rest.save(tsss_rest_path, overwrite=True)

        print("→ Applying SSS to ERM...")
        raw_erm = preprocessing.max_filter(
            raw_erm,
            calibration=calibration if os.path.exists(calibration) else None,
            cross_talk=cross_talk if os.path.exists(cross_talk) else None,
            st_duration=sss_erm_st_duration,
            head_pos=None,
        )
        raw_erm.save(tsss_erm_path, overwrite=True)
    
    # ---- PSD after tSSS ----
    fig = raw_rest.compute_psd(fmax=250,
            method="welch",
            n_fft=int(4 * raw_rest.info["sfreq"]),     # 4-second window
            n_overlap=int(2 * raw_rest.info["sfreq"]),     # 50% overlap (2-second)
            average='mean',
            window='hann').plot(picks="data", exclude="bads", amplitude=True, show=False)
    report.add_figure(fig=fig, title="PSD after tSSS")

    # ---- Temporal cropping ----
    try:
        raw_erm.crop(tmin=crop_tmin[0], tmax=crop_tmax[0])
        raw_rest.crop(tmin=crop_tmin[1], tmax=crop_tmax[1])
    except Exception as e:
        print(f"⚠️ Cropping failed: {e}")

    # ---- Filtering & notch ----
    print(f"→ Filtering {l_freq}-{h_freq} Hz, notch {line_freqs}")
    raw_rest = preprocessing.filter_data(raw_rest, l_freq=l_freq, h_freq=h_freq, line_freqs=line_freqs)
    raw_erm = preprocessing.filter_data(raw_erm, l_freq=l_freq, h_freq=h_freq, line_freqs=line_freqs)

    # ---- Downsample ----
    if downsample:
        raw_rest.resample(downsample)
        raw_erm.resample(downsample)
        fig = raw_rest.compute_psd(fmax=250,
            method="welch",
            n_fft=int(4 * raw_rest.info["sfreq"]),     # 4-second window
            n_overlap=int(2 * raw_rest.info["sfreq"]),     # 50% overlap (2-second)
            average='mean',
            window='hann').plot(picks="data", exclude="bads", amplitude=True, show=False)

        report.add_figure(fig, title=f"PSD after filters + downsample ({downsample} Hz)")

    # ---- Additional bad channels ----
    if additional_bads:
        raw_rest.info["bads"].extend(additional_bads)
        raw_erm.info["bads"].extend(additional_bads)

    # ---- ECG/EOG QC ----
    try:
        ecg_ev = mne.preprocessing.create_ecg_epochs(raw_rest, ch_name=ecg_ch).average()
        fig = ecg_ev.plot_joint(show=False)
        report.add_figure(fig, title="ECG events")
    except Exception as e:
        print(f"⚠️ ECG QC failed: {e}")
    try:
        eog_ev = mne.preprocessing.create_eog_epochs(raw_rest, ch_name=eog_ch).average()
        fig = eog_ev.plot_joint(show=False)
        report.add_figure(fig, title="EOG events")
    except Exception as e:
        print(f"⚠️ EOG QC failed: {e}")

    # ---- ERM-based SSP ----
    try:
        er_proj = mne.compute_proj_raw(raw_erm, n_grad=0, n_mag=3, verbose=True)


        #%% Create SSP ecg/eog projectors
        ecg_proj, ecg_array = mne.preprocessing.compute_proj_ecg(raw_rest,n_grad=3,n_mag=3) # For ECG proj, first pca is always enough
        fig = mne.viz.plot_projs_joint(ecg_proj, ecg_ev, show=False)
        fig.suptitle("ECG projectors")
        exp_var = []
        for i in range(len(ecg_proj)):
            exp_var.append(str(np.round(ecg_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        report.add_figure(fig, title='Ecg Projections', caption = f"{', '.join(exp_var)} — num of proj selected = {num_proj[0]}")
            
        eog_proj, eog_array = mne.preprocessing.compute_proj_eog(raw_rest,n_grad=3,n_mag=3) # Default options look fine
        fig = mne.viz.plot_projs_joint(eog_proj, eog_ev, show=False)
        fig.suptitle("EOG projectors")
        exp_var = []
        for i in range(len(eog_proj)):
            exp_var.append(str(np.round(eog_proj[i]['explained_var'],2)))
            exp_var.append('%, ')
        report.add_figure(fig, title='Eog Projections', caption = f"{', '.join(exp_var)} — num of proj selected = {num_proj[1]}")

        # EOG/ECG projections
        for i in range(0,num_proj[0]):
            raw_rest.add_proj(ecg_proj[i]) #For ECG proj, first pca is always enough
            raw_erm.add_proj(ecg_proj[i]) 

        for i in range(0,num_proj[1]):
            raw_rest.add_proj(eog_proj[i]) #For EOG proj, first pca seems enough
            raw_erm.add_proj(eog_proj[i])

        raw_rest.apply_proj()
        raw_erm.apply_proj()
    except Exception as e:
        print(f"⚠️ SSP computation failed: {e}")

    # ---- Data and noise covariance ----

    data_cov = mne.compute_raw_covariance(raw_rest, tmin=0, tmax=300)
    noise_cov = mne.compute_raw_covariance(raw_erm, tmin=0, tmax=300)

    report.add_covariance(data_cov, info=raw_rest.info, title='Data covariance')
    report.add_covariance(noise_cov, info=raw_erm.info, title='Noise covariance')

    # ---- Save filtered raw ----
    filt_path = os.path.join(deriv_dir, f"{resting}_rest_filt_proj_raw.fif")
    raw_rest.save(filt_path, overwrite=True)

    # ======================================================
    #       SOURCE MODELING (BEM / SRC / FORWARD / INVERSE)
    # ======================================================

    # ---- Coregistration metrics + report visualization ----


    trans_path = os.path.join(meg_dir, f"{subject}-trans_corr.fif")

    if os.path.exists(trans_path):
        # Load the .trans file
        trans = mne.read_trans(trans_path)

        # Compute dig → MRI distances
        distances = mne.dig_mri_distances(
            info=raw_rest.info,
            trans=trans,
            subject=subject,
            subjects_dir=fs_dir
        )

        mean_distance_mm = np.mean(distances) * 1000
        std_distance_mm  = np.std(distances)  * 1000

        note = f"Distance: {mean_distance_mm:.2f} +- {std_distance_mm:.2f} mm"

        report.add_trans(
            trans=trans_path,
            info=raw_rest.info,
            subject=subject,
            subjects_dir=fs_dir,
            plot_kwargs=dict(surfaces='head-dense',
            mri_fiducials=True, meg={"helmet": 0.1, "sensors": 0.1, "ref": 1}),
            title=f'Coregistration._{note}',
            alpha=1
        )
    else:
        print(f"⚠️ Missing trans file: {trans_path}")

    # ---- BEM ----
    bem_path = os.path.join(fs_dir, subject, "bem", f"{subject}-5120-5120-5120-bem-sol.fif")
    bem_dir = os.path.join(fs_dir, "bem")
    src_path = os.path.join(deriv_dir, f"src.fif")

    if compute_bem_if_missing and not os.path.exists(bem_path):
        os.makedirs(bem_dir, exist_ok=True)
        conductivity = (0.3,)   # Single layer for MEG
        model = mne.make_bem_model(subject=subject, ico=4, #The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280. If None, no subsampling is applied.
                            conductivity=conductivity, 
                            subjects_dir=fs_dir) #bem conductivity model
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_path, bem_sol)
        if bem_watershed:
            print("→ Creating watershed BEM (if missing)...")
            try:
                mne.bem.make_watershed_bem(subject=subject, subjects_dir=fs_dir, overwrite=True)
                mne.bem.make_scalp_surfaces(subject=subject, subjects_dir=fs_dir, overwrite=True) #Creates the high resolution -head-dense.fif

            except Exception as e:
                print(f"⚠️ Watershed BEM failed: {e}")

    if not os.path.exists(src_path):
        print("→ Setting up source space...")
        src = mne.setup_source_space(subject=subject, subjects_dir=fs_dir, add_dist="patch")
        src.save(src_path, overwrite=True)
    else:
        src = mne.read_source_spaces(src_path)

    # ---- BEM / alignment QC ----
    try:
        # Plot BEM (2D slices)
        fig = mne.viz.plot_bem(subject=subject, subjects_dir=fs_dir, src=src)
        report.add_figure(fig, title="Sources on BEM")

        # Plot 3D alignment (no 'show' kwarg)
        fig = mne.viz.plot_alignment(
            subject=subject,
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

    # ---- Forward ----

    try:
        print('Foward Solutions')
        fwd = mne.make_forward_solution(
                    raw_rest.info, trans=trans_path, src=src, bem=bem_path, meg=True, eeg=False, mindist=0.0, n_jobs=n_jobs)
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)

    except Exception as e:
        print(f"⚠️ Foward solution failed: {e}")

    # ---- Inverse ----

    # Choose the STC directory based on session

    stc_dir = os.path.join(deriv_dir, inv_method + '_stc')

    # Ensure directory exists
    os.makedirs(stc_dir, exist_ok=True)

    # Base filename for the STC
    stc_base = f"{inv_method}_{resting}_stc"
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

            # Silence annoying joblib warnings
            os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
            os.environ["JOBLIB_NO_MPI"] = "1"

            # if STC does not exist: compute it
            if not os.path.exists(stc_path + "-lh.stc"):
                print("→ Forward solution...")
                print(f"→ Inverse operator ({inv_method})...")

                inv = mne.minimum_norm.make_inverse_operator(
                    raw_rest.info, fwd, noise_cov, loose=0.2, depth=0.8
                )

                lambda2 = 1.0 / (snr ** 2)

                stc = mne.minimum_norm.apply_inverse_raw(
                    raw_rest, inv, lambda2=lambda2, method=inv_method
                )

                try:
                    stc.save(stc_path, overwrite=True)
                    print(f"→ STC saved at {stc_path}")
                except Exception as e:
                    print(f"⚠️ Could not save STC: {e}")

            else:
                print(f"→ Reading existing STC ({inv_method})...")
                stc = mne.read_source_estimate(stc_path)

        # ------------------ BEAMFORMER ------------------
        else:
            if not os.path.exists(stc_path + "-lh.stc"):
                print("Computing Source Estimation Beamformer...")
                start, stop = raw_rest.time_as_index([crop_tmin[0], crop_tmax[0]])

                #Whats all this hyperparameters?! Make it more clear to you and everyone
                filters = mne.beamformer.make_lcmv(
                    raw_rest.info,
                    fwd,
                    data_cov,
                    reg=0.05, #whats the regularization?
                    noise_cov=noise_cov,
                    pick_ori="max-power",
                    weight_norm="unit-noise-gain",
                    rank='info'
                )

                stc = mne.beamformer.apply_lcmv_raw(raw_rest, filters,
                                                    start=start, stop=stop)

                try:
                    stc.save(stc_path, overwrite=True)
                    print(f"→ STC saved at {stc_path}")
                except Exception as e:
                    print(f"⚠️ Could not save STC: {e}")

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
                        subject=subject, 
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
                                          tmin=0, tstep=.25, subject=subject)

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
        output_path = os.path.join(deriv_dir, f"PSD_band_dist_{subject}_{inv_method}_{resting}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    p = argparse.ArgumentParser(description="Preprocess MEGIN task-free MEG data with ERM-SSP, tSSS, QC, and source modeling.")
    p.add_argument("--root_dir", required=True, type=str)
    p.add_argument("--subject_id", required=True, type=str)
    p.add_argument("--session", default=None, required=True, help="Session by date or order (e.g., 01012020 or ses-1)")
    p.add_argument("--resting", default='rest1', required=True, help="name or the resating state recording to porcess")
    p.add_argument("--rest_basename", type=str, default="{sub}_{rest}_raw.fif")
    p.add_argument("--erm_basename", type=str, default="{sub}_erm_raw.fif")
    p.add_argument("--tsss_dir", type=str, default="/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023")
    p.add_argument("--st_duration", type=float, default=10.0)
    p.add_argument("--sss_erm_st_duration", type=float, default=None)
    p.add_argument("--l_freq", type=float, default=0.5)
    p.add_argument("--h_freq", type=float, default=200.0)
    p.add_argument("--line_freqs", type=float, nargs="*", default=[60, 120, 180])
    p.add_argument("--downsample", type=int, default=500)
    p.add_argument("--crop_tmin", type=float, nargs=2, default=[0.0, 30.0])
    p.add_argument("--crop_tmax", type=float, nargs=2, default=[300.0, 300.0])
    p.add_argument("--ecg_ch", type=str, default="ECG003")
    p.add_argument("--eog_ch", type=str, default="EOG001")
    p.add_argument("--reject_mag", type=float, default=4e-12)
    p.add_argument("--reject_grad", type=float, default=4000e-13)
    p.add_argument("--subjects_dir_name", type=str, default="MRI/freesurfer")
    p.add_argument("--compute_bem_if_missing", action="store_true", default=True)
    p.add_argument("--no_compute_bem_if_missing", dest="compute_bem_if_missing", action="store_false")
    p.add_argument("--bem_watershed", action="store_true", default=True)
    p.add_argument("--no_bem_watershed", dest="bem_watershed", action="store_false")
    p.add_argument("--inv_method", type=str, default="beamformer", choices=["MNE", "dSPM", "sLORETA","beamformer"])
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--num_proj", type=int, default=[1,1])
    # additional_bads como lista
    p.add_argument("--additional_bads", type=str, nargs="*", default=[])
    p.add_argument("--verbose", action="store_true", help="Enable verbose MNE output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_subject(
        root_dir=args.root_dir,
        subject_id=args.subject_id,
        session=args.session,
        resting=args.resting,
        rest_basename=args.rest_basename,
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
        subjects_dir_name=args.subjects_dir_name,
        compute_bem_if_missing=args.compute_bem_if_missing,
        bem_watershed=args.bem_watershed,
        inv_method=args.inv_method,
        snr=args.snr,
        additional_bads=tuple(args.additional_bads),
        n_jobs=args.n_jobs,
        num_proj=args.num_proj,
        verbose=args.verbose,
    )