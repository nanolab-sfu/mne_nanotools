import os
import mne
import pandas as pd
from scipy.stats import median_abs_deviation
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
# Last modified 27/11/2025
mne.set_log_level("ERROR")
def read_data(fname):
    """
    Read MEG data in either .fif or .ds format.
    
    Parameters
    ----------
    fname : str
        Path to the MEG file (.fif or .ds directory).
    
    Returns
    -------
    raw : mne.io.Raw
        The loaded MNE Raw object.
    """
    import os
    import mne

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    # Detect file type
    if fname.endswith('.fif'):
        print(f"→ Reading FIF file: {fname}")
        raw = mne.io.read_raw_fif(fname, preload=True)
    elif fname.endswith('.ds') and os.path.isdir(fname):
        print(f"→ Reading CTF .ds directory: {fname}")
        raw = mne.io.read_raw_ctf(fname, system_clock='ignore', preload=True)
    else:
        raise ValueError(f"Unsupported file type: {fname}. Expected .fif or .ds")

    # Fix coil types (recommended by MNE)
    mne.channels.fix_mag_coil_types(raw.info)
    return raw

def compute_head_movement_report(raw, report, subject_id, deriv_dir, system):

    # ---- Output directory ----
    out_dir = os.path.join(deriv_dir, "head_movement")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"{subject_id}_head_movement_timeseries.csv")
    out_metrics_csv = os.path.join(out_dir, f"{subject_id}_head_movement_metrics.csv")

    # ---- Compute head position (system-dependent) ----

    if system == "CTF":
        print("Using CTF head localization (HLC channels)")
        chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)

    elif system == "MEGIN":
        print("Using MEGIN cHPI pipeline")
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)

    else:
        raise ValueError(f"Unsupported system: {system}")

    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)

    # ---- Extract translation ----
    time = head_pos[:, 0]
    xyz = head_pos[:, 4:7]  # meters

    # ---- Displacement relative to start ----
    xyz0 = xyz[0, :]
    disp_xyz = xyz - xyz0

    disp_xyz_mm = disp_xyz * 1000
    distance_mm = np.linalg.norm(disp_xyz_mm, axis=1)

    # ---- Save time series CSV ----
    df = pd.DataFrame({
        "time_s": time,
        "x_mm": disp_xyz_mm[:, 0],
        "y_mm": disp_xyz_mm[:, 1],
        "z_mm": disp_xyz_mm[:, 2],
        "distance_mm": distance_mm,
    })
    df.to_csv(out_csv, index=False)

    # ---- Metrics ----
    metrics = {
        "subject_id": subject_id,
        "rms_x_mm": np.sqrt(np.mean(disp_xyz_mm[:, 0] ** 2)),
        "rms_y_mm": np.sqrt(np.mean(disp_xyz_mm[:, 1] ** 2)),
        "rms_z_mm": np.sqrt(np.mean(disp_xyz_mm[:, 2] ** 2)),
        "rms_distance_mm": np.sqrt(np.mean(distance_mm ** 2)),
        "max_abs_x_mm": np.max(np.abs(disp_xyz_mm[:, 0])),
        "max_abs_y_mm": np.max(np.abs(disp_xyz_mm[:, 1])),
        "max_abs_z_mm": np.max(np.abs(disp_xyz_mm[:, 2])),
        "max_distance_mm": np.max(distance_mm),
        "mean_distance_mm": np.mean(distance_mm),
        "median_distance_mm": np.median(distance_mm),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_metrics_csv, index=False)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(time, disp_xyz_mm[:, 0], label="X")
    ax.plot(time, disp_xyz_mm[:, 1], label="Y")
    ax.plot(time, disp_xyz_mm[:, 2], label="Z")
    ax.plot(time, distance_mm, label="Distance", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm, relative to start)")
    ax.set_title(f"Head movement: {subject_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Add to report ----
    report.add_figure(
        fig=fig,
        title=f"{subject_id} head movement",
        section="Head movement",
        tags=("head-movement", subject_id),
    )

    report.add_html(
        title=f"{subject_id} movement metrics",
        html=metrics_df.to_html(index=False, float_format="%.3f"),
        section="Head movement",
        tags=("head-movement", subject_id, "metrics"),
    )


def compute_noise_cov(er_fname, raw, calibration, cross_talk):
    # Important to apply the same preprocessing steps to empty room recording as subject recording
    er_raw = read_data(er_fname)
    er_raw.del_proj()
    er_raw = mne.preprocessing.maxwell_filter_prepare_emptyroom(er_raw, raw=raw)
    er_raw = mne.preprocessing.maxwell_filter(
        er_raw, calibration=calibration, cross_talk=cross_talk
    )
    er_raw = filter_data(er_raw)
    er_raw.add_proj(raw.info["projs"])
    noise_cov = mne.compute_raw_covariance(er_raw, tmin=0, tmax=None)
    return noise_cov


def mark_bad_channels(raw,calibration,cross_talk,):
    # Mark bad channels, necessary to avoid noise spreading in Maxwell filtering
    # Ideally, this would be done manually at the time of recording
    # auto_noisy_chs, auto_flat_chs, auto_scores = (
    #    mne.preprocessing.find_bad_channels_maxwell(raw, return_scores=True)
    # )
    auto_noisy_chs, auto_flat_chs, auto_scores = (
        mne.preprocessing.find_bad_channels_maxwell(raw, calibration=calibration,
        cross_talk=cross_talk, return_scores=True)
    )
    bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
    raw.info["bads"] = bads
    return raw


def compute_head_position(raw):
    # Compute head position indicator coil amplitudes (pretty slow)
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # Compute head position indicator coil locations (pretty slow)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    # Compute head position (much faster than the last two steps)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    return head_pos


def max_filter(raw, calibration, cross_talk, st_duration, head_pos, extended_proj):
    # Fine calibration file?
    # Crosstalk file?
    # Spatiotemporal or just spatial?
    # Detect bad channels, necessary to avoid noise spreading, ideally done manually
    if head_pos is not None:
        coord_frame = 'head'
        raw = mark_bad_channels(raw,calibration=calibration,cross_talk=cross_talk,)
    else:
        coord_frame = 'meg'
        
    # Apply Maxwell filtering with head motion correction
    raw = mne.preprocessing.maxwell_filter(
        raw,
        head_pos=head_pos,
        calibration=calibration,
        cross_talk=cross_talk,
        st_duration=st_duration,
        coord_frame=coord_frame,
        extended_proj = extended_proj,
        verbose="ERROR",
    )
    return raw


def add_ecg_projectors(raw):
    ecg_proj, ecg_array = mne.preprocessing.compute_proj_ecg(
        raw
    )  # Default options look fine
    raw.add_proj(ecg_proj)
    raw.apply_proj()
    return raw


def add_eog_projectors(raw):
    eog_proj, eog_array = mne.preprocessing.compute_proj_eog(
        raw
    )  # Default options look fine
    raw.add_proj(eog_proj)
    raw.apply_proj()
    return raw


def remove_eog_ecg(ica, raw):
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(
        raw, method="correlation"
    )  # Default method 'ctps' identified too many components as heartbeat artifacts
    ica.exclude = eog_indices + ecg_indices
    ica.apply(raw)
    return raw


def filter_data(raw, l_freq=0.1, h_freq=100, line_freqs=(50, 100, 150)):
    # l_freq and h_freq: bandpass filter
    # line_freqs: power line artifatcs (default values are because data was recorded in UK)
    meg_picks = mne.pick_types(raw.info, meg=True)  # Only filter MEG channels
    raw.notch_filter(
        freqs=line_freqs, picks=meg_picks
    )  # Use a notch filter to take out power line noise
    raw.filter(
        l_freq=l_freq, h_freq=h_freq, picks=meg_picks
    )  # Bandpass filter data (probably not any detectable high gamma activity in resting state because SNR is too low)
    return raw


def fit_ICA(raw, reject, random_state, picks, method="picard", n_components=40):
    ica = mne.preprocessing.ICA(
        n_components=40, method=method, random_state=random_state
    )
    try:
        ica.fit(raw, picks=picks, reject=reject)
    except:
        print("ICA could not run. Large environmental artifacts.\n")
    return ica


def remove_EOG_artifact(raw, ica, reject):
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, reject=reject)
    if eog_epochs.events.size != 0:
        eog_inds, scores = ica.find_bads_eog(eog_epochs)
        if len(eog_inds) != 0:
            ica.exclude.extend(eog_inds)
        else:
            print("No ICA component correlated with EOG\n")
    else:
        print("No EOG events found\n")
    return ica


def remove_ECG_artifact(raw, ica, method="ctps", tmin=-0.5, tmax=0.5):
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=tmin, tmax=tmax)
    if ecg_epochs.events.size != 0:
        ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method=method)
        if len(ecg_inds) != 0:
            ica.exclude.extend(ecg_inds)
        else:
            print("No ICA component correlated with ECG\n")
    else:
        print("No ECG events found\n")
    return ica


def do_ICA(
    raw, picks, method="picard", reject=dict(mag=5e-12, grad=4000e-13), random_state=23
):
    ica = fit_ICA(
        raw, picks=picks, method=method, reject=reject, random_state=random_state
    )
    ica = remove_EOG_artifact(raw, ica, reject=reject)
    ica = remove_ECG_artifact(raw, ica)
    ica.apply(raw)
    return raw, ica


def read_montage(file_path):

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract channels, remove spaces, and ignore the header line
    channels = [
        line.split(":")[0].replace(" ", "").strip() for line in lines if ":" in line
    ]
    channels.extend(["EOG001", "EOG002", "ECG003"])
    return channels



def detect_bad_mad_grads_mags(raw, win_length=1.0, n_mad=3, abs_gradient=True, manual_thresholds=None, picks=None,):
    """
    Detect bad windows using MAD thresholds, automatically adapting to
    the presence of MEG grad/mag channels. Works with CTF (grad only),
    Elekta (grad+mag), or magnetometer-only systems.

    Returns thresholds and metrics ONLY for sensors that exist.
    A window is rejected if ANY of:
        - max_p2p_grads
        - max_grad_grads
        - max_p2p_mags
        - max_grad_mags
    exceed thresholds.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.
    win_length : float
        Window size in seconds.
    n_mad : float
        Number of MADs above the median for auto-thresholds.
    abs_gradient : bool
        Use absolute value for gradient.
    manual_thresholds : dict or None
        {
            "p2p_grads": value,
            "grad_grads": value,
            "p2p_mags": value,
            "grad_mags": value,
        }
        If None → AUTO mode.
    """

    sfreq = raw.info["sfreq"]

    # ---------------------------------------------------------
    # Pick channel types available in this system
    # ---------------------------------------------------------
    picks_grads = mne.pick_types(raw.info, meg="grad")
    picks_mags  = mne.pick_types(raw.info, meg="mag")

    has_grads = len(picks_grads) > 0
    has_mags  = len(picks_mags) > 0

    # Load data only if those sensors exist
    data_grads = raw.get_data(picks_grads, reject_by_annotation="omit") if has_grads else None
    data_mags  = raw.get_data(picks_mags, reject_by_annotation="omit")  if has_mags  else None

    n_samples = raw.n_times
    win_samples = int(round(win_length * sfreq))
    n_windows = n_samples // win_samples

    # ---------------------------------------------------------
    # Initialize metrics only for available sensor types
    # ---------------------------------------------------------
    metrics = {"win_bounds": []}

    if has_grads:
        metrics["max_p2p_grads"] = np.zeros(n_windows)
        metrics["max_grad_grads"] = np.zeros(n_windows)

    if has_mags:
        metrics["max_p2p_mags"] = np.zeros(n_windows)
        metrics["max_grad_mags"] = np.zeros(n_windows)

    # ---------------------------------------------------------
    # Loop windows
    # ---------------------------------------------------------
    for i in range(n_windows):
        start = i * win_samples
        stop  = start + win_samples
        metrics["win_bounds"].append((start, stop))

        # Extract window segments
        if has_grads:
            seg_grads = data_grads[:, start:stop]
        if has_mags:
            seg_mags  = data_mags[:, start:stop]

        # --- P2P ---
        if has_grads:
            p2p_g = seg_grads.max(axis=1) - seg_grads.min(axis=1)
            metrics["max_p2p_grads"][i] = p2p_g.max()

        if has_mags:
            p2p_m = seg_mags.max(axis=1) - seg_mags.min(axis=1)
            metrics["max_p2p_mags"][i] = p2p_m.max()

        # --- Gradient ---
        if has_grads:
            grad_g = np.gradient(seg_grads, 1/sfreq, axis=1)
            if abs_gradient:
                grad_g = np.abs(grad_g)
            metrics["max_grad_grads"][i] = grad_g.max()

        if has_mags:
            grad_m = np.gradient(seg_mags, 1/sfreq, axis=1)
            if abs_gradient:
                grad_m = np.abs(grad_m)
            metrics["max_grad_mags"][i] = grad_m.max()

    # Thresholds based on MAD (only for available sensors)

    thresholds = {}

    if manual_thresholds is None:
        if has_grads:
            thresholds["p2p_grads"] = (
                np.median(metrics["max_p2p_grads"]) +
                n_mad * median_abs_deviation(metrics["max_p2p_grads"], scale=1)
            )
            thresholds["grad_grads"] = (
                np.median(metrics["max_grad_grads"]) +
                n_mad * median_abs_deviation(metrics["max_grad_grads"], scale=1)
            )

        if has_mags:
            thresholds["p2p_mags"] = (
                np.median(metrics["max_p2p_mags"]) +
                n_mad * median_abs_deviation(metrics["max_p2p_mags"], scale=1)
            )
            thresholds["grad_mags"] = (
                np.median(metrics["max_grad_mags"]) +
                n_mad * median_abs_deviation(metrics["max_grad_mags"], scale=1)
            )
    else:
        thresholds = manual_thresholds

    
    # Detect bad windows
    bad_windows = []
    bad_times = []
    bounds = metrics["win_bounds"]

    for i in range(n_windows):
        cond = False

        if has_grads:
            cond |= metrics["max_p2p_grads"][i] > thresholds["p2p_grads"]
            cond |= metrics["max_grad_grads"][i] > thresholds["grad_grads"]

        if has_mags:
            cond |= metrics["max_p2p_mags"][i] > thresholds["p2p_mags"]
            cond |= metrics["max_grad_mags"][i] > thresholds["grad_mags"]

        if cond:
            bad_windows.append(i)
            t0, t1 = bounds[i]
            bad_times.append((t0 / sfreq, t1 / sfreq))

    return n_windows, bad_windows, metrics, thresholds, bad_times


def plot_mad_qc(n_windows, bad_windows, metrics, thresholds, subject_name="Subject"):
    """
    Produce Brainstorm-style QC histograms for P2P and gradient values.
    Automatically suppresses plots for sensors that do not exist.
    """

    has_grads = "max_p2p_grads" in metrics
    has_mags = "max_p2p_mags" in metrics

    # Determine subplot grid size
    n_rows = int(has_grads) + int(has_mags)
    n_cols = 2  # P2P + Grad

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    row = 0

    # -------------------------
    # Gradiometers
    # -------------------------
    if has_grads:
        p2p = metrics["max_p2p_grads"] * 1e15
        grd = metrics["max_grad_grads"] * 1e15

        axes[row, 0].hist(p2p, bins=20, color="dodgerblue")
        axes[row, 0].axvline(thresholds["p2p_grads"] * 1e15, color="black", ls="--")
        axes[row, 0].set_title("P2P Grads Distribution")
        axes[row, 0].set_xlabel("fT")
        axes[row, 0].set_ylabel("Windows count")
        axes[row, 0].legend(["Threshold", "P2P range"])

        axes[row, 1].hist(grd, bins=20, color="salmon")
        axes[row, 1].axvline(thresholds["grad_grads"] * 1e15, color="black", ls="--")
        axes[row, 1].set_title("Gradient Grads Distribution")
        axes[row, 1].set_xlabel("fT")
        axes[row, 1].set_ylabel("Windows count")
        axes[row, 1].legend(["Threshold", "Gradient range"])

        row += 1

    # -------------------------
    # Magnetometers
    # -------------------------
    if has_mags:
        p2p = metrics["max_p2p_mags"] * 1e15
        grd = metrics["max_grad_mags"] * 1e15

        axes[row, 0].hist(p2p, bins=20, color="dodgerblue")
        axes[row, 0].axvline(thresholds["p2p_mags"] * 1e15, color="black", ls="--")
        axes[row, 0].set_title("P2P Mags Distribution")
        axes[row, 0].set_xlabel("fT")
        axes[row, 0].set_ylabel("Windows count")
        axes[row, 0].legend(["Threshold", "P2P range"])

        axes[row, 1].hist(grd, bins=20, color="salmon")
        axes[row, 1].axvline(thresholds["grad_mags"] * 1e15, color="black", ls="--")
        axes[row, 1].set_title("Gradient Mags Distribution")
        axes[row, 1].set_xlabel("fT")
        axes[row, 1].set_ylabel("Windows count")
        axes[row, 1].legend(["Threshold", "Gradient range"])

    plt.suptitle(f'Rejected {len(bad_windows)} / {n_windows} windows — {subject_name}')
    plt.tight_layout()
    return fig
