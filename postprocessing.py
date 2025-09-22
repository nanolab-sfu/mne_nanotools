"""
postporcessing.py - Spectral analysis utilities for MEG/EEG source data using MNE-Python.

Provides functions for computing power spectral density (PSD) and extracting frequency-specific
power per vertex from SourceEstimate objects, as well as morphing results to a standard space.

Author: isaant
"""

import mne
from scipy.signal import welch
from joblib import Parallel, delayed
import numpy as np

def _compute_psd_single(vertex_ts, sfreq, window_samples, overlap_samples):
    """
    Compute PSD for a single vertex time series using Welch's method.

    Parameters
    ----------
    vertex_ts : ndarray, shape (n_times,)
        Time series of a single vertex.
    sfreq : float
        Sampling frequency in Hz.
    window_samples : int
        Number of samples per window.
    overlap_samples : int
        Number of samples to overlap between segments.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    psd : ndarray
        Power spectral density for the vertex.
    """
    freqs, psd = welch(
        vertex_ts,
        fs=sfreq,
        window="hann",
        nperseg=window_samples,
        noverlap=overlap_samples,
        scaling="density"
    )
    #psd_db = 10 * np.log10(psd + 1e-20)  # Convert to dB scale
    return freqs, psd

def PSD_per_vertex_parallel(stc, bands, n_jobs=-1):
    """
    Compute normalized PSD and average band power for each vertex using parallel processing.

    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate object containing vertex time series data.
    bands : dict
        Dictionary of frequency bands, with keys as band names and values as (fmin, fmax) tuples.
        Example: {'alpha': (8, 13), 'beta': (13, 30)}
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 uses all available CPUs.

    Returns
    -------
    psd_normalized : ndarray, shape (n_vertices, n_freqs)
        Normalized power spectral density for each vertex.
    band_powers : dict
        Dictionary with the same keys as `bands`, where each value is an array of shape (n_vertices,)
        containing the average power for that band.
    """
    data = stc.data  # shape: (n_vertices, n_times)
    sfreq = stc.sfreq
    window_samples = int(4 * sfreq)
    overlap_samples = window_samples // 2

    # Compute PSD per vertex in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_psd_single)(vertex_ts, sfreq, window_samples, overlap_samples)
        for vertex_ts in data
    )

    # Extract frequency axis and PSDs
    freqs = results[0][0]
    psd_all = np.array([psd for _, psd in results])

    # Normalize PSD
    psd_normalized = psd_all / psd_all.sum(axis=1, keepdims=True)

    # Compute average band power for each frequency band
    band_powers = {
        band: psd_normalized[:, (freqs >= fmin) & (freqs <= fmax)].mean(axis=1)
        for band, (fmin, fmax) in bands.items()
    }

    return psd_all, freqs, psd_normalized, band_powers


def stc_per_band(morph, power, stc, subject):
    """
    Creates a SourceEstimate object for a specific frequency band and applies morphing.

    Parameters:
        morph: mne.SourceMorph
            The morph object used to morph the SourceEstimate to a common space.
        power: ndarray
            Power values for the frequency band, corresponding to vertices.
        stc: mne.SourceEstimate
            SourceEstimate object containing the original vertex and subject information.
        subject: str
            Subject identifier in the FreeSurfer directory.

    Returns:
        stc_band_morph: mne.SourceEstimate
            Morphed SourceEstimate object for the frequency band.
    """
    # Create the SourceEstimate for the band
    stc_band = mne.SourceEstimate(
        data=power, vertices=stc.vertices, tmin=0, tstep=0.25, subject=subject
    )

    # Apply morphing to the SourceEstimate
    stc_band_morph = morph.apply(stc_band)
    return stc_band_morph


def generate_brain_screenshot(stc_band_morph, views, power, surfer_kwargs):
    """
    Generates a screenshot of the brain visualization for a given SourceEstimate.

    Parameters:
        stc_band_morph: mne.SourceEstimate
            The morphed SourceEstimate object to be visualized.
        views: list or str
            View(s) to display (e.g., 'lateral', 'medial', etc.).
        power: ndarray
            Power values to define the color limits (clim).
        surfer_kwargs: dict
            Additional keyword arguments for the brain visualization.

    Returns:
        img: ndarray
            Screenshot image of the brain visualization.
    """
    # Update visualization parameters
    surfer_kwargs["views"] = views
    surfer_kwargs["hemi"] = "lh"
    if views == "dorsal":
        surfer_kwargs["hemi"] = "both"
    clim = dict(kind="value", lims=[0, max(power) / 2, max(power)])  # Colorband limits
    brain = stc_band_morph.plot(
        **surfer_kwargs, clim=clim
    )  # Plot the brain with the specified clim and additional arguments
    img = brain.screenshot()  # Capture the screenshot
    brain.close()  # Close the interactive brain object

    return img


def brains_plot(i, band, axes, img_lateral, img_medial, img_dorsal):
    """
    Places brain images into subplots and labels the frequency band.

    Parameters:
        i: int
            Row index in the subplot grid.
        band: str
            Name of the frequency band to display as a label.
        axes: ndarray
            Array of subplot axes.
        img_lateral: ndarray
            Image array for the lateral view of the brain.
        img_medial: ndarray
            Image array for the medial view of the brain.

    Returns:
        None
    """
    # Place the images into the subplots
    axes[i, 0].imshow(img_lateral)
    axes[i, 0].axis("off")
    axes[i, 1].imshow(img_medial)
    axes[i, 1].axis("off")
    axes[i, 2].imshow(img_dorsal)
    axes[i, 2].axis("off")

    # Add the frequency band name as a vertical title in the first column
    axes[i, 0].text(
        -0.1,
        0.5,
        band,
        fontsize=14,
        va="center",
        ha="right",
        transform=axes[i, 0].transAxes,
        rotation=90,
    )


def calculate_parcellation_centroids(labels, src):
    """
    Calculate the centroid coordinates for each parcel in a parcellation.

    Parameters:
        labels (list of mne.Label): List of labels from the parcellation.
        src (list of dict): Source space loaded with MNE.

    Returns:
        np.ndarray: Coordinates (x, y, z) of the centroids for each parcel.
    """

    centroids = []

    for label in labels:
        # Get the vertices associated with the label
        vertices = label.vertices

        # Determine the hemisphere and retrieve the vertex coordinates
        if label.hemi == "lh":  # Left hemisphere
            coords = src[0]["rr"][vertices]
        else:  # Right hemisphere
            coords = src[1]["rr"][vertices]

        # Compute the centroid (mean of the coordinates)
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)  # [x,y,z] = x:left/right(-/+), y: antero-posterior

    # Convert to a numpy array
    return np.array(centroids)


def bp_gen_band(parc_ts, sfreq, band):
    """
    Generator that applies a band-pass filter for a specific frequency band.

    Parameters:
        parc_ts (dict,ndarray): Dict of Array of time series with shape (Epochs,n_labels, n_samples).
                                If single array with shape (n_labels, n_samples), yeild wont be used
        sfreq (float): Sampling frequency.
        band (tuple): Frequency range for the band-pass filter (low, high).

    Yields:
        ndarray: Filtered signal for the specified frequency band.
    """
    what_type = type(parc_ts)
    if what_type == dict:
        for ts in parc_ts:  # Iterate over the first axis (signals)
            yield mne.filter.filter_data(ts, sfreq, band[0], band[1])
    else:
        return mne.filter.filter_data(parc_ts, sfreq, band[0], band[1])[
            np.newaxis, :, :
        ]


def compute_band_correlations(parc_ts, sfreq, bands):
    """
    Computes correlation matrices for multiple frequency bands.

    Parameters:
        parc_ts (ndarray): Array of time series with shape (n_labels, n_samples).
        sfreq (float): Sampling frequency.
        bands (dict): Dictionary with band names and frequency ranges.

    Returns:
        dict: Dictionary containing correlation matrices for each band.
    """
    correlations = {}

    for band_name, band in bands.items():
        # Generate filtered signals for the current band
        filtered_gen = mne.filter.filter_data(parc_ts, sfreq, band[0], band[1])[
            np.newaxis, :, :
        ]

        # Compute the envelope correlation
        corr_obj = mne_connectivity.envelope_correlation(
            filtered_gen, orthogonalize="pairwise"
        )

        # Combine correlations and get dense data
        corr = corr_obj.combine()
        correlations[band_name] = corr.get_data(output="dense")[:, :, 0]

    return correlations


def plot_corr(corr):
    for band_name, band in corr.items():
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        clim = np.percentile(band, [5, 95])
        # sns.heatmap(band, cmap="jet", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True, square=True, annot=True, fmt=".2f")
        sns.heatmap(band, cmap="mako", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True)
        # ax.imshow(band, cmap="jet", clim = np.percentile(band, [5, 95]))
        fig.suptitle("pairwise correlation, " + band_name)
    plt.show()


def plot_corr_singlefig(corr):
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True
    )  # 2 filas, 3 columnas
    axes = axes.flatten()  # Aplanar la matriz de ejes para iterar más fácilmente

    for idx, (band_name, band) in enumerate(corr.items()):
        ax = axes[idx]  # Seleccionar el subplot correspondiente
        clim = np.percentile(band, [5, 95])
        sns.heatmap(band, cmap="mako", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True)
        ax.set_title("Pairwise correlation: " + band_name)

    # Si hay menos de 6 subplots, ocultar los restantes
    for idx in range(len(corr), len(axes)):
        fig.delaxes(axes[idx])

    plt.show()


def plot_corr_test(corr, labels, out_dir):
    """
    Plot connectivity matrices with optional reordering and boundary marking.

    Parameters:
        corr (dict): Dictionary of connectivity matrices (e.g., {band_name: matrix}).
        labels (list): List of MNE Label objects.
    """
    Vis = ["VisCen", "VisPer"]
    change_of_net_pos = []
    current_net = re.search(r"H_(.*?)_", str(labels[0])).group(1)
    all_nets = [re.search(r"H_(.*?)_", str(labels[0])).group(1)]
    for i, label in enumerate(labels[1:]):
        match = re.search(r"H_(.*?)_", str(label)).group(1)
        if current_net != match:
            current_net = match
            change_of_net_pos.append(i + 1)
            all_nets.append(match)
            # if len(change_of_net_pos) > 3 and all_nets[1] == all_nets[-1]:
            #   break

    change_of_supnet_pos = []
    current_net = re.search(r"H_(.*?)_", str(labels[0])).group(1)[:-1]
    all_supnets = [re.search(r"H_(.*?)_", str(labels[0])).group(1)[:-1]]
    for i, label in enumerate(labels[1:]):
        match = re.search(r"H_(.*?)_", str(label)).group(1)[:-1]
        if match == Vis[0] or match == Vis[1]:
            match = "Vis"
        if current_net != match:
            current_net = match
            change_of_supnet_pos.append(i + 1)
            all_supnets.append(match)

    for idx, (band_name, band) in enumerate(corr.items()):

        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        clim = np.percentile(band, [5, 95])
        sns.heatmap(band, cmap="mako", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True)
        ax.set_title(f"Pairwise correlation: {band_name}")

        ax.hlines(
            100, *ax.get_xlim(), colors="white", linewidth=2.0, linestyles="dashed"
        )
        ax.vlines(
            100, *ax.get_ylim(), colors="white", linewidth=2.0, linestyles="dashed"
        )

        start = 0
        end = 200
        for i in range(len(change_of_net_pos) - 1):
            ax.hlines(
                change_of_net_pos[i],
                start,
                change_of_net_pos[i + 1],
                colors="red",
                linewidth=2.0,
            )
            ax.vlines(
                change_of_net_pos[i],
                start,
                change_of_net_pos[i + 1],
                colors="red",
                linewidth=2.0,
            )
            start = change_of_net_pos[i]
        ax.hlines(change_of_net_pos[-1], start, end, colors="red", linewidth=2.0)
        ax.vlines(change_of_net_pos[-1], start, end, colors="red", linewidth=2.0)

        start = 0
        for i in range(len(change_of_supnet_pos) - 1):
            ax.hlines(
                change_of_supnet_pos[i],
                start,
                change_of_supnet_pos[i + 1],
                colors="violet",
                linewidth=2.0,
            )
            ax.vlines(
                change_of_supnet_pos[i],
                start,
                change_of_supnet_pos[i + 1],
                colors="violet",
                linewidth=2.0,
            )
            start = change_of_supnet_pos[i]
        ax.hlines(change_of_supnet_pos[-1], start, end, colors="violet", linewidth=2.0)
        ax.vlines(change_of_supnet_pos[-1], start, end, colors="violet", linewidth=2.0)

        label_pos = []
        start = 0
        for i in range(np.ceil(len(all_nets) / 2).astype(int)):
            label_pos.append(start + (change_of_net_pos[i] - start) / 2)
            start = change_of_net_pos[i]

        start = int(len(band) / 2)
        for i in np.arange(np.ceil(len(all_supnets) / 2).astype(int), len(all_supnets)):
            if i != len(all_supnets) - 1:
                label_pos.append(start + (change_of_supnet_pos[i] - start) / 2)
                start = change_of_supnet_pos[i]
                continue
            label_pos.append(start + (end - start) / 2)

        x_labels = (
            all_nets[: np.ceil(len(all_nets) / 2).astype(int)]
            + all_supnets[np.ceil(len(all_supnets) / 2).astype(int) :]
        )
        ax.set_xticks(label_pos)
        ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
        ax.set_yticks([50, 150])
        ax.set_yticklabels(["LH", "RH"], fontsize=8)
        output_path = os.path.join(out_dir, f"AEC_Mean_{band_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
