"""
postporcessing.py - Spectral analysis utilities for MEG/EEG source data using MNE-Python.

Provides functions for computing power spectral density (PSD) and extracting frequency-specific
power per vertex from SourceEstimate objects, as well as morphing results to a standard space.

Author: isaant

Last modified 27/11/2025

"""

import mne
import os
from joblib import Parallel, delayed
from typing import cast
from scipy.signal.windows import get_window
from scipy.signal._short_time_fft import ShortTimeFFT, FFT_MODE_TYPE
import numpy as np
import mne_connectivity
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from os import PathLike
from pathlib import Path
from scipy.signal import hilbert

def make_epochs(data, sfreq, duration, overlap=0.0):
    """Convierte (n_parcelas, n_tiempos) en épocas."""
    window = int(round(duration * sfreq))
    step = int(round(window * (1.0 - overlap)))

    starts = np.arange(
        0,
        data.shape[1] - window + 1,
        step,
    )

    return np.stack(
        [data[:, start:start + window] for start in starts],
        axis=0,
    )

#this function was modify from https://github.com/scipy/scipy/blob/v1.18.0/scipy/signal/_spectral_py.py#L515-L687. 
#welch calls csd, so I modify csd and called it welch here.
#This function rejects segments overlaping the BAD windows
def welch(x, fs=1.0, window='hann_periodic', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean', bad_intervals=None):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's method.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a periodic Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None` and may
        not be greater than `nperseg`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    axis : int, optional
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).
    average : { 'mean', 'median' }, optional
        Method to use when averaging periodograms. If the spectrum is
        complex, the average is computed separately for the real and
        imaginary parts. Defaults to 'mean'.

        .. versionadded:: 1.2.0

    bad_intervals : array_like of shape (n_intervals, 2), optional
        Bad intervals as [start, end) sample indices relative to the start
        of the input signal. Windows overlapping any interval are excluded
        before averaging, across all channels. None disables rejection.
        Upstream QC is expected to ensure at least one valid window remains.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density or cross power spectrum of x,y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method. [Equivalent to
           csd(x,x)]
    coherence: Magnitude squared coherence by Welch's method.

    Notes
    -----
    By convention, Pxy is computed with the conjugate FFT of X
    multiplied by the FFT of Y.

    If the input series differ in length, the shorter series will be
    zero-padded to match.

    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade-off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    The ratio of the cross spectrum (``scaling='spectrum'``) divided by the cross
    spectral density (``scaling='density'``) is the constant factor of
    ``sum(abs(window)**2)*fs / abs(sum(window))**2``.
    If `return_onesided` is ``True``, the values of the negative frequencies are added
    to values of the corresponding positive ones.

    Consult the :ref:`tutorial_SpectralAnalysis` section of the :ref:`user_guide`
    for a discussion of the scalings of a spectral density and an (amplitude) spectrum.

    Welch's method may be interpreted as taking the average over the time slices of a
    (cross-) spectrogram. Internally, this function utilizes the  `ShortTimeFFT`  to
    determine the required (cross-) spectrogram. An example below illustrates that it
    is straightforward to calculate `Pxy` directly with the `ShortTimeFFT`. However,
    there are some notable differences in the behavior of the `ShortTimeFFT`:

    * There is no direct `ShortTimeFFT` equivalent for the `csd` parameter
      combination ``return_onesided=True, scaling='density'``, since
      ``fft_mode='onesided2X'`` requires ``'psd'`` scaling. The is due to `csd`
      returning the doubled squared magnitude in this case, which does not have a
      sensible interpretation.
    * `ShortTimeFFT` uses `float64` / `complex128` internally, which is due to the
      behavior of the utilized `~scipy.fft` module. Thus, those are the dtypes being
      returned. The `csd` function casts the return values to `float32` / `complex64`
      if the input is `float32` / `complex64` as well.
    * The `csd` function calculates ``np.conj(Sx[q,p]) * Sy[q,p]``, whereas
      `~ShortTimeFFT.spectrogram` calculates ``Sx[q,p] * np.conj(Sy[q,p])`` where
      ``Sx[q,p]``, ``Sy[q,p]`` are the STFTs of `x` and `y`. Also, the window
      positioning is different.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975

    Examples
    --------
    The following example plots the cross power spectral density of two signals with
    some common features:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    ...
    ... # Generate two test signals with some common features:
    >>> N, fs = 100_000, 10e3  # number of samples and sampling frequency
    >>> amp, freq = 20, 1234.0  # amplitude and frequency of utilized sine signal
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
    ...
    ... # Compute and plot the magnitude of the cross spectral density:
    >>> nperseg, noverlap, win = 1024, 512, 'hann'
    >>> f, Pxy = signal.csd(x, y, fs, win, nperseg, noverlap)
    >>> fig0, ax0 = plt.subplots(tight_layout=True)
    >>> ax0.set_title(f"CSD ({win.title()}-window, {nperseg=}, {noverlap=})")
    >>> ax0.set(xlabel="Frequency $f$ in kHz", ylabel="CSD Magnitude in V²/Hz")
    >>> ax0.semilogy(f/1e3, np.abs(Pxy))
    >>> ax0.grid()
    >>> plt.show()

    The cross spectral density is calculated by taking the average over the time slices
    of a spectrogram:

    >>> SFT = signal.ShortTimeFFT.from_window('hann', fs, nperseg, noverlap,
    ...                                       scale_to='psd', fft_mode='onesided2X',
    ...                                       phase_shift=None)
    >>> Sxy1 = SFT.spectrogram(y, x, detr='constant', k_offset=nperseg//2,
    ...                        p0=0, p1=(N-noverlap) // SFT.hop)
    >>> Pxy1 = Sxy1.mean(axis=-1)
    >>> np.allclose(Pxy, Pxy1)  # same result as with csd()
    True

    As discussed in the Notes section, the results of using an approach analogous to
    the code snippet above and the `csd` function may deviate due to implementation
    details.

    Note that the code snippet above can be easily adapted to determine other
    statistical properties than the mean value.
    """
    # The following lines are resembling the behavior of the originally utilized
    # `_spectral_helper()` function:
    
    x = np.asarray(x)
    y=x
    same_data, axis = y is x, int(axis)
    if not same_data:
        y = np.asarray(y)
        # Check if we can broadcast the outer axes together
        x_outer, y_outer  = list(x.shape), list(y.shape)
        x_outer.pop(axis)
        y_outer.pop(axis)
        try:
            outer_shape = np.broadcast_shapes(x_outer, y_outer)
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e
        if x.size == 0 or y.size == 0:
            out_shape = outer_shape + (min([x.shape[axis], y.shape[axis]]),)
            empty_out = np.moveaxis(np.empty(out_shape), -1, axis)
            return empty_out, empty_out
        out_dtype = np.result_type(x, y, np.complex64)
    else:  # x is y:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape)
        out_dtype = np.result_type(x, np.complex64)

    n = x.shape[axis] if same_data else max(x.shape[axis], y.shape[axis])
    if isinstance(window, str) or isinstance(window, tuple):
        nperseg = int(nperseg) if nperseg is not None else 256
        if nperseg < 1:
            raise ValueError(f"Parameter {nperseg=} is not a positive integer!")
        elif n < nperseg:
            warnings.warn(f"{nperseg=} is greater than signal length max(len(x), " +
                          f"len(y)) = {n}, using nperseg = {n}", stacklevel=3)
            nperseg = n
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if nperseg is None:
            nperseg = len(win)
    if nperseg != len(win):
        raise ValueError(f"{nperseg=} does not equal {len(win)=}")

    nfft = int(nfft) if nfft is not None else nperseg
    if nfft < nperseg:
        raise ValueError(f"{nfft=} must be greater than or equal to {nperseg=}!")
    noverlap = int(noverlap) if noverlap is not None else nperseg // 2
    if noverlap >= nperseg:
        raise ValueError(f"{noverlap=} must be less than {nperseg=}!")
    if np.iscomplexobj(x) and return_onesided:
        return_onesided = False

    if x.shape[axis] < y.shape[axis]:  # zero-pad x to shape of y:
        z_shape = list(y.shape)
        z_shape[axis] = y.shape[axis] - x.shape[axis]
        x = np.concatenate((x, np.zeros(z_shape)), axis=axis)
    elif y.shape[axis] < x.shape[axis]:  # zero-pad y to shape of x:
        z_shape = list(x.shape)
        z_shape[axis] = x.shape[axis] - y.shape[axis]
        y = np.concatenate((y, np.zeros(z_shape)), axis=axis)

    # using cast() to make mypy happy:
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if return_onesided else 'twosided')
    if scaling not in (scales := {'spectrum': 'magnitude', 'density': 'psd'}):
        raise ValueError(f"Parameter {scaling=} not in {scales}!")

    SFT = ShortTimeFFT(win, nperseg - noverlap, fs, fft_mode=fft_mode, mfft=nfft,
                       scale_to=scales[scaling], phase_shift=None)
    # csd() calculates X.conj()*Y instead of X*Y.conj():
    Pxy = SFT.spectrogram(y, x, detr=None if detrend is False else detrend,
                          p0=0, p1=(n - noverlap) // SFT.hop, k_offset=nperseg // 2,
                          axis=axis)
    # Note:
    # 'onesided2X' scaling of ShortTimeFFT conflicts with the
    # scaling='spectrum' parameter, since it doubles the squared magnitude,
    # which in the view of the ShortTimeFFT implementation does not make sense.
    # Hence, the doubling of the square is implemented here:
    if return_onesided:
        f_axis = Pxy.ndim - 1 + axis if axis < 0 else axis
        Pxy = np.moveaxis(Pxy, f_axis, -1)
        Pxy[..., 1:-1 if SFT.mfft % 2 == 0 else None] *= 2
        Pxy = np.moveaxis(Pxy, -1, f_axis)

    # Reject windows overlapping bad intervals before averaging. Interval
    # endpoints are sample indices relative to the start of the input signal.
    if bad_intervals is not None:
        bad_intervals = np.asarray(bad_intervals, dtype=float)
        if bad_intervals.size:
            if (
                bad_intervals.ndim != 2
                or bad_intervals.shape[1] != 2
                or not np.isfinite(bad_intervals).all()
                or np.any(bad_intervals[:, 1] <= bad_intervals[:, 0])
            ):
                raise ValueError(
                    "bad_intervals must contain finite [start, end) pairs "
                    "with end > start."
                )

            # The spectrogram above starts at sample zero and uses full windows.
            window_starts = np.arange(Pxy.shape[-1]) * SFT.hop
            window_ends = window_starts + nperseg
            valid_windows = np.ones(Pxy.shape[-1], dtype=bool)
            for bad_start, bad_end in bad_intervals:
                overlaps = (window_starts < bad_end) & (bad_start < window_ends)
                valid_windows &= ~overlaps

            Pxy = Pxy[..., valid_windows]

    # Average over windows.
    if Pxy.shape[-1] > 1:
        if average == 'median':
            # np.median must be passed real arrays for the desired result
            bias = _median_bias(Pxy.shape[-1])
            if np.iscomplexobj(Pxy):
                Pxy = (np.median(np.real(Pxy), axis=-1) +
                       np.median(np.imag(Pxy), axis=-1) * 1j)
            else:
                Pxy = np.median(Pxy, axis=-1)
            Pxy /= bias
        elif average == 'mean':
            Pxy = Pxy.mean(axis=-1)
        else:
            raise ValueError(f"Parameter {average=} must be 'median' or 'mean'!")
    else:
        Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    # cast output type;
    Pxy = Pxy.astype(out_dtype)
    return SFT.f, Pxy.real

def _generate_bad_intervals(bad_intervals, n_times, sfreq, tmin=None):
    """
    Generate signal-relative bad intervals from an array or STC metadata JSON.

    JSON annotations labeled BAD_mad are aligned to the source signal,
    clipped to its duration, and discarded if they fall entirely outside it.
    Invalid inputs produce a warning and disable bad-segment rejection.

    Parameters
    ----------
    bad_intervals : path-like | array_like | None
        Path to an STC metadata JSON referencing an annotations file, or
        an array of shape (n_intervals, 2) containing [start, stop) sample
        indices relative to the first signal sample. Array endpoints must
        be finite integers within [0, n_times], with stop > start.
        None disables rejection.
    n_times : int
        Number of samples in the analyzed signal. Must match the STC length
        stored in the metadata when a JSON file is supplied.
    sfreq : float
        Sampling frequency of the analyzed signal in Hz. Used to convert
        annotation times to samples and validate the JSON sampling rate.
    tmin : float | None, optional
        STC start time in seconds, compared with stc_tmin_s in the metadata.
        If None, this comparison is skipped. Ignored for array inputs.

    Returns
    -------
    intervals : ndarray of int64, shape (n_intervals, 2) | None
        Bad intervals with inclusive starts and exclusive stops, relative
        to the first signal sample. An empty array indicates no bad
        intervals. None indicates absent or invalid input; invalid input
        also emits a RuntimeWarning. Welch window viability is not checked.
    """
    if bad_intervals is None:
        return None
    try:
        if isinstance(bad_intervals, (str, PathLike)):
            metadata_path = Path(bad_intervals).expanduser()
            with metadata_path.open(encoding="utf-8") as fid:
                metadata = json.load(fid)

            # Check that the metadata describes this STC's time axis.
            if metadata["stc_n_times"] != n_times:
                raise ValueError("Metadata and STC lengths differ.")
            if not np.isclose(
                metadata["stc_sfreq"], sfreq, rtol=1e-6, atol=0
            ):
                raise ValueError("Metadata and STC sampling rates differ.")
            if tmin is not None and not np.isclose(
                metadata["stc_tmin_s"], tmin,
                rtol=0, atol=0.01 / sfreq,
            ):
                raise ValueError("Metadata and STC start times differ.")

            raw_sfreq = float(metadata["raw_sfreq"])
            raw_first_time = float(metadata["raw_first_time_s"])
            source_start = float(metadata["source_start_sample"])

            if (
                not np.isfinite(
                    [raw_sfreq, raw_first_time, source_start]
                ).all()
                or raw_sfreq <= 0
                or source_start < 0
                or source_start != np.floor(source_start)
            ):
                raise ValueError("Invalid timing metadata.")

            annot_path = Path(metadata["annotations_file"]).expanduser()
            if not annot_path.is_absolute():
                annot_path = metadata_path.parent / annot_path

            annot = mne.read_annotations(annot_path)
            bad = annot[annot.description == "BAD_mad"]

            starts_s = (bad.onset - raw_first_time - source_start / raw_sfreq)

            if (
                not np.isfinite(starts_s).all()
                or not np.isfinite(bad.duration).all()
                or np.any(bad.duration < 0)
            ):
                raise ValueError("Invalid annotation times.")

            # Zero-duration annotations do not mark an interval.
            positive_duration = bad.duration > 0
            starts_s = starts_s[positive_duration]
            durations_s = bad.duration[positive_duration]

            intervals = np.column_stack((
                np.floor(starts_s * sfreq),
                np.ceil((starts_s + durations_s) * sfreq),
            ))

            # Original annotations may extend beyond the STC crop.
            intervals = np.clip(intervals, 0, n_times)
            intervals = intervals[intervals[:, 1] > intervals[:, 0]]
        else:
            intervals = np.asarray(bad_intervals, dtype=float)

        # Accept [] as an empty interval list.
        if intervals.shape == (0,):
            intervals = intervals.reshape(0, 2)

        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError("Expected an (n_intervals, 2) array.")
        if not np.isfinite(intervals).all():
            raise ValueError("Interval endpoints must be finite.")
        if np.any(intervals != np.floor(intervals)):
            raise ValueError("Interval endpoints must be integer samples.")
        if np.any(intervals < 0) or np.any(intervals > n_times):
            raise ValueError("Intervals extend outside the STC.")
        if np.any(intervals[:, 1] <= intervals[:, 0]):
            raise ValueError("Each interval must satisfy stop > start.")

        intervals = intervals.astype(np.int64)


    except Exception as exc:
        warnings.warn(
            f"Bad-segment rejection disabled: {exc} Using all samples.",
            RuntimeWarning,
            stacklevel=2,
        )
        intervals = None
    return intervals

def resolve_bad_segments(option, stc_path):
    """Return a metadata path or CSV intervals; missing inputs disable rejection."""
    if option is None:
        return None
    try:
        if option.lower() == "json":
            base = str(stc_path).removesuffix("-lh.stc").removesuffix("-rh.stc")
            metadata_path = Path(base + "_metadata.json")
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            print(f"[PSD] Using metadata: {metadata_path}", flush=True)
            return str(metadata_path)

        csv_path = Path(option).expanduser()
        if csv_path.suffix.lower() != ".csv":
            raise ValueError("--bad_seg must be 'json' or a CSV file path.")
        table = pd.read_csv(csv_path)
        csv_structure = (
            "CSV must contain columns named exactly 'start_sample' and 'stop_sample'. "
            "Values must be finite integer sample indices relative to the first STC "
            "sample, with 0 <= start_sample < stop_sample <= STC sample count "
            "(start inclusive, stop exclusive; not seconds). "
            "Example: header 'start_sample,stop_sample' and data row '100,500'."
        )
        if not {"start_sample", "stop_sample"}.issubset(table.columns):
            raise ValueError(csv_structure)
        try:
            intervals = table[["start_sample", "stop_sample"]].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(csv_structure) from exc
        print(f"[PSD] Using bad intervals from: {csv_path}", flush=True)
        # PSD_per_vertex validates sample bounds and interval ordering.
        return intervals
    except (OSError, ValueError, KeyError) as exc:
        print(f"[PSD] Bad-window rejection disabled: {exc}. Using all windows.", flush=True)
        return None

def PSD_per_vertex(stc, bands, window_length=4, bad_intervals=None):
    """
    Compute vertex-wise PSD and normalized mean band power.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source time series.
    bands : dict
        Band names mapped to (fmin, fmax), in Hz.
    window_length : float
        Window length in seconds for PSD computation (default: 4.0).
    bad_intervals : path-like | array_like | None
        Path to an STC metadata JSON, or an (n_intervals, 2) array
        of [start, stop) sample indices relative to the STC.
        Invalid rejection inputs trigger a warning and disable rejection.

    Returns
    -------
    psd_all : ndarray
        Power spectral density, shape (n_vertices, n_freqs).
    freqs : ndarray
        Frequency bins in Hz.
    psd_normalized : ndarray
        PSD normalized by its sum across frequencies.
    band_powers : dict
        Mean normalized PSD within each band.

    Notes
    -----
    Requires the custom welch function supporting bad_intervals.
    """
    data = stc.data
    sfreq = float(1.0 / stc.tstep)
    n_times = data.shape[-1]
    window_samples = min(int(round(window_length * sfreq)), n_times)
    overlap_samples = window_samples // 2
    intervals = None

    if bad_intervals is not None:
        intervals = _generate_bad_intervals(bad_intervals, n_times, sfreq, tmin=stc.tmin)
        
    print(intervals.shape)

    freqs, psd_all = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=window_samples,
        noverlap=overlap_samples,
        scaling="density",
        bad_intervals=intervals,
    )

    total_power = psd_all.sum(axis=1, keepdims=True)
    psd_normalized = psd_all / total_power
    

    band_powers = {
        band: psd_normalized[
            :, (freqs >= fmin) & (freqs <= fmax)
        ].mean(axis=1)
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


def generate_brain_screenshot(stc_band_morph, views, power, surfer_kwargs, clim='calculate'):
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

    if clim == 'calculate':
        clim = dict(kind="value", lims=[0, max(power) / 2, max(power)])  # Colorband limits
    
    brain = stc_band_morph.plot(
        **surfer_kwargs, clim= clim
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


def compute_band_oAEC(parc_ts, sfreq, bands, bad_intervals=None):
    """
    Compute sample-weighted orthogonal envelope correlations over clean segments.

    Filter each full time series and apply SciPy's Hilbert transform before
    splitting it into contiguous segments between bad intervals. Compute
    pairwise oAEC separately for each segment and average the matrices using
    segment sample counts as weights.

    Parameters
    ----------
    parc_ts : ndarray, shape (n_labels, n_samples)
        Real-valued parcel time series.
    sfreq : float
        Sampling frequency in Hz.
    bands : dict
        Band names mapped to (fmin, fmax) in Hz.
    bad_intervals : path-like | array_like | None, optional
        STC metadata JSON path or (n_intervals, 2) array of [start, stop)
        sample indices relative to the input signal. None uses the entire
        signal. Invalid inputs warn and disable rejection through
        _generate_bad_intervals.

    Returns
    -------
    correlations : dict
        Band names mapped to symmetric (n_labels, n_labels) oAEC matrices.
        Each matrix is sum(n_samples * segment_matrix) / sum(n_samples).
        Segments with fewer than two samples are omitted with a warning.
        Raises ValueError if no usable clean segment remains.
    """
    n_times = parc_ts.shape[-1]
    intervals = _generate_bad_intervals(bad_intervals, n_times, sfreq)
    # The complement of sorted intervals also handles overlaps and duplicates.
    clean_segments = []
    cursor = 0
    if intervals is not None and intervals.size:
        for start, stop in intervals[np.argsort(intervals[:, 0])]:
            if start > cursor:
                clean_segments.append((cursor, int(start)))
            cursor = max(cursor, int(stop))
    if cursor < n_times:
        clean_segments.append((cursor, n_times))

    usable_segments = [(start, stop) for start, stop in clean_segments
                       if stop - start >= 2]
    if len(usable_segments) != len(clean_segments):
        warnings.warn("Clean segments with fewer than two samples were omitted.",
                      RuntimeWarning, stacklevel=2)
    if not usable_segments:
        raise ValueError("No clean segment with at least two samples remains for oAEC.")
    
    total_samples = sum(stop - start for start, stop in usable_segments)

    correlations = {}
    for band_name, band in bands.items():
        filtered = mne.filter.filter_data(parc_ts, sfreq, band[0], band[1])
        analytic = hilbert(filtered, axis=-1)
        weighted_sum = np.zeros((parc_ts.shape[0], parc_ts.shape[0]), dtype=float)
        for start, stop in usable_segments:
            # Complex input tells envelope_correlation Hilbert is already done.
            corr_obj = mne_connectivity.envelope_correlation(
                analytic[np.newaxis, :, start:stop], orthogonalize="pairwise"
            )
            matrix = corr_obj.combine().get_data(output="dense")[:, :, 0]
            weighted_sum += (stop - start) * matrix
        correlations[band_name] = weighted_sum / total_samples

    return correlations


def compute_band_wPLI(parc_ts, sfreq, bands, n_jobs=1, debiased=True):
    # 300 s -> aproximadamente 30 épocas de 10 s
    epochs = make_epochs(
        parc_ts,
        sfreq=sfreq,
        duration=10.0,
        overlap=0.0,
    )

    band_names = list(bands)
    fmin = tuple(bands[name][0] for name in band_names)
    fmax = tuple(bands[name][1] for name in band_names)

    con = mne_connectivity.spectral_connectivity_epochs(
        epochs,
        method="wpli2_debiased" if debiased else "wpli",
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,

        # Mucho más rápido que True; normalmente suficiente.
        mt_adaptive=False,
        mt_low_bias=True,

        n_jobs=n_jobs,
        block_size=1000,
        verbose=True,
    )

    # Shape: (n_rois, n_rois, n_bandas)
    dense = con.get_data(output="dense")

    # MNE almacena solamente un triángulo para medidas bivariadas.
    dense = dense + dense.transpose(1, 0, 2)

    return {
        band_name: dense[:, :, band_idx]
        for band_idx, band_name in enumerate(band_names)
    }

def compute_band_PLV(parc_ts, sfreq, bands, n_jobs=1, epoch_duration=10.0):
    """
    Compute Phase-Locking Value (PLV) connectivity matrices across epochs.

    Parameters
    ----------
    parc_ts : ndarray, shape (n_rois, n_times)
        Continuous parcellated time series.
    sfreq : float
        Sampling frequency in Hz.
    bands : dict
        Frequency bands, for example:
        {"alpha": (8, 12), "beta": (15, 29)}.
    n_jobs : int
        Number of parallel workers.
    epoch_duration : float
        Duration of each epoch in seconds.

    Returns
    -------
    plv_matrices : dict
        Dictionary mapping each band name to a symmetric PLV matrix with
        shape (n_rois, n_rois).
    """
    epochs = make_epochs(
        parc_ts,
        sfreq=sfreq,
        duration=epoch_duration,
        overlap=0.0,
    )

    band_names = list(bands)
    fmin = tuple(bands[name][0] for name in band_names)
    fmax = tuple(bands[name][1] for name in band_names)

    con = mne_connectivity.spectral_connectivity_epochs(
        epochs,
        method="plv",
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=False,
        mt_low_bias=True,
        n_jobs=n_jobs,
        block_size=1000,
        verbose=True,
    )

    # Shape: (n_rois, n_rois, n_bands)
    dense = con.get_data(output="dense")

    # MNE stores only one triangle for bivariate connectivity measures.
    dense = dense + dense.transpose(1, 0, 2)

    # By definition, each signal is perfectly phase-locked with itself.
    for band_idx in range(len(band_names)):
        np.fill_diagonal(dense[:, :, band_idx], 1.0)

    return {
        band_name: dense[:, :, band_idx]
        for band_idx, band_name in enumerate(band_names)
    }

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


def plot_corr_with_labels(corr, labels, out_dir,metric):
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
        sns.heatmap(band, cmap="viridis", vmin=clim[0], vmax=clim[1], ax=ax, cbar=True)
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
        output_path = os.path.join(out_dir, f"{metric}_Mean_{band_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
