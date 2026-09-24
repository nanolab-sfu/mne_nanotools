# Sensor montages for ECG/EOG

`generic_taskfree.py` accepts physical channels or sensor groups through the same flags:

```bash
python generic_taskfree.py --root_dir /path/to/project --subject_id sub-01 \
  --system MEGIN --eog_ch eog_montage --ecg_ch ecg_montage
```

Supported names also include `Left-frontal`, `Right-frontal`, `Left-parietal`,
`Right-parietal`, `Left-occipital`, `Right-occipital`, `Left-temporal`, and `Right-temporal`.
Montage names are case-insensitive; hyphens, spaces (inside quotes), and underscores
are interchangeable. The aliases `oeg_montage` and `ocg_monatge` are also accepted.
Physical channels still work; multiple EOG channels can be separated by commas.
With `auto`, or when the requested conventional EOG/ECG channels are unavailable,
the pipeline first looks for good channels of the corresponding type, then falls
back to the default montage. Unknown custom names raise an error to catch typos.

## Files and lookup order

The default directory is `<root_dir>/montages`; use `--montages_dir` to specify another path.
For each name, files are checked in this order:

1. `montages/MEGIN/<name>.json` or `.txt` (or `CTF`, depending on the system).
2. `montages/<name>.json` or `.txt`.
3. For built-in names, a selection is generated and reused at
   `montages/<system>/generated/<geometry-fingerprint>/<name>.json`.

The requested montage is generated when needed, without downloading recordings.
The fingerprint includes channel names, positions, and the transformation to head
coordinates, preventing reuse of a geometric selection with a different helmet or
head position. Existing user files take precedence. To use a reviewed selection
across subjects, copy the generated JSON to `montages/<system>/<name>.json` and
edit `channels`.

A JSON file can contain a list, or an object with `channels` and an optional `system`:

```json
{"system": "CTF", "channels": ["MLT11", "MLT12", "MLT13"]}
```

TXT files support one channel per line, comma-separated names, and `#` comments.
Internal spaces, as in `MEG 0111`, are preserved. CTF serial-number suffixes are
resolved against the current recording. Missing or ambiguous channels and
incompatible sensor types raise an error; channels marked bad are excluded when
the montage is used. MEGIN sensor types are not mixed.

## Generation and row definitions

MEGIN uses MNE's regional Vectorview selections, restricted to magnetometers.
CTF uses primary sensors identified by the prefixes `MLF/MRF`, `MLP/MRP`,
`MLO/MRO`, and `MLT/MRT`, excluding reference sensors. CTF sensors are axial
gradiometers; they are not physically identical to MEGIN magnetometers.

- `eog_montage`: the three lowest rows of the combined left and right frontal
  regions (also including CTF's central frontal `MZF` sensors).
- `ecg_montage`: the two lowest rows of the left temporal region.

Rows are **estimated** from bottom to top in head coordinates. The median
nearest-neighbor distance within the region defines the sensor spacing. Each row
includes sensors from the lowest remaining `z` coordinate up to half that spacing
above it. This is repeated three or two times. This geometric criterion does not
correspond to an official manufacturer row numbering scheme; review and edit the
JSON files to specify an exact anatomical selection. If valid positions or the
required transformation are missing, an error indicates that an explicit sensor
list is needed.

## Signal used for detection

The group's first principal component is computed from centered signals and
normalized to avoid cancellation of opposite polarities when averaging. It is
added as a temporary EOG/ECG reference for QC, SSP, and BCG, then removed before
covariance estimation, saving, and source modeling. It does not represent a
physical measurement in volts. PC1 may contain activity unrelated to the artifact,
so the events and projections in the report still need to be checked.

References: [Vectorview selections](https://mne.tools/stable/generated/mne.read_vectorview_selection.html),
[CTF sensors and coordinates](https://mne.tools/stable/documentation/implementation.html).

Tests: `python -m unittest discover -s tests -p 'test_artifact_montages.py'`.
