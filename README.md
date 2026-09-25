# mne_nanotools

**mne_nanotools** is a lightweight, reusable Python toolkit maintained by the NanoLab (SFU) to support **MEG workflows built around MNE-Python**, with utilities for **pre-processing, post-processing, I/O handling, and pipeline orchestration**. The repository is designed to be imported inside larger analysis scripts rather than used as a standalone application.  

**Author:** Santiago Isaac Flores Alonso
**Co-developer:** Isabel Wilson
**Version:** 0.1.0

---

## What this repo is for

This package provides convenience functions to make MEG projects more reproducible and less repetitive, including:

- **Preprocessing helpers** (e.g., cleaning/standardization steps used across projects)
- **Postprocessing helpers** (e.g., feature extraction/aggregation utilities used after preprocessing)
- **I/O utilities** (consistent reads/writes, bookkeeping, path helpers)
- **Workflow scripts** that can be used as entry points or templates (e.g., task-free pipelines, coreg/handling digitization)

> Note: Specific functions and their signatures are expected to evolve as the toolkit matures; treat this as a “lab toolbox” intended for internal and collaborative use.  [oai_citation:1‡GitHub](https://github.com/nanolab-sfu/mne_nanotools)

---

## Repository structure (high level)

Common modules/scripts you’ll find here include:

- `preprocessing.py` — preprocessing utilities  
- `postprocessing.py` — postprocessing / feature utilities  
- `io_handlers.py` — I/O helpers  
- `corregistration.py` — MEG–MRI coregistration helper script/template  
- `generic_taskfree_MEGIN.py` — a task-free MEG workflow script/template  
- `remove_digpoints.py` — utilities for managing/removing digitization points  

(Names may change over time; check the repo root for the current list.)  [oai_citation:2‡GitHub](https://github.com/nanolab-sfu/mne_nanotools/blob/main/preprocessing.py)

---
nanotools: pipeline flags reference

Reference for the command-line interface of `generic_taskfree.py` and the CSV interface provided by `submit_prepro_from_csv.py`, reviewed on 25 September 2026.

Defaults below come from `_parse_args()`, which controls command-line and CSV runs. Calling `preprocess_subject()` directly from Python can use different defaults, notably for the inverse method and EOG reference. These are implementation defaults, not a recommended recipe for every dataset.

See [Data layout and naming conventions](NANOTOOLS_DATA_LAYOUT.md) for required files and [Sensor montages](MONTAGES.md) for artifact-reference selections.

## 1. How to supply settings

A value-taking CLI option uses a flag followed by its value:

```bash
--system CTF
--l_freq 0.5
--inv_method dSPM
```

A boolean action flag appears on its own:

```bash
--verbose
--overwrite
--no_compute_bem_if_missing
```

Do not write `--overwrite False`: omitting `--overwrite` leaves it false.

In Excel/CSV, the header is the option name **without `--`**. For example, `inv_method` is a column whose cell might contain `dSPM`. Preserve capitalization: `eSSS`, `MEGIN`, `CTF`, `MNE`, `dSPM`, and `sLORETA` are case-sensitive where the parser expects them.

### CSV launcher controls

These two columns belong to the launcher, not the pipeline CLI:

| Column | Meaning | Example |
| --- | --- | --- |
| `sub_ses_run` | Recording identifier parsed into subject, session, task, and run. | `sub-001_ses-01_task-rest_run-01` |
| `run-prepro` | Selects whether to execute this row. | `TRUE` |

The launcher supplies `root_dir` and `tsss_dir` from its configured paths and derives `subject_id`, `session`, and `run` from `sub_ses_run`. Configure those at their source rather than duplicating them as CSV columns.

**Always fill the `task` column explicitly.** The current launcher skips empty cells before its intended task fallback. A missing task cell therefore does not reliably inherit `task-rest` from the identifier and can leave the pipeline using `rest1`.

### CSV value rules

| Cell content | Launcher behavior |
| --- | --- |
| Blank | Omits the option, allowing its CLI default. |
| `TRUE`, `YES`, `Y` | Adds the flag without a value. Intended for boolean action flags. |
| `FALSE`, `NO`, `N` | Omits the option. Does not override a default-true setting. |
| `0` | Passes the numeric/string value `0`; it is not a boolean false. |
| Any other nonempty value | Passes the flag and the entire cell as one argument. |

Boolean words are case-insensitive. Use numeric values for numeric fields: `TRUE` in a numeric column produces an argument without its required value. Do not write `None` to request a default; leave the cell blank.

Quote CSV cells containing commas. Do not add arbitrary notes columns: nonempty cells in such columns become unrecognized CLI flags.

### Multi-value options

The current launcher does **not** split a cell into multiple command-line arguments.

| Option | Direct CLI example | Current CSV support |
| --- | --- | --- |
| `crop_tmin` / `crop_tmax` | `--crop_tmin 10 10` | Requires two arguments. Use defaults, direct CLI, or adapt the launcher. |
| `line_freqs` | `--line_freqs 50 100 150` | One numeric frequency works. Multiple frequencies require direct CLI or launcher adaptation. |
| `num_proj_*` | `--num_proj_ecg 1 3` | One index works. Multiple indices require direct CLI or launcher adaptation. |
| `additional_bads` | `--additional_bads MEG0113 MEG1421` | Comma-separated names work because the pipeline splits commas itself. |
| `eSSS` / `raw_ssp_band` | `--eSSS 42-45,50-53` | Comma-separated ranges work because these flags have a custom parser. |
| `eog_ch` | `--eog_ch EOG001,EOG002` | A comma-separated reference string works. |

For example, a literal CSV cell `"1 3"` is not a valid way to choose two PCA indices with the current launcher.

## 2. Dataset, recording, and file selection

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--root_dir` | Required | Dataset root containing `MEG`, anatomy, and output locations. Use the dataset root for the complete workflow. The launcher uses `ROOT_DIR`. |
| `--subject_id` | Required | MEG subject identifier, such as `sub-001`. Also supplies the default FreeSurfer subject ID. |
| `--session` | `None` | Session identifier, such as `ses-01`. Use an explicit, consistent session for the recommended workflow. Legacy sessionless discovery exists, but transform lookup has limitations. |
| `--task` | `None` | Task label such as `rest`, used in input discovery. Takes precedence over `resting`. |
| `--resting` | `rest1` | Legacy task fallback when `task` is absent. Prefer `task` for new configurations. |
| `--run` | `None` | Run selection. Accepts `1`, `01`, `run-1`, or `run-01`. The launcher supplies it from the identifier. |
| `--system` | `MEGIN` | `MEGIN` selects FIF inputs and the Maxwell branch. `CTF` selects `.ds` inputs and skips Maxwell filtering. |
| `--in_file` | `None` | Explicit MEG input path, bypassing automatic recording discovery. For CTF, supply a complete `.ds` directory. |
| `--erm_file` | `None` | Explicit empty-room input path, bypassing ERM discovery. Useful for shared noise recordings or ambiguous matches. |
| `--prefer` | `any` | Ranking preference during MEG discovery: `any`, `raw`, or a filename substring such as `digFiltered`. It is not an exact-file selector. |
| `--task_basename` | `{sub}_{task}_raw.fif` | Legacy basename template. **Currently ineffective for normal CLI/CSV discovery**: the entry point resolves and passes an explicit input path, and does not forward this argument to `preprocess_subject()`. Use `in_file` instead. |
| `--erm_basename` | `{sub}_erm_raw.fif` | Legacy ERM basename template with the same CLI limitation. Use `erm_file` instead. |

MEG discovery first chooses a processing-state group: raw, then tSSS with movement compensation, then tSSS. `prefer` ranks within that group. It cannot force a tSSS file over an available raw file. ERM discovery prioritizes raw, then tSSS, then movement-compensated tSSS.

Explicit recording paths do not relocate the anatomy or coregistration lookup. See the data-layout guide for exact names.

## 3. Time selection, filtering, and resampling

| Flag | CLI default | Units and behavior |
| --- | --- | --- |
| `--crop_tmin` | `10 10` | Start times in seconds: **empty room first, MEG second**. Requires exactly two values. |
| `--crop_tmax` | `110 250` | End times in seconds, in the same order: **empty room first, MEG second**. Requires exactly two values. |
| `--l_freq` | `0.5` | Lower passband cutoff in Hz, passed to the nanotools filtering helper for MEG and ERM. |
| `--h_freq` | `200.0` | Upper passband cutoff in Hz, passed to the same helper. |
| `--line_freqs` | `60 120 180` | Notch frequencies in Hz. Replace them with values appropriate to the recording. Accepts zero or more numeric CLI values. |
| `--downsample` | `500` | Target sample rate in Hz. `0` skips the resampling branch. The code calls resampling for any other supplied integer; the name does not enforce downsampling only. |

Example: keep ERM from 10–110 seconds and MEG from 20–260 seconds:

```bash
--crop_tmin 10 20 --crop_tmax 110 260
```

Choose intervals contained in each recording. Cropping errors are caught and logged, so check that the intended crop actually happened.

The script applies filtering after cropping, then resamples. Choose passband and notch frequencies compatible with the data and sample rate. The CLI accepts floats for `l_freq` and `h_freq`, not the string `None`. Several QC spectra request fixed upper frequencies of 180 or 200 Hz, so aggressive changes to the sample rate also require checking QC compatibility.

Supplying `--line_freqs` without values passes an empty list to the external filtering helper; the behavior of that helper should be checked before treating this as a supported no-notch recipe.

## 4. MEGIN Maxwell filtering and eSSS

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--tsss_dir` | `/Users/isaant/Documents/PosDoc/Projects/tsss_params/2023` | Local author-specific default. Set your own directory containing `sss_cal.dat` and `ct_sparse.fif`. The launcher supplies `TSSS_PARAMS`. |
| `--st_duration` | `10.0` | Temporal window in seconds passed to the Maxwell helper for the MEG recording. |
| `--sss_erm_st_duration` | `None` | Independent temporal-window setting passed for ERM. The default passes `None`; it does not inherit `st_duration`. ERM receives no head-position trajectory. |
| `--eSSS` | `None` | Optional external projection basis derived from unprocessed ERM in specified frequency ranges, then passed to the Maxwell helper. Example: `42-45,50-53`. |

`eSSS` is a range-valued option, **not a boolean**. A CSV cell with two ranges should be quoted, for example `"42-45,50-53"`. Leave it blank to omit it.

The implementation computes an eSSS basis when the MEG input needs a new Maxwell pass and unprocessed ERM is available. If ERM is already SSS-processed, it warns and disables eSSS for that run. Reusing a cached tSSS MEG result bypasses this recomputation.

For the eSSS basis, the first range uses `n_mag=1, n_grad=1`; later ranges use `n_mag=3, n_grad=3`, with `meg="combined"`. These are fixed in the current code. None of the `num_proj_*` flags selects eSSS components.

CTF skips this entire branch. Input already identified as SSS/tSSS-processed skips a new Maxwell pass even with `overwrite`. The flags describe what is passed to the project's external `nanotools.preprocessing.max_filter` helper; confirm helper compatibility when deploying the environment.

## 5. Artifact references and bad channels

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--ecg_ch` | `ECG003` | Cardiac-artifact detection reference: a physical channel, supported montage name such as `ecg_montage`, or `auto`. |
| `--eog_ch` | `EOG001` | Ocular-artifact reference: a physical channel, comma-separated EOG channels, supported montage name such as `eog_montage`, or `auto`. |
| `--montages_dir` | `None` | Custom selection directory. `None` resolves to `<root_dir>/montages`. |
| `--additional_bads` | Empty list | Additional channel names to mark bad in both MEG and ERM. CLI accepts separate names or comma-separated names. CSV should use a quoted comma-separated list. |
| `--reject_mag` | `4e-12` | **Currently unused.** Accepted and passed into `preprocess_subject()`, but not consumed by its implementation. Changing it does not tune artifact rejection. |
| `--reject_grad` | `4000e-13` | **Currently unused**, with the same limitation. |

Physical channels are preferred when available. The montage helper can use sensor-based temporary references when conventional channels are unavailable. Those references support artifact detection; they are removed before covariance, saved recordings, and source modeling. Review their detected events and selected projections in the QC report.

Additional bad-channel names enter **after Maxwell filtering, filtering, and resampling** in the current sequence. They do not retroactively exclude channels from an earlier Maxwell computation. Verify acquisition bad-channel metadata before that stage.

The active bad-segment detector uses fixed settings `win_length=1.0` and `n_mad=3` in `detect_bad_mad_grads_mags()`. There are currently no CLI flags for those settings. The ECG/EOG SSP calls use `reject=None`, independently of the unused `reject_mag` and `reject_grad` arguments.

## 6. Optional SSP branches

SSP means signal-space projection. The following options enable additional projection estimates; the next section explains which components are selected.

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--raw_ssp_band` | `None` | Estimate generic SSP from the MEG recording in one or more bands, such as `42-45,50-53`. Select components using `num_proj_raw`. |
| `--erm_ssp_band` | `None` | Estimate ERM SSP. Accepts `broad` or one range such as `10-20`. Select components using `num_proj_erm`. |
| `--bcglike_ssp` | `False` | Enables ECG-locked SSP for ballistocardiographic-like artifacts. Current computation uses a fixed 1.5–8 Hz band. Select components using `num_proj_bcglike`. |

`broad` uses the ERM signal available at this stage, which has already undergone earlier filtering. It does not mean untouched broadband input. `erm_ssp_band` accepts one range, while `raw_ssp_band` accepts multiple comma-separated ranges.

Omitting a band option leaves that optional branch disabled. Setting `num_proj_erm` or `num_proj_raw` alone does not enable its branch. Enabling a branch without selecting any components may still compute diagnostics.

## 7. Selecting SSP components: `num_proj_*`

**These flags select PCA indices, not component counts.** All five default to `[1]` and accept indices `1`, `2`, and `3`, or `0` alone.

| Flag | Applies to |
| --- | --- |
| `--num_proj_ecg` | Cardiac ECG SSP. |
| `--num_proj_eog` | Ocular EOG SSP. |
| `--num_proj_erm` | Optional ERM SSP, enabled by `erm_ssp_band`. |
| `--num_proj_raw` | Optional generic MEG SSP, separately for each `raw_ssp_band` range. |
| `--num_proj_bcglike` | Optional BCG-like SSP, enabled by `bcglike_ssp`. |

| CLI example | Interpretation |
| --- | --- |
| `--num_proj_ecg 1` | Select only PCA component 1. |
| `--num_proj_ecg 3` | Select only component 3, not the first three. |
| `--num_proj_ecg 1 3` | Select components 1 and 3. |
| `--num_proj_ecg 1 2 3` | Select all three available indices. |
| `--num_proj_ecg 0` | Apply no ECG SSP components. |

For MEGIN, indices apply within each available magnetometer/gradiometer projection group. For CTF, selection uses a single MEG group in its original PCA order. Thus selecting `1` need not mean one projection in total across MEGIN sensor types.

`0` cannot be combined with other indices. Duplicate indices are removed. An unavailable requested index raises an error inside selection; some surrounding pipeline stages catch errors, so inspect the logs and report.

Setting `num_proj_ecg=0` or `num_proj_eog=0` suppresses application of that family's selected components, but it does **not** bypass reference preparation, event detection, or projection estimation. Use QC to choose indices instead of assuming that more removed components always improves the result.

## 8. Anatomy and source reconstruction

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--subjects_dir_name` | `MRI/freesurfer` | FreeSurfer subjects-directory location relative to the dataset root. |
| `--suffix` | `None` | Adds `_<suffix>` to the MEG subject ID to select a FreeSurfer folder when JSON mode is off. Example: `ses-01` selects `sub-001_ses-01`. |
| `--trans` | `-corr_trans.fif` | Transform filename suffix, not a full-path override. See the naming guide for the two subject/session prefixes searched. |
| `--json` | `False` | Enables FreeSurfer subject selection through the coordinate JSON's `IntendedFor` field. Requires the compatible JSON/coregistration workflow and nanotools helper. |
| `--compute_bem_if_missing` | `True` | Enables creation of a missing BEM solution. Already enabled when omitted. |
| `--no_compute_bem_if_missing` | Absent | Sets `compute_bem_if_missing=False`. An existing BEM is still needed for source reconstruction. |
| `--bem_watershed` | `True` | Enables watershed and dense-scalp generation inside the missing-BEM creation branch. Already enabled when omitted. |
| `--no_bem_watershed` | Absent | Sets `bem_watershed=False`. Does not disable source reconstruction. |
| `--inv_method` | `beamformer` | One of `beamformer` (LCMV), `MNE`, `dSPM`, or `sLORETA`. Determines the inverse branch and method-specific output folder. |
| `--snr` | `3.0` | Assumed SNR used in the minimum-norm branch through `lambda2 = 1 / snr²`. It does not tune the LCMV branch. Use a positive nonzero value. |
| `--n_jobs` | `8` | Worker setting passed to forward computation. It does not submit multiple subjects or universally control all parallel work. |

Prepare and inspect BEM surfaces before the cohort run, preferably with `create_missing_bem.py`. The missing-BEM branch tries to build the model before its watershed call. With an existing BEM solution, `bem_watershed` does not trigger surface regeneration.

To disable a default-true setting in CSV, use the negative flag as a boolean column:

```csv
no_compute_bem_if_missing,no_bem_watershed
TRUE,TRUE
```

A cell `compute_bem_if_missing=FALSE` merely omits the positive flag and leaves the CLI default true. Do not enable both positive and negative forms for the same setting.

For a custom transform suffix beginning with `-`, direct CLI syntax such as `--trans=-corr_trans.fif` keeps it attached to its option. The CSV launcher passes the value separately; leaving the default suffix blank avoids that issue. Validate a custom leading-hyphen suffix before using it through the launcher.

The current LCMV implementation fixes settings including `reg=0.05`, `pick_ori="max-power"`, `weight_norm="unit-noise-gain"`, and `rank="info"`. Those are not independently exposed as CLI flags. PSD band definitions are also not exposed through `_parse_args()`.

## 9. Logging, reruns, and cache behavior

| Flag | CLI default | Meaning and use |
| --- | --- | --- |
| `--verbose` | `False` | Enables MNE INFO logging. The default sets MNE logging to ERROR and suppresses selected warnings; pipeline print messages still appear. |
| `--overwrite` | `False` | Controls reuse of some existing tSSS and source-estimate outputs. It is not a universal refresh or file-protection switch. |
| `-h`, `--help` | Not applicable | Shows CLI usage and exits. Use directly, not as an analysis CSV column. |

### What `overwrite` actually changes

| Output/stage | Without `overwrite` | With `overwrite` |
| --- | --- | --- |
| tSSS output derived from an unprocessed input | Reuses an existing recognized cached tSSS result. | Computes a new Maxwell result from the unprocessed input. |
| Input already identified as SSS/tSSS | Skips a new Maxwell pass. | Still skips a new Maxwell pass. |
| Source estimate | Reuses the expected STC when the checked file exists. | Recomputes the source estimate. |
| Existing BEM solution | Reuses it. | Still reuses it. |
| Existing source-space FIF | Reuses it. | Still reuses it. |
| Existing head-position sidecar | Reads it. | Still reads it. |
| Filtered/projected sensor FIF | Saves with overwrite enabled. | Same behavior. |
| QC report and parameter file | Writes the current results when the corresponding save path is reached. | Same behavior. |

When filters, projections, cropping, or inverse settings change, an existing STC can otherwise remain stale. Request recomputation where needed. If Maxwell settings change, use the original unprocessed input and recompute its tSSS result. Changes to anatomy, source space, or head-position estimation require deliberate cache management beyond this flag.

The launcher also overwrites its per-row log when rerunning the same subject/session/run. Keep the previous CSV, logs, and relevant outputs when comparing recipes. Review warnings and required outputs even if the process exits successfully, because some stage exceptions are caught.

## 10. Examples

### Small CSV using supported single-cell values

The following illustrates syntax only. Select frequencies, references, and PCA components from your own QC.

```csv
sub_ses_run,task,system,l_freq,h_freq,downsample,ecg_ch,eog_ch,num_proj_ecg,num_proj_eog,additional_bads,inv_method,overwrite,run-prepro
sub-001_ses-01_task-rest_run-01,rest,MEGIN,0.5,200,500,auto,auto,1,1,,beamformer,FALSE,TRUE
sub-002_ses-01_task-rest_run-01,rest,CTF,0.5,200,500,auto,auto,1,0,,dSPM,FALSE,FALSE
```

The second row is disabled. These rows leave cropping and notch frequencies at their CLI defaults; review those defaults before using the example.

Example optional range cells:

```csv
eSSS,raw_ssp_band,erm_ssp_band,bcglike_ssp,num_proj_raw,num_proj_erm,num_proj_bcglike
"42-45,50-53",,broad,FALSE,1,1,1
```

### Direct CLI with multiple values

This illustrates a 50 Hz notch configuration and explicit cropping/PCA selections. It is not a universal acquisition prescription.

```bash
python generic_taskfree.py \
  --root_dir /path/to/DATASET \
  --subject_id sub-001 \
  --session ses-01 \
  --task rest \
  --run run-01 \
  --system MEGIN \
  --tsss_dir /path/to/tsss_params \
  --crop_tmin 10 10 \
  --crop_tmax 110 250 \
  --l_freq 0.5 \
  --h_freq 200 \
  --line_freqs 50 100 150 \
  --downsample 500 \
  --ecg_ch auto \
  --eog_ch auto \
  --num_proj_ecg 1 3 \
  --num_proj_eog 1 \
  --inv_method beamformer
```

For an intentional rerun after changing relevant settings, add `--overwrite` after reviewing the cache table above.

## 11. BEM helper flags

`create_missing_bem.py` has a separate, smaller CLI:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--root_dir` | Required | Dataset root. |
| `--subjects_dir_name` | `MRI/freesurfer` | Directory containing FreeSurfer subjects. |
| `--overwrite` | `False` | Re-enters BEM generation for subjects that already have a solution and passes overwrite to surface generation. |
| `-h`, `--help` | Not applicable | Prints usage. |

The helper currently calls `mne.write_bem_solution()` without an explicit overwrite argument. Consequently, its overwrite flag should not be assumed to replace an existing solution successfully in every installed MNE version. Check each subject's result and handle an existing solution deliberately before regeneration.

Implementation sources: `generic_taskfree.py` argument parser and processing branches, `submit_prepro_from_csv.py`, `artifact_montages.py`, `MONTAGES.md`, and `create_missing_bem.py`. External nanotools helper internals are not fully included in this repository snapshot.


---
# Data layout and naming conventions

This guide describes the current `generic_taskfree.py`, `submit_prepro_from_csv.py`, and `create_missing_bem.py` implementation, reviewed on 25 September 2026. Examples use fictional subject `sub-001`, session `ses-01`, task `rest`, and run `run-01`.

The recommended layout is **BIDS-like**. These scripts do not validate full BIDS compliance. Use the exact capitalization shown, especially `MEG` and `MRI`.

## 1. Recommended dataset tree

Pass the **dataset root** as `--root_dir`, not its `MEG` subfolder. The file finder can search a MEG root, but the complete pipeline builds anatomy, transform, and output paths relative to the dataset root.

```text
DATASET/                                      # --root_dir
├── MEG/
│   └── sub-001/
│       └── ses-01/
│           └── meg/
│               ├── sub-001_ses-01_task-rest_run-01_meg.fif
│               ├── sub-001_ses-01_task-rest_run-02_meg.fif
│               ├── sub-001_ses-01_task-erm_meg.fif
│               ├── sub-001_ses-01-corr_trans.fif
│               └── sub-001_ses-01_task-rest_run-01_meg_head_pos.pos
│                   # Optional existing MEGIN head-position file
├── MRI/
│   └── freesurfer/                            # --subjects_dir_name MRI/freesurfer
│       └── sub-001/                           # Complete FreeSurfer subject
│           ├── mri/
│           ├── surf/
│           ├── label/
│           ├── bem/
│           │   ├── inner_skull.surf
│           │   ├── sub-001-head-dense.fif
│           │   └── sub-001-5120-5120-5120-bem-sol.fif
│           └── ...                           # Keep other FreeSurfer outputs
├── tsss_params/                              # Example location, configurable
│   └── acquisition_config/
│       ├── sss_cal.dat
│       └── ct_sparse.fif
├── python_fun/                               # Suggested location, configurable
│   ├── flags-to-use.xlsx
│   └── flags-to-use.csv
├── montages/                                 # Optional custom sensor selections
│   ├── MEGIN/
│   │   └── eog_montage.json
│   └── CTF/
│       └── ecg_montage.json
├── derivatives/                              # Created by pipeline
├── logfiles/                                 # Created by CSV launcher
└── logs/                                     # Pipeline diagnostic logs
```

The final `meg/` folder is optional. Files may instead sit directly in `MEG/sub-001/ses-01/`. If `meg/` exists, the pipeline uses it for transform and optional sidecar lookup. Keep those files together in that selected folder.

The BEM subtree above highlights expected files; it is not a complete list of anatomical outputs. Empty folders and renamed files do not substitute for completed FreeSurfer reconstruction or a valid transform.

## 2. Identifiers and recording names

Use consistent identifiers in the folders, recording names, CSV rows, and FreeSurfer subject mapping.

| Item | Recommended value | Rule |
| --- | --- | --- |
| Subject | `sub-001` | Preserve the `sub-` prefix and any leading zeros. |
| Session | `ses-01` | Use the same full value in the folder, filename, and configuration. |
| Task | `rest` | Set the `task` CSV column explicitly to this value. |
| Run | `run-01` | Use a numeric label. Discovery accepts `run-1` and `run-01`. |
| System | `MEGIN` or `CTF` | These are the current CLI choices. |

For the CSV workflow, use simple alphanumeric labels after each prefix. Do not put underscores inside a subject, session, task, or run label: underscores separate entities in the launcher's parser.

### MEGIN recordings

```text
<subject>_<session>_task-<task>_run-<number>_meg.fif
sub-001_ses-01_task-rest_run-01_meg.fif
```

### CTF recordings

```text
<subject>_<session>_task-<task>_run-<number>_meg.ds/
sub-001_ses-01_task-rest_run-01_meg.ds/
```

A CTF `.ds` recording is a **directory**, not a FIF file. Preserve the complete recording and its internal vendor files. The `.ds` examples describe the expected dataset path; do not casually rename a vendor dataset without preserving its internal consistency. Use `--in_file` when an existing recording has another name.

### Empty-room recording (ERM)

The current pipeline requires an empty-room input. Recommended names are:

```text
sub-001_ses-01_task-erm_meg.fif
sub-001_ses-01_task-erm_meg.ds/
```

`task-noise` is also recognized. Store the matching noise recording in the same subject/session search tree, or supply its actual location through `--erm_file`. A shared empty-room directory elsewhere is not automatically associated with the subject. If multiple ERM candidates exist, specify the intended file explicitly.

## 3. Coregistration transform

The default `--trans` value is the **filename suffix** `-corr_trans.fif`. The pipeline checks these names in order inside the selected MEG folder:

```text
<subject>-<session>-corr_trans.fif
<subject>_<session>-corr_trans.fif
```

For the example dataset:

```text
sub-001-ses-01-corr_trans.fif    # First lookup
sub-001_ses-01-corr_trans.fif    # Fallback, used in the tree above
```

Supply one valid MEG-to-MRI transform for the intended recording geometry. This is a transform FIF, not a raw-data FIF containing digitization points. The pipeline consumes the transform; it does not automatically perform coregistration as part of the main preprocessing run.

`--trans` is not an arbitrary full-path override. For example, `--trans=_run-01-corr_trans.fif` makes the fallback name `sub-001_ses-01_run-01-corr_trans.fif`. This allows run-specific transforms when required. A CSV `trans` column can hold the same suffix. For direct CLI use, the `--trans=...` form also handles suffixes beginning with `-`.

The default name does not include the task or run. Only reuse a transform across recordings when its alignment remains appropriate.

## 4. FreeSurfer subject and BEM names

By default, the anatomy directory is:

```text
<root_dir>/MRI/freesurfer/<subject_id>/
```

Set `--subjects_dir_name` to change the FreeSurfer directory. With `--suffix ses-01`, the selected FreeSurfer subject becomes:

```text
MRI/freesurfer/sub-001_ses-01/
```

The pipeline's BEM solution lookup is:

```text
<FreeSurfer directory>/<FreeSurfer subject>/bem/<subject_id>-5120-5120-5120-bem-sol.fif
```

The BEM filename uses the **MEG subject ID**, even if the FreeSurfer folder has a suffix. The current BEM helper uses the **FreeSurfer directory name** when naming its solution. Therefore, a suffixed anatomy folder can produce a mismatch:

```text
Helper produces:   sub-001_ses-01-5120-5120-5120-bem-sol.fif
Pipeline expects:  sub-001-5120-5120-5120-bem-sol.fif
```

Resolve that mapping before running, for example by copying the verified solution to the expected name within the same subject's `bem/` folder. Do not substitute another subject's BEM. The simplest initial setup uses the same subject ID for MEG and FreeSurfer.

Prepare anatomical surfaces and BEM solutions before batch preprocessing:

```bash
python create_missing_bem.py \
  --root_dir /path/to/DATASET \
  --subjects_dir_name MRI/freesurfer
```

Run this in an environment with FreeSurfer configured. Inspect its per-subject results. The main pipeline's missing-BEM branch attempts model creation before its watershed call, so it should not be relied on to bootstrap missing surfaces.

The repeated `5120` tokens are part of the current filename convention. They do not imply a three-layer model: the helper builds a single-layer MEG BEM.

## 5. System-specific and optional sidecars

### MEGIN tSSS parameters

Point `--tsss_dir` (or the launcher's `TSSS_PARAMS`) at a directory containing these exact filenames:

```text
sss_cal.dat
ct_sparse.fif
```

The directory may live outside the dataset. Use calibration and cross-talk files appropriate to the acquisition system and configuration. CTF skips the tSSS/Maxwell branch.

For unprocessed MEGIN data, the pipeline reads an existing head-position file or attempts to compute one from the recording. For the example input, the preferred sidecar is:

```text
sub-001_ses-01_task-rest_run-01_meg_head_pos.pos
```

For input ending in `_tsss.fif` or `_tsss_mc.fif`, lookup first removes that terminal processing suffix before adding `_head_pos.pos`, then tries the actual input stem. Already SSS-processed recordings skip new cHPI position estimation.

### CTF digitization sidecar

When present, this optional raw FIF supplies digitization information to the CTF recording:

```text
sub-001_ses-01_hsp_ready.fif
```

It must contain appropriate fiducials/head-shape information. It does not replace the coregistration transform.

### Coordinate JSON mode

Leave `--json` off for the basic layout above. When enabled, the pipeline expects:

```text
sub-001_ses-01_coordsystem.json
```

It uses the JSON `IntendedFor` string to derive the FreeSurfer subject, progressively removing trailing underscore-delimited components until it finds a matching directory. This mode assumes the project's compatible anatomy/coregistration workflow. It also depends on `nanotools.io_handlers.extract_bids_id`; the standalone `io_handlers.py` in this repository snapshot does not define that helper, so verify the installed nanotools module before using this mode.

### Sensor montage files

Custom ECG/EOG sensor selections are optional. Default lookup starts at:

```text
<root_dir>/montages/<system>/<name>.json
<root_dir>/montages/<system>/<name>.txt
```

Files directly under `montages/` are also supported. Built-in selections can be generated under `montages/<system>/generated/<geometry-fingerprint>/`. See `MONTAGES.md` for formats and selection rules. Use `--montages_dir` for another location.

## 6. Excel and CSV naming

The workbook name is a convenience. The launcher reads the file named by `CSV_FILE`; it does not read `.xlsx` directly. Export a **comma-separated, UTF-8 CSV** with the existing header names.

Each row represents one subject/session/task/run. The exact `sub_ses_run` pattern is:

```text
sub-<label>_ses-<label>_task-<label>_run-<number>
```

Do not include `_meg`, a file extension, or a directory path in this field.

Minimal naming example, not a validated scientific processing recipe:

```csv
sub_ses_run,task,system,run-prepro
sub-001_ses-01_task-rest_run-01,rest,MEGIN,TRUE
sub-001_ses-01_task-rest_run-02,rest,MEGIN,FALSE
```

Important launcher behavior:

- `run-prepro` accepts `true`, `yes`, or `y`, ignoring case. Other values skip the row.
- Fill `task` explicitly, even though the row identifier contains it. The current launcher skips blank cells before its intended task fallback, so a missing `task` can leave the pipeline using its `rest1` default.
- Other column names match CLI options without `--`, for example `ecg_ch`, `inv_method`, `in_file`, and `erm_file`. Do not add free-text comment columns: nonempty cells are passed as CLI arguments.
- Blank cells omit options. Boolean false cells also omit options; they do not override default-true settings. Use the corresponding `no_*` action flag when needed.
- Quote CSV cells containing commas. This is useful for comma-separated channel names or range strings supported by the pipeline.
- The current launcher passes each nonboolean cell as one command-line argument. Space-separated lists such as `line_freqs = 50 100 150` or `crop_tmin = 10 10` need launcher adaptation or direct CLI execution. A single numeric value for `num_proj_ecg` works, but multiple space-separated indices need proper argument splitting.

Set the following paths at the top of `submit_prepro_from_csv.py`:

```python
ROOT_DIR = Path('/path/to/DATASET')
CSV_FILE = ROOT_DIR / 'python_fun' / 'flags-to-use.csv'
TSSS_PARAMS = ROOT_DIR / 'tsss_params' / 'acquisition_config'
PY_SCRIPT = Path('/path/to/nanotools/generic_taskfree.py')
```

Then run:

```bash
python /path/to/nanotools/submit_prepro_from_csv.py
```

The current launcher uses `xvfb-run -a python`, targets a configured Linux environment, runs selected rows sequentially, and stops when a subprocess returns a nonzero exit status. The complete nanotools dependency modules must be available to that Python environment.

## 7. Direct command-line example

This example demonstrates path and identifier handling. Review filters, cropping, artifact references, and other analysis defaults for the actual acquisition before processing.

```bash
python /path/to/nanotools/generic_taskfree.py \
  --root_dir /path/to/DATASET \
  --subject_id sub-001 \
  --session ses-01 \
  --task rest \
  --run run-01 \
  --system MEGIN \
  --tsss_dir /path/to/DATASET/tsss_params/acquisition_config
```

To bypass recording-name discovery, add explicit paths:

```text
--in_file /path/to/actual/resting_recording.fif
--erm_file /path/to/actual/empty_room_recording.fif
```

For CTF, use `--system CTF` and paths to complete `.ds` directories. Explicit input paths do not change the expected anatomy or transform directory.

## 8. Accepted alternatives and discovery limits

- Legacy recording names include `<subject>_<task>_raw*.fif` and `<subject>_erm_raw*.fif` (or `_noise_raw*`). The CTF equivalent uses `.ds`.
- Legacy numeric session directories can be discovered, but downstream sidecar lookup uses the supplied session value literally. Prefer consistent `ses-...` names throughout.
- BIDS-like files without a run entity can be found when calling the pipeline directly without `--run`. The CSV launcher always extracts and passes a run, so runless names need an explicit `in_file` or launcher adaptation.
- Sessionless direct calls are partially supported. Current transform formatting inserts the literal `None` when no session is supplied. Use an explicit session for the recommended complete workflow.
- The automatic search accepts suffixed inputs such as `_meg_digFiltered.fif`. `--prefer digFiltered` favors that token within the selected processing-state group.
- Resting discovery prefers raw inputs, then `_tsss_mc`, then `_tsss`. ERM discovery prefers raw, then `_tsss`, then `_tsss_mc`. `--prefer` does not override this processing-state priority.
- Input discovery excludes names containing `_filt`, `_ssp`, `_bp`, `_notch`, `_proj`, `_src`, `_stc`, `_head_pos`, or `_qc_report` (case-insensitive). Avoid those tokens in original recording labels.
- Multiple matches are ranked rather than rejected. Review the printed selected paths, or use `in_file` and `erm_file` to remove ambiguity.

## 9. Generated files and reruns

You do not need to create `derivatives/`, `logfiles/`, or `logs/` in advance. Parent locations must be writable. Some outputs are saved **beside the MEG and ERM inputs**, not under `derivatives/`.

```text
DATASET/
├── MEG/sub-001/ses-01/meg/
│   ├── <recording-stem>_tsss_mc.fif           # MEGIN, when computed
│   ├── <empty-room-stem>_tsss.fif            # MEGIN, when computed
│   ├── <active-MEG-stem>_notch_bp_SSP.fif
│   └── <active-ERM-stem>_notch_bp_SSP.fif
├── derivatives/sub-001/ses-01/
│   ├── <active-MEG-stem>_src.fif
│   └── beamformer/                          # Or MNE, dSPM, sLORETA
│       ├── stc/
│       │   ├── <active-MEG-stem>_beamformer_stc-lh.stc
│       │   └── <active-MEG-stem>_beamformer_stc-rh.stc
│       └── report/
│           ├── <active-MEG-stem>_QC_report.html
│           ├── PSD_band_dist<active-MEG-stem>.png
│           └── <original-input-stem>_<prefer>_hyperparameters.txt
├── logfiles/
│   └── sub-001_ses-01_run-01.log
└── logs/
    └── generic_taskfree_sub-001_<timestamp>.txt
```

`active-MEG-stem` means the filename without its extension after any tSSS input selection. It can include `_tsss_mc`; the hyperparameter filename instead uses the originally selected input stem. The default `prefer` value adds `_any` before `_hyperparameters.txt`.

The diagnostic text log is written on relevant error paths, so it need not exist for every successful run. The main HTML report is currently saved inside the successful source-PSD branch; a missing report requires investigation even if some sensor outputs exist.

When adjusting parameters, select the affected CSV rows and review whether cached tSSS/source results require recomputation with `overwrite`. Some sensor/report outputs overwrite existing files independently. Preserve the previous CSV and needed outputs before a comparison run.

Launcher log names omit the task and open in write mode. Two rows with the same subject/session/run but different tasks, or a rerun of a row, reuse the same log path. Archive logs or adapt their naming if those runs must remain separately traceable.

## 10. Before the first run

1. Match the folder names, recording entities, and CSV identifiers exactly.
2. Confirm the intended MEG and empty-room paths, including the full CTF dataset where applicable.
3. Confirm the FreeSurfer subject mapping, BEM filename, and valid coregistration transform.
4. For MEGIN tSSS, supply the correct parameter files and usable head-position information.
5. Set the launcher paths and verify the complete nanotools Python environment.
6. Run one representative recording, inspect the selected paths and QC report, then expand to the cohort.

Implementation references: `generic_taskfree.py` (`find_meg`, `find_erm`, `_head_pos_candidates`, `preprocess_subject`, CLI entry point), `submit_prepro_from_csv.py`, `create_missing_bem.py`, and `MONTAGES.md`.

---
Sensor montages for ECG/EOG

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

---

## Installation

### Option A — Editable install (recommended for development)
#From a local clone:
#```bash
#pip install -e .
