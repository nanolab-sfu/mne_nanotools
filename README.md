# mne_nanotools

**mne_nanotools** is a lightweight, reusable Python toolkit maintained by the NanoLab (SFU) to support **MEG workflows built around MNE-Python**, with utilities for **pre-processing, post-processing, I/O handling, and pipeline orchestration**. The repository is designed to be imported inside larger analysis scripts rather than used as a standalone application.  

**Author:** Santiago Isaac Flores Alonso  
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

## Installation

### Option A — Editable install (recommended for development)
#From a local clone:
#```bash
#pip install -e .
