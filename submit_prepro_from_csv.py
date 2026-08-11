#!/usr/bin/env python3

import re
import sys
import csv
import subprocess
from pathlib import Path

path2project= sys.argv[1]

ROOT_DIR = Path(path2project)#"/home/isaant/Documents/BRHRN" 
CSV_FILE = ROOT_DIR / "python_fun" / "flags-to-use_BRHRN.csv"
TSSS_PARAMS = Path("/home/isaant/mne_nanotools/tsss_params/2023")

PY_SCRIPT = Path("/home/isaant/mne_nanotools/generic_taskfree.py")

LOG_DIR = ROOT_DIR / "logfiles"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def is_empty(v):
    return v is None or str(v).strip() == ""


def is_true(v):
    return str(v).strip().lower() in {"true", "yes", "y"}


def is_false(v):
    return str(v).strip().lower() in {"false", "no", "n"}


def clean_value(v):
    return str(v).strip()


with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row_idx, row_data in enumerate(reader, start=2):

        sub_ses_run = row_data.get("sub_ses_run")
        run_prepro = row_data.get("run-prepro")

        if is_empty(sub_ses_run):
            continue

        if not is_true(run_prepro):
            print(f"Skipping row {row_idx}: run-prepro is not true")
            continue

        match = re.match(
            r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_(?P<run>run-[^_]+)$",
            str(sub_ses_run).strip()
        )

        if not match:
            raise ValueError(f"Could not parse sub_ses_run in row {row_idx}: {sub_ses_run}")

        subject_id = match.group("subject")
        session = match.group("session")
        run = match.group("run")
        parsed_task = match.group("task")

        python_args = [
            "--root_dir", str(ROOT_DIR) + "/",
            "--subject_id", subject_id,
            "--session", session,
            "--run", run,
            "--tsss_dir", str(TSSS_PARAMS) + "/",
        ]

        for flag, value in row_data.items():

            if flag is None:
                continue

            flag = str(flag).strip()

            if flag == "" or flag.startswith("Unnamed"):
                continue

            if flag in {"sub_ses_run", "run-prepro"}:
                continue

            if is_empty(value):
                continue

            if flag == "task":
                value = value if not is_empty(value) else parsed_task

            if is_true(value):
                python_args.append(f"--{flag}")
            elif is_false(value):
                continue
            else:
                python_args.extend([f"--{flag}", clean_value(value)])

        logfile = LOG_DIR / f"{subject_id}_{session}_{run}.log"

        cmd = [
            "xvfb-run","-a","python", str(PY_SCRIPT),
            *python_args,
        ]

        print("=" * 60)
        print(f"Running row {row_idx}: {subject_id} {session} {run}")
        print(f"Log file: {logfile}")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 60)

        with open(logfile, "w") as log:
            log.write("Command:\n")
            log.write(" ".join(cmd) + "\n\n")
            log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(line, end="")
                log.write(line)

            return_code = process.wait()

        if return_code != 0:
            print(f"❌ Failed row {row_idx}: {subject_id} {session} {run}")
            print(f"See log: {logfile}")
            raise subprocess.CalledProcessError(return_code, cmd)

        print(f"✅ Finished row {row_idx}: {subject_id} {session} {run}")