"""Named MEG sensor selections and temporary ECG/EOG references.

These are channel groups, not MNE DigMontage objects. See MONTAGES.md.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings

import mne
import numpy as np
from mne.io.constants import FIFF


REGIONS = tuple(f"{side}-{region}" for side in ("Left", "Right")
                for region in ("frontal", "parietal", "occipital", "temporal"))
ALIASES = {"oeg-montage": "eog_montage", "ocg-monatge": "ecg_montage",
           "ecg-monatge": "ecg_montage", "ocg-montage": "ecg_montage",
           "eog-montage": "eog_montage", "ecg-montage": "ecg_montage"}


def _key(name):
    return re.sub(r"[\s_]+", "-", name.strip()).lower()


def _canonical(name):
    key = _key(name)
    return ALIASES.get(key, next((r for r in REGIONS if _key(r) == key), name))


def _sensor_key(name):
    # CTF serial suffixes and old/new Vectorview spacing.
    return name.split("-")[0].replace(" ", "").upper()


def _sensor_picks(info, system):
    # CTF primary axial gradiometers are represented as mag by some MNE versions.
    return mne.pick_types(info, meg="mag" if system == "MEGIN" else True,
                          ref_meg=False, exclude=[])


def _region(info, system, name):
    picks = _sensor_picks(info, system)
    if system == "MEGIN":
        names = {_sensor_key(n) for n in mne.read_vectorview_selection(name, info=info)}
        return [i for i in picks if _sensor_key(info["ch_names"][i]) in names]
    side, region = name.split("-")
    prefix = "M" + side[0] + {"frontal": "F", "parietal": "P",
                                "occipital": "O", "temporal": "T"}[region]
    return [i for i in picks if _sensor_key(info["ch_names"][i]).startswith(prefix)]


def _lower_rows(info, picks, n_rows):
    """Estimate inferior-to-superior rows as bands of half a sensor spacing.

    Use head coordinates, never device axes implicitly. Irregular helmets do
    not define universal rows; the exported list is intended for review.
    """
    xyz = []
    for i in picks:
        ch = info["chs"][i]
        pos = ch["loc"][:3].copy()
        if ch["coord_frame"] == FIFF.FIFFV_COORD_DEVICE:
            trans = info.get("dev_head_t")
            if trans is None:
                raise ValueError("Row selection requires dev_head_t or head-coordinate sensors")
            pos = mne.transforms.apply_trans(trans, pos)
        elif ch["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError("Unsupported sensor coordinate frame for row selection")
        xyz.append(pos)
    xyz = np.asarray(xyz)
    if len(xyz) < 2 or not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz, axis=1) == 0):
        raise ValueError("Not enough valid sensor positions to estimate montage rows")
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    distances[distances < 1e-6] = np.inf
    spacing = np.median(distances.min(axis=1))
    if not np.isfinite(spacing):
        raise ValueError("Sensor positions do not define a spacing")
    z = xyz[:, 2]
    remaining = np.argsort(z)
    selected = []
    for _ in range(n_rows):
        if not len(remaining):
            break
        mask = z[remaining] <= z[remaining[0]] + spacing * 0.5
        selected.extend(remaining[mask])
        remaining = remaining[~mask]
    return [picks[i] for i in selected]


def load_montage(info, name, directory, system):
    """Load a user list or generate a layout-specific built-in selection."""
    system = system.upper()
    if system not in ("MEGIN", "CTF"):
        raise ValueError("Montages support MEGIN and CTF")
    name = _canonical(name)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a montage name, not a path")
    directory = Path(directory).expanduser()
    path = None
    # Explicit system lists override project-wide lists; generated lists are last.
    for folder in (directory / system, directory):
        matches = sorted(p for p in folder.glob("*") if p.suffix in (".json", ".txt")
                         and _key(p.stem) == _key(name))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous montage files: {matches}")
        if matches:
            path = matches[0]
            break
    if path is None:
        if name not in (*REGIONS, "eog_montage", "ecg_montage"):
            raise ValueError(f"Unknown channel or montage {name!r} in {directory}")
        # Include positions/transform: a different helmet or head pose gets its own list.
        picks = _sensor_picks(info, system)
        signature = [(info["ch_names"][i], info["chs"][i]["loc"][:3].tolist(),
                      int(info["chs"][i]["coord_frame"])) for i in picks]
        trans = info.get("dev_head_t")
        signature.append(None if trans is None else trans["trans"].tolist())
        digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
        path = directory / system / "generated" / digest / f"{name}.json"
        if not path.exists():
            if name == "eog_montage":
                picks = _region(info, system, "Left-frontal") + _region(info, system, "Right-frontal")
                if system == "CTF":
                    picks += [i for i in _sensor_picks(info, system)
                              if _sensor_key(info["ch_names"][i]).startswith("MZF")]
                picks = list(dict.fromkeys(picks))
                picks = _lower_rows(info, picks, 3)
            elif name == "ecg_montage":
                picks = _lower_rows(info, _region(info, system, "Left-temporal"), 2)
            else:
                picks = _region(info, system, name)
            if not picks:
                raise ValueError(f"No sensors found for {system} {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"system": system, "name": name,
                       "method": "regional selection; artifact rows estimated bottom-to-top, half-spacing bands",
                       "channels": [info["ch_names"][i] for i in picks]}
            # Atomic replacement avoids partially written JSON in parallel jobs.
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                json.dump(payload, f, indent=2)
                temporary = f.name
            os.replace(temporary, path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and payload.get("system", system).upper() != system:
            raise ValueError(f"Montage {path} is for a different MEG system")
        names = payload.get("channels") if isinstance(payload, dict) else payload
    else:
        names = [n.strip() for line in path.read_text().splitlines()
                 for n in line.split("#", 1)[0].split(",") if n.strip()]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Montage {path} must contain a nonempty channel list")
    available = {}
    for i in _sensor_picks(info, system):
        available.setdefault(_sensor_key(info["ch_names"][i]), []).append(info["ch_names"][i])
    selected = []
    for name in names:
        matches = available.get(_sensor_key(name), [])
        if len(matches) != 1:
            raise ValueError(f"Montage {path}: missing, ambiguous or incompatible sensor {name!r}")
        if matches[0] not in info["bads"] and matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise ValueError(f"Montage {path}: all selected sensors are marked bad")
    print(f"→ Montage {path}: {', '.join(selected)}")
    return selected


def prepare_artifact_reference(raw, selection, kind, directory, system):
    """Return (MNE ch_name, temporary names); preserve explicit real channels.

    Missing conventional ECG/EOG names fall back to typed channels, then the
    default montage. Unknown custom names fail instead of hiding typos.
    """
    names = ([n.strip() for n in selection.split(",") if n.strip()]
             if isinstance(selection, str) else list(selection or []))
    good = [n for n in names if n in raw.ch_names and n not in raw.info["bads"]]
    if good and len(good) == len(names):
        if kind == "ecg" and len(good) != 1:
            raise ValueError("ECG expects one physical channel or one montage name")
        return (good[0] if len(good) == 1 else good), []
    fallback = not names or names == ["auto"] or all(
        re.fullmatch(r"(?:EOG|ECG|HEOG|VEOG)\d*", n, re.I) for n in names)
    if fallback:
        typed = mne.pick_types(raw.info, meg=False, ref_meg=False, exclude="bads", **{kind: True})
        if len(typed):
            good = [raw.ch_names[i] for i in typed]
            return (good[0] if kind == "ecg" or len(good) == 1 else good), []
        montage = f"{kind}_montage"
        warnings.warn(f"No usable {kind.upper()} channel; using {montage}", RuntimeWarning)
    elif len(names) == 1:
        montage = names[0]
    else:
        raise ValueError(f"Missing/bad {kind.upper()} channels: {names}")
    channels = load_montage(raw.info, montage, directory, system)
    data = raw.get_data(picks=channels)
    data -= data.mean(axis=1, keepdims=True)
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError(f"Montage {montage} has no finite, nonzero signal")
    # PC1 avoids cancellation of opposite MEG field polarities in a plain mean.
    _, vectors = np.linalg.eigh(data @ data.T)
    weights = vectors[:, -1]
    weights *= np.sign(weights[np.argmax(np.abs(weights))])
    signal = weights @ data
    # The synthetic channel is a normalized detection proxy, not measured volts.
    signal = signal / np.std(signal) * 1e-4
    name = f"{kind.upper()}-MONTAGE"
    while name in raw.ch_names:
        name += "_"
    ref = mne.io.RawArray(signal[None], mne.create_info([name], raw.info["sfreq"], [kind]),
                          first_samp=raw.first_samp, verbose=False)
    raw.add_channels([ref], force_update_info=True)
    return name, [name]
