"""Build embedded spectral reference data (run once, offline at runtime).

Sources (CIE standard tables, fetched at build time only):
- CIE 1931 2-degree standard observer colour-matching functions, 5 nm grid
  (380-780 nm), from the RIT/CIE data distributed by colour-science.
- CIE standard illuminants A, D65, F11 spectral power distributions,
  5 nm grid (380-780 nm), CIE 15:2004 tables distributed by colour-science.

Output: app/data/*.json
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data"

CMF_URL = (
    "https://raw.githubusercontent.com/colour-science/color-data/"
    "master/observers/cie-1931-2.yaml"
)
SDS_URL = (
    "https://raw.githubusercontent.com/colour-science/colour/develop/"
    "colour/colorimetry/datasets/illuminants/sds.py"
)

W_MIN, W_MAX, STEP = 380.0, 780.0, 5.0
GRID = [round(W_MIN + i * STEP, 1) for i in range(int((W_MAX - W_MIN) / STEP) + 1)]
# API name -> key in colour's CIE 15:2004 SPD table
ILLUMINANTS = {"A": "A", "D65": "D65", "F11": "FL11"}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def build_cmf(text: str) -> dict:
    """Parse '- [wavelength_m, x_bar, y_bar, z_bar]' rows, resample to 5 nm."""
    row_re = re.compile(
        r"^\s*-\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,"
        r"\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]"
    )
    table: dict[float, tuple[float, float, float]] = {}
    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        wl_nm = round(float(m.group(1)) * 1e9, 3)
        table[wl_nm] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    x_bar, y_bar, z_bar = [], [], []
    for wl in GRID:
        if wl not in table:
            raise RuntimeError(f"CMF grid point missing: {wl}")
        x, y, z = table[wl]
        x_bar.append(x)
        y_bar.append(y)
        z_bar.append(z)
    return {
        "wavelengths_nm": GRID,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "z_bar": z_bar,
        "source": "CIE 1931 2 deg standard observer (RIT/CIE 15 table), 5 nm grid",
    }


def extract_illuminant_dicts(source: str) -> dict[str, dict]:
    """Statically parse sds.py and pull the literal SPD dicts by source key."""
    source_names = set(ILLUMINANTS.values())
    tree = ast.parse(source)
    wanted: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in source_names:
                if isinstance(value, ast.Dict):
                    try:
                        d = ast.literal_eval(value)
                    except ValueError:
                        continue
                    if d and all(isinstance(k, (int, float)) for k in d):
                        wanted[key.value] = {float(k): float(v) for k, v in d.items()}
    return wanted


def resample_spd(spd: dict[float, float], name: str) -> list[float]:
    missing = [wl for wl in GRID if wl not in spd]
    if missing:
        raise RuntimeError(f"{name}: missing SPD grid points: {missing[:5]} ...")
    return [spd[wl] for wl in GRID]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cmf = build_cmf(fetch(CMF_URL))
    (OUT / "cmf_cie1931_2deg.json").write_text(
        json.dumps(cmf, indent=1), encoding="utf-8"
    )

    source_spds = extract_illuminant_dicts(fetch(SDS_URL))
    found = set(source_spds)
    if not set(ILLUMINANTS.values()).issubset(found):
        raise RuntimeError(
            f"missing illuminant tables: {set(ILLUMINANTS.values()) - found}")

    bundle = {
        "wavelengths_nm": GRID,
        "illuminants": {
            api_name: {
                "name": api_name,
                "aliases": [source_name] if source_name != api_name else [],
                "power": resample_spd(source_spds[source_name], api_name),
            }
            for api_name, source_name in ILLUMINANTS.items()
        },
        "source": "CIE 15:2004 standard illuminant SPD tables, 5 nm grid",
    }
    (OUT / "illuminants.json").write_text(
        json.dumps(bundle, indent=1), encoding="utf-8"
    )

    # quick sanity print
    for api_name in ILLUMINANTS:
        powers = bundle["illuminants"][api_name]["power"]
        print(f"{api_name:5s} {len(powers)} pts  sum={sum(powers):.2f}")
    print(f"CMF   {len(cmf['x_bar'])} pts  sum(y_bar)={sum(cmf['y_bar']):.3f}")


if __name__ == "__main__":
    main()
