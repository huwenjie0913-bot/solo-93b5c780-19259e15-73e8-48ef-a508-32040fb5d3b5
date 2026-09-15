"""Spectral review engine: grid alignment, per-illuminant colorimetry,
metamerism flagging and dominant-band attribution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colorimetry import (
    trapz,
    WL_MAX,
    WL_MIN,
    ciede2000,
    cmf_on_grid,
    illuminant_on_grid,
    interp_linear,
    make_grid,
    tristimulus,
    xyz_to_lab,
)
from .validation import (
    RequestValidationError,
    Spectrum,
    parse_spectrum,
    validate_grid,
    validate_illuminant_ref,
)


@dataclass
class IlluminantInput:
    key: str            # unique key used in responses ("D65", "custom:mall-LED")
    display_name: str
    kind: str           # "builtin" | "custom"
    spectrum: Spectrum | None = None
    ref_name: str | None = None


def resolve_illuminants(specs, field_prefix: str) -> list[IlluminantInput]:
    """Validate IlluminantSpec list, returning resolved inputs."""
    issues: list[dict] = []
    resolved: list[IlluminantInput] = []
    seen_keys: set[str] = set()

    for i, spec in enumerate(specs):
        field = f"{field_prefix}[{i}]"
        if bool(spec.name) == bool(spec.custom is not None):
            issues.append({
                "field": field,
                "code": "illuminant_ambiguous",
                "reason": "provide exactly one of 'name' (built-in) or 'custom' (uploaded SPD)",
            })
            continue
        if spec.name:
            name = spec.name.strip().upper()
            if not validate_illuminant_ref(name, f"{field}.name", WL_MIN, WL_MAX, issues):
                continue
            key = name
            display = spec.label or name
        else:
            sp = parse_spectrum(
                spec.custom.model_dump(),
                field=f"{field}.custom",
                label=spec.label or f"custom[{i}]",
                is_reflectance=False,
                issues=issues,
            )
            if sp is None:
                continue
            if not np.any(sp.values > 0):
                issues.append({
                    "field": f"{field}.custom.values",
                    "code": "zero_illuminant_power",
                    "reason": "custom illuminant spectral power is zero everywhere",
                })
                continue
            key = f"custom:{i}"
            display = spec.label or f"custom[{i}]"
            resolved_sp = sp
        if key in seen_keys:
            issues.append({
                "field": field,
                "code": "duplicate_illuminant",
                "reason": f"illuminant {key} is listed more than once",
            })
            continue
        seen_keys.add(key)
        if spec.name:
            resolved.append(IlluminantInput(key=key, display_name=display,
                                            kind="builtin", ref_name=name))
        else:
            resolved.append(IlluminantInput(key=key, display_name=display,
                                            kind="custom", spectrum=resolved_sp))

    if issues:
        raise RequestValidationError(issues)
    if not resolved:
        raise RequestValidationError([{
            "field": field_prefix,
            "code": "no_valid_illuminant",
            "reason": "at least one valid illuminant is required",
        }])
    return resolved


def align_reflectance(spectrum: Spectrum, grid: np.ndarray) -> np.ndarray:
    r = interp_linear(spectrum.wavelengths, spectrum.values, grid)
    return np.clip(r, 0.0, 1.0)


def _band_edges(grid: np.ndarray, width_nm: float) -> list[tuple[int, int]]:
    """Partition grid points into wavelength bands of ~width_nm."""
    edges = list(np.arange(grid[0], grid[-1] + 1e-9, width_nm))
    if edges[-1] < grid[-1]:
        edges.append(float(grid[-1]) + 0.0001)
    bands: list[tuple[int, int]] = []
    start = 0
    for e in edges[1:]:
        idx = np.searchsorted(grid, e, side="left")
        if idx > start:
            bands.append((start, idx))
            start = idx
    if start < grid.size:
        if bands:
            bands[-1] = (bands[-1][0], grid.size)
        else:
            bands.append((start, grid.size))
    return bands


def _band_contributions(grid: np.ndarray, target: np.ndarray, sample: np.ndarray,
                        power: np.ndarray, x_bar: np.ndarray, y_bar: np.ndarray,
                        z_bar: np.ndarray, width_nm: float
                        ) -> list[dict]:
    """Attribute the target-sample XYZ displacement to wavelength bands.

    For band b the trapezoidal contribution to Delta XYZ is computed. Its
    magnitude in the direction of the total Delta XYZ vector (per channel
    scaled by the white-normalised Lab sensitivity ordering) is approximated
    by the Euclidean norm of the band Delta XYZ; the sign of the projection
    onto total Delta XYZ distinguishes bands driving the observed mismatch.
    """
    delta_per_point = np.column_stack([
        power * x_bar * (sample - target),
        power * y_bar * (sample - target),
        power * z_bar * (sample - target),
    ])
    denom = trapz(power * y_bar, grid)
    bands = _band_edges(grid, width_nm)
    total_dxyz = 100.0 * np.array([
        trapz(delta_per_point[:, k], grid) for k in range(3)
    ]) / denom
    total_norm = float(np.linalg.norm(total_dxyz))

    results: list[dict] = []
    for start, end in bands:
        dxyz = 100.0 * np.array([
            _trapz_slice(delta_per_point[:, k], grid, start, end) for k in range(3)
        ]) / denom
        magnitude = float(np.linalg.norm(dxyz))
        projection = float(np.dot(dxyz, total_dxyz) / total_norm) if total_norm > 0 else 0.0
        results.append({
            "wavelength_range_nm": [float(grid[start]), float(grid[end - 1])],
            "delta_xyz": [round(v, 6) for v in dxyz],
            "magnitude": round(magnitude, 6),
            "signed_projection": round(projection, 6),
        })
    results.sort(key=lambda b: b["magnitude"], reverse=True)
    return results


def _trapz_slice(values: np.ndarray, grid: np.ndarray, start: int, end: int) -> float:
    """Trapezoidal integral over index range [start, end), shared edge included."""
    stop = min(end + 1, grid.size)
    return float(trapz(values[start:stop], grid[start:stop]))


def _common_support_range(target: Spectrum, sample: Spectrum,
                          illuminants: list[IlluminantInput],
                          step_nm: float) -> tuple[float, float]:
    """Intersection of every input series' measured wavelength support,
    snapped outward to regular grid nodes inside 380-780 nm."""
    lo = max(float(target.wavelengths.min()),
             float(sample.wavelengths.min()))
    hi = min(float(target.wavelengths.max()),
             float(sample.wavelengths.max()))
    for ill in illuminants:
        if ill.kind == "custom":
            lo = max(lo, float(ill.spectrum.wavelengths.min()))
            hi = min(hi, float(ill.spectrum.wavelengths.max()))

    start = WL_MIN + np.ceil((lo - WL_MIN) / step_nm - 1e-9) * step_nm
    end = WL_MIN + np.floor((hi - WL_MIN) / step_nm + 1e-9) * step_nm
    start = min(max(start, WL_MIN), WL_MAX)
    end = min(max(end, WL_MIN), WL_MAX)
    if end - start < step_nm:
        raise RequestValidationError([{
            "field": "wavelength_range_nm",
            "code": "insufficient_overlap",
            "reason": (
                "the target, sample and custom illuminants do not share enough "
                f"overlapping wavelength support for a {step_nm:g} nm grid "
                f"(intersection is [{lo:g}, {hi:g}] nm)"
            ),
            "intersection_nm": [round(float(lo), 2), round(float(hi), 2)],
        }])
    return round(float(start), 4), round(float(end), 4)


def evaluate_pair(target: Spectrum, sample: Spectrum,
                  illuminants: list[IlluminantInput],
                  step_nm: float, tolerance: float,
                  band_width_nm: float, top_bands: int,
                  wl_min: float = WL_MIN, wl_max: float = WL_MAX) -> dict:
    grid_issues = validate_grid(step_nm, wl_min, wl_max)
    if grid_issues:
        raise RequestValidationError(grid_issues)

    # Restrict the integration grid to the range every input (including any
    # custom SPD) actually covers; reflectance is guaranteed to span 390-770.
    wl_min, wl_max = _common_support_range(target, sample, illuminants, step_nm)
    grid = make_grid(step_nm, wl_min, wl_max)
    r_t = align_reflectance(target, grid)
    r_s = align_reflectance(sample, grid)
    x_bar, y_bar, z_bar = cmf_on_grid(grid)

    per_ill: list[dict] = []
    for ill in illuminants:
        if ill.kind == "builtin":
            power = illuminant_on_grid(ill.ref_name, grid)
        else:
            power = interp_linear(ill.spectrum.wavelengths, ill.spectrum.values, grid)
            if trapz(power * y_bar, grid) <= 0:
                raise RequestValidationError([{
                    "field": f"illuminants:{ill.key}",
                    "code": "zero_normalisation_denominator",
                    "reason": (
                        "normalisation denominator integral "
                        f"S(lambda)y_bar(lambda) is zero for illuminant {ill.display_name!r}; "
                        "the SPD must emit within the visible (especially 450-650 nm) range"
                    ),
                }])

        xyz_t = tristimulus(r_t, power, grid, x_bar, y_bar, z_bar)
        xyz_s = tristimulus(r_s, power, grid, x_bar, y_bar, z_bar)
        # Each illuminant uses its own adapted white (the perfect reflector
        # under that SPD); otherwise a neutral would look yellow under "A".
        white_xyz = tristimulus(
            np.ones_like(grid), power, grid, x_bar, y_bar, z_bar)
        lab_t = xyz_to_lab(xyz_t, tuple(float(v) for v in white_xyz))
        lab_s = xyz_to_lab(xyz_s, tuple(float(v) for v in white_xyz))
        de00 = ciede2000(lab_t, lab_s)

        bands = _band_contributions(grid, r_t, r_s, power, x_bar, y_bar, z_bar,
                                    band_width_nm)[:top_bands]

        per_ill.append({
            "illuminant": ill.key,
            "illuminant_label": ill.display_name,
            "kind": ill.kind,
            "white_xyz": [round(float(v), 4) for v in white_xyz],
            "xyz_target": [round(v, 4) for v in xyz_t],
            "xyz_sample": [round(v, 4) for v in xyz_s],
            "lab_target": {k: round(v, 4) for k, v in lab_t.as_dict().items()},
            "lab_sample": {k: round(v, 4) for k, v in lab_s.as_dict().items()},
            "delta_lab": {
                "dL": round(lab_s.L - lab_t.L, 4),
                "da": round(lab_s.a - lab_t.a, 4),
                "db": round(lab_s.b - lab_t.b, 4),
            },
            "delta_e00": round(de00, 4),
            "pass": bool(de00 <= tolerance + 1e-9),
            "tolerance_de00": tolerance,
            "top_bands": bands,
        })

    pass_flags = [r["pass"] for r in per_ill]
    metameric = any(pass_flags) and not all(pass_flags)
    passing = [r["illuminant"] for r in per_ill if r["pass"]]
    failing = [r["illuminant"] for r in per_ill if not r["pass"]]

    worst = max(per_ill, key=lambda r: r["delta_e00"])
    best = min(per_ill, key=lambda r: r["delta_e00"])

    return {
        "grid_nm": list(grid),
        "target_reflectance": [round(float(v), 6) for v in r_t],
        "sample_reflectance": [round(float(v), 6) for v in r_s],
        "results_by_illuminant": per_ill,
        "metamerism_risk": {
            "flag": bool(metameric),
            "passes_under": passing,
            "fails_under": failing,
            "worst_illuminant": worst["illuminant"],
            "worst_de00": worst["delta_e00"],
            "best_illuminant": best["illuminant"],
            "best_de00": best["delta_e00"],
            "spread_de00": round(worst["delta_e00"] - best["delta_e00"], 4),
            "note": (
                "sample passes under at least one requested illuminant but fails "
                "under another: illuminant-dependent (metameric) mismatch"
                if metameric else
                ("fails under every requested illuminant" if failing else
                 "passes under every requested illuminant")
            ),
        },
        "max_de00": worst["delta_e00"],
        "mean_de00": round(float(np.mean([r["delta_e00"] for r in per_ill])), 4),
        "pass_all": all(pass_flags),
    }
