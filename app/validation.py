"""Request validation with machine-readable, field-specific errors."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .colorimetry import (
    BUILTIN_ILLUMINANTS,
    REFLECTANCE_MAX,
    REFLECTANCE_MIN,
    WL_MAX,
    WL_MIN,
    illuminant_table,
)

# A reflectance spectrum useful for colorimetry must span essentially the
# whole visible range. Custom illuminant SPDs, however, may legitimately emit
# only part of the visible range (e.g. narrow-band lamps); the integration grid
# is then restricted to the common overlap of all input series.
# Reflectance spectra are required to cover the practical spectrophotometer
# range 390-770 nm. Data reaching closer to the reference-table edges is used
# directly; the integration grid snaps to whatever all inputs support.
MIN_REFLECTANCE_COVERAGE = (WL_MIN + 10.0, WL_MAX - 10.0)  # 390-770 nm
MIN_ILLUMINANT_COVERAGE = 80.0  # custom SPD must span at least this many nm
MIN_SAMPLE_POINTS = 10
MIN_ILLUMINANT_POINTS = 10


class RequestValidationError(Exception):
    """Collects one or more validation issues for a 422 response."""

    def __init__(self, issues: Sequence[dict]):
        self.issues = list(issues)
        super().__init__("; ".join(str(i.get("reason", "")) for i in self.issues))


@dataclass(frozen=True)
class Spectrum:
    wavelengths: np.ndarray
    values: np.ndarray
    label: str  # e.g. "target", "sample[2]", "illuminant[custom:A]"


def _issue(field: str, reason: str, code: str, **extra) -> dict:
    issue = {"field": field, "code": code, "reason": reason}
    issue.update(extra)
    return issue


def parse_spectrum(pairs, field: str, label: str, *,
                   is_reflectance: bool, issues: list[dict]) -> Spectrum | None:
    """Validate a {wavelengths, values} style payload into a Spectrum."""
    if is_reflectance:
        coverage_required = MIN_REFLECTANCE_COVERAGE
        min_points = MIN_SAMPLE_POINTS
        kind = "reflectance"
    else:
        coverage_required = None  # custom SPDs may cover a partial range
        min_points = MIN_ILLUMINANT_POINTS
        kind = "illuminant SPD"
    ok = True

    if not isinstance(pairs, dict):
        issues.append(_issue(field, "must be an object", "invalid_object"))
        return None

    wavelengths = pairs.get("wavelengths")
    values = pairs.get("values")

    if not isinstance(wavelengths, list) or not wavelengths:
        issues.append(_issue(f"{field}.wavelengths",
                             "wavelengths must be a non-empty array of numbers",
                             "missing_or_empty"))
        ok = False
    if not isinstance(values, list) or not values:
        issues.append(_issue(f"{field}.values",
                             "values must be a non-empty array of numbers",
                             "missing_or_empty"))
        ok = False
    if not ok:
        return None

    if len(wavelengths) != len(values):
        issues.append(_issue(
            field,
            f"wavelengths/values length mismatch: {len(wavelengths)} vs {len(values)}",
            "length_mismatch",
            wavelengths_count=len(wavelengths), values_count=len(values),
        ))
        return None

    try:
        wl = np.asarray([float(v) for v in wavelengths], dtype=float)
    except (TypeError, ValueError):
        issues.append(_issue(f"{field}.wavelengths",
                             "all wavelengths must be numbers", "non_numeric"))
        return None
    try:
        val = np.asarray([float(v) for v in values], dtype=float)
    except (TypeError, ValueError):
        issues.append(_issue(f"{field}.values",
                             "all values must be numbers", "non_numeric"))
        return None

    if not np.all(np.isfinite(wl)):
        bad = [w for w in wavelengths if not _is_finite(w)]
        issues.append(_issue(f"{field}.wavelengths",
                             "wavelengths contain NaN or infinite values",
                             "non_finite", values=bad[:10]))
        ok = False
    if not np.all(np.isfinite(val)):
        bad = [v for v in values if not _is_finite(v)]
        issues.append(_issue(f"{field}.values",
                             "values contain NaN or infinite values",
                             "non_finite", values=bad[:10]))
        ok = False
    if not ok:
        return None

    unique, counts = np.unique(wl, return_counts=True)
    duplicated = unique[counts > 1]
    if duplicated.size:
        issues.append(_issue(
            f"{field}.wavelengths",
            f"duplicate wavelength entries at {_fmt_wl(duplicated.tolist())} nm",
            "duplicate_wavelength",
            wavelengths_nm=[float(x) for x in duplicated],
        ))
        ok = False

    if wl.size > 1:
        order = np.argsort(wl)
        sorted_wl = wl[order]
        if np.any(np.diff(sorted_wl) <= 0):
            issues.append(_issue(f"{field}.wavelengths",
                                 "wavelengths are not strictly monotonic",
                                 "non_monotonic"))
            ok = False
    else:
        issues.append(_issue(field,
                             "at least two wavelength samples are required",
                             "too_few_points", points=1))
        ok = False

    # wavelength gaps (holes within the measured range larger than 20 nm)
    if wl.size >= 2:
        sorted_wl = np.sort(wl)
        gaps = np.diff(sorted_wl)
        worst = int(np.argmax(gaps))
        if gaps[worst] > 20.0 + 1e-9:
            issues.append(_issue(
                f"{field}.wavelengths",
                f"wavelength gap of {gaps[worst]:g} nm between "
                f"{sorted_wl[worst]:g} nm and {sorted_wl[worst + 1]:g} nm "
                "(maximum allowed is 20 nm)",
                "wavelength_gap",
                gap_start_nm=float(sorted_wl[worst]),
                gap_end_nm=float(sorted_wl[worst + 1]),
                gap_nm=float(gaps[worst]),
            ))
            ok = False

    if wl.size < min_points:
        issues.append(_issue(field,
                             f"at least {min_points} samples are required, got {wl.size}",
                             "too_few_points", points=int(wl.size)))
        ok = False

    if coverage_required is not None:
        lo, hi = coverage_required
        if wl.min() > lo + 1e-9 or wl.max() < hi - 1e-9:
            issues.append(_issue(
                field,
                f"spectral coverage [{wl.min():g}, {wl.max():g}] nm is insufficient; "
                f"the full visible range [{lo:g}, {hi:g}] nm must be covered",
                "insufficient_coverage",
                range_nm=[float(wl.min()), float(wl.max())],
                required_range_nm=[lo, hi],
            ))
            ok = False
    elif wl.max() - wl.min() < MIN_ILLUMINANT_COVERAGE - 1e-9:
        issues.append(_issue(
            field,
            f"custom illuminant covers only {wl.max() - wl.min():g} nm "
            f"[{wl.min():g}, {wl.max():g}]; at least {MIN_ILLUMINANT_COVERAGE:g} nm "
            "of spectral support is required",
            "insufficient_coverage",
            range_nm=[float(wl.min()), float(wl.max())],
            minimum_span_nm=MIN_ILLUMINANT_COVERAGE,
        ))
        ok = False

    if is_reflectance:
        out_low = np.nonzero(val < REFLECTANCE_MIN - 1e-9)[0]
        out_high = np.nonzero(val > REFLECTANCE_MAX + 1e-9)[0]
        if out_low.size or out_high.size:
            details = []
            for idx in out_low[:5]:
                details.append({"wavelength_nm": float(wl[idx]), "value": float(val[idx])})
            for idx in out_high[:5]:
                details.append({"wavelength_nm": float(wl[idx]), "value": float(val[idx])})
            issues.append(_issue(
                f"{field}.values",
                "reflectance must lie within [0, 1]; "
                f"{out_low.size} value(s) below 0 and {out_high.size} above 1",
                "reflectance_out_of_range",
                allowed_range=[REFLECTANCE_MIN, REFLECTANCE_MAX],
                violations=details,
            ))
            ok = False
    else:
        if np.any(val < 0):
            neg = int(np.count_nonzero(val < 0))
            issues.append(_issue(
                f"{field}.values",
                f"spectral power must be non-negative; {neg} negative value(s)",
                "negative_power", negative_count=neg,
            ))
            ok = False

    if not ok:
        return None
    return Spectrum(wavelengths=wl, values=val, label=label)


def validate_illuminant_ref(name: str, field: str, grid_min: float,
                            grid_max: float, issues: list[dict]) -> bool:
    if name not in BUILTIN_ILLUMINANTS:
        issues.append(_issue(
            field,
            f"unknown built-in illuminant {name!r}; "
            f"choose one of {', '.join(BUILTIN_ILLUMINANTS)} or upload a custom SPD",
            "unknown_illuminant",
            allowed=list(BUILTIN_ILLUMINANTS),
        ))
        return False
    wl = illuminant_table()["wavelengths"]
    if wl.min() > grid_min + 1e-9 or wl.max() < grid_max - 1e-9:
        issues.append(_issue(
            field,
            f"illuminant {name} coverage [{wl.min():g}, {wl.max():g}] nm does not "
            f"cover the computation grid [{grid_min:g}, {grid_max:g}] nm",
            "insufficient_coverage",
        ))
        return False
    return True


def validate_grid(step: float, wl_min: float, wl_max: float) -> list[dict]:
    issues: list[dict] = []
    if not (5.0 <= step <= 20.0):
        issues.append(_issue("grid_step_nm",
                             "grid step must be between 5 and 20 nm",
                             "invalid_grid_step", value=step))
    if wl_min < WL_MIN or wl_max > WL_MAX or wl_min >= wl_max:
        issues.append(_issue("wavelength_range_nm",
                             f"range [{wl_min}, {wl_max}] must lie within "
                             f"[{WL_MIN:g}, {WL_MAX:g}] nm and be increasing",
                             "invalid_wavelength_range",
                             value=[wl_min, wl_max],
                             allowed_range=[WL_MIN, WL_MAX]))
    span = wl_max - wl_min
    if abs(span / step - round(span / step)) > 1e-6:
        issues.append(_issue("wavelength_range_nm",
                             "wavelength range is not an integer multiple of the step",
                             "range_step_mismatch"))
    return issues


def _is_finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fmt_wl(values: list[float], limit: int = 8) -> str:
    shown = ", ".join(f"{v:g}" for v in values[:limit])
    if len(values) > limit:
        shown += ", ..."
    return shown
