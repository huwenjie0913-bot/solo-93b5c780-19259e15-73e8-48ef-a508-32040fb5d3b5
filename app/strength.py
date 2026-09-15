"""Dye-strength analysis via the Kubelka-Munk K/S transform.

Splits a composite target/sample colour difference into

* a depth-of-shade component -- the K/S strength ratio at the main
  absorption wavelength, over the analysis band, and integrated across the
  visible range (the composite strength ratio); and
* a hue/recipe component -- the residual Delta E00 after the sample K/S has
  been rescaled by the non-negative least-squares factor that best matches
  the target strength over the analysis band.

The rescaled K/S is inverted back to a corrected reflectance spectrum, so
both the "before" and "after correction" states are reported as Lab and
Delta E00 under every requested illuminant.
"""
from __future__ import annotations

import numpy as np

from .colorimetry import (
    ciede2000,
    cmf_on_grid,
    illuminant_on_grid,
    interp_linear,
    make_grid,
    trapz,
    tristimulus,
    xyz_to_lab,
)
from .engine import IlluminantInput, _common_support_range, align_reflectance
from .validation import RequestValidationError, Spectrum

# K/S = (1-R)^2/(2R) diverges as R -> 0; below this floor the transform is
# numerically meaningless for strength analysis.
MIN_REFLECTANCE_FOR_KS = 1e-3

# In auto mode the main absorption region is the contiguous set of
# wavelengths around the K/S peak staying above this fraction of the peak.
AUTO_REGION_FRACTION = 0.5

# K/S magnitudes at or below this are treated as "no absorption".
KS_EPS = 1e-12


def kubelka_munk_ks(reflectance: np.ndarray) -> np.ndarray:
    """K/S = (1 - R)^2 / (2R); requires R > 0."""
    r = np.asarray(reflectance, dtype=float)
    return (1.0 - r) ** 2 / (2.0 * r)


def ks_to_reflectance(ks: np.ndarray) -> np.ndarray:
    """Invert K/S: R = 1 + K/S - sqrt((K/S)^2 + 2 K/S), in (0, 1] for K/S >= 0."""
    ks = np.asarray(ks, dtype=float)
    return 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)


def _auto_region(ks_target: np.ndarray) -> tuple[int, int]:
    """Index range [start, end) of the target's main absorption region: the
    contiguous wavelengths around the K/S peak whose K/S stays above
    AUTO_REGION_FRACTION of the peak."""
    peak = int(np.argmax(ks_target))
    threshold = AUTO_REGION_FRACTION * float(ks_target[peak])
    start = peak
    while start > 0 and ks_target[start - 1] >= threshold:
        start -= 1
    end = peak + 1
    while end < ks_target.size and ks_target[end] >= threshold:
        end += 1
    return start, end


def _near_zero_issues(grid: np.ndarray, r_t: np.ndarray, r_s: np.ndarray
                      ) -> list[dict]:
    issues: list[dict] = []
    for field, r in (("target.values", r_t), ("sample.values", r_s)):
        low = np.nonzero(r < MIN_REFLECTANCE_FOR_KS)[0]
        if low.size:
            issues.append({
                "field": field,
                "code": "near_zero_reflectance",
                "reason": (
                    f"{low.size} reflectance value(s) below "
                    f"{MIN_REFLECTANCE_FOR_KS:g}; K/S = (1-R)^2/(2R) is "
                    "undefined at near-zero reflectance"
                ),
                "floor": MIN_REFLECTANCE_FOR_KS,
                "violations": [
                    {"wavelength_nm": float(grid[i]), "value": float(r[i])}
                    for i in low[:10]
                ],
            })
    return issues


def evaluate_strength(target: Spectrum, sample: Spectrum,
                      illuminants: list[IlluminantInput],
                      step_nm: float,
                      primary_wavelength_nm: float | None,
                      analysis_band_nm: tuple[float, float] | None,
                      strength_tolerance: float,
                      residual_tolerance_de00: float) -> dict:
    """Kubelka-Munk strength split of a target/sample pair.

    Exactly one of *primary_wavelength_nm* / *analysis_band_nm* may select
    the analysis scope; with neither, the target's main absorption region is
    located automatically.
    """
    wl_lo, wl_hi = _common_support_range(target, sample, illuminants, step_nm)
    grid = make_grid(step_nm, wl_lo, wl_hi)
    r_t = align_reflectance(target, grid)
    r_s = align_reflectance(sample, grid)

    issues = _near_zero_issues(grid, r_t, r_s)
    if issues:
        raise RequestValidationError(issues)

    ks_t = kubelka_munk_ks(r_t)
    ks_s = kubelka_munk_ks(r_s)

    # Composite (visible-range integrated) strength ratio.
    int_t = float(trapz(ks_t, grid))
    if int_t <= KS_EPS:
        raise RequestValidationError([{
            "field": "target.values",
            "code": "no_absorption",
            "reason": (
                "target K/S integrates to zero over the visible range "
                "(a white/perfect reflector absorbs nothing); the strength "
                "ratio is undefined"
            ),
        }])
    int_s = float(trapz(ks_s, grid))
    ratio_integrated = int_s / int_t

    # --- resolve the analysis scope ---
    if analysis_band_nm is not None:
        lo, hi = analysis_band_nm
        b_lo = max(lo, float(grid[0]))
        b_hi = min(hi, float(grid[-1]))
        mask = (grid >= b_lo - 1e-9) & (grid <= b_hi + 1e-9)
        if not np.any(mask):
            raise RequestValidationError([{
                "field": "analysis_band_nm",
                "code": "no_common_band",
                "reason": (
                    f"analysis band [{lo:g}, {hi:g}] nm shares no grid points "
                    f"with the common support [{grid[0]:g}, {grid[-1]:g}] nm"
                ),
                "band_nm": [lo, hi],
                "grid_range_nm": [float(grid[0]), float(grid[-1])],
            }])
        idx = np.nonzero(mask)[0]
        grid_band, ks_t_band, ks_s_band = grid[idx], ks_t[idx], ks_s[idx]
        primary_wl = float(grid_band[int(np.argmax(ks_t_band))])
        source = "requested_band"
        scope_field = "analysis_band_nm"
    elif primary_wavelength_nm is not None:
        lam = float(primary_wavelength_nm)
        if lam < grid[0] - 1e-9 or lam > grid[-1] + 1e-9:
            raise RequestValidationError([{
                "field": "primary_wavelength_nm",
                "code": "wavelength_out_of_support",
                "reason": (
                    f"primary wavelength {lam:g} nm lies outside the common "
                    f"support [{grid[0]:g}, {grid[-1]:g}] nm of the submitted "
                    "spectra"
                ),
                "grid_range_nm": [float(grid[0]), float(grid[-1])],
            }])
        grid_band = np.array([lam])
        ks_t_band = np.array([float(np.interp(lam, grid, ks_t))])
        ks_s_band = np.array([float(np.interp(lam, grid, ks_s))])
        primary_wl = lam
        source = "requested_wavelength"
        scope_field = "primary_wavelength_nm"
    else:
        start, end = _auto_region(ks_t)
        grid_band = grid[start:end]
        ks_t_band, ks_s_band = ks_t[start:end], ks_s[start:end]
        primary_wl = float(grid[int(np.argmax(ks_t))])
        source = "auto"
        scope_field = "target.values"

    # Strength ratio at the reported main absorption wavelength.
    ks_t_at = float(np.interp(primary_wl, grid, ks_t))
    ks_s_at = float(np.interp(primary_wl, grid, ks_s))
    if ks_t_at <= KS_EPS:
        raise RequestValidationError([{
            "field": scope_field,
            "code": "no_absorption",
            "reason": (
                f"target K/S is zero at {primary_wl:g} nm (the target does "
                "not absorb there); the strength ratio is undefined"
            ),
        }])
    ratio_at = ks_s_at / ks_t_at

    # Strength ratio integrated over the analysis band.
    if grid_band.size >= 2:
        band_t = float(trapz(ks_t_band, grid_band))
        band_s = float(trapz(ks_s_band, grid_band))
    else:
        band_t, band_s = float(ks_t_band[0]), float(ks_s_band[0])
    if band_t <= KS_EPS:
        raise RequestValidationError([{
            "field": scope_field,
            "code": "no_absorption",
            "reason": (
                "target K/S integrates to zero over the analysis band "
                f"[{grid_band[0]:g}, {grid_band[-1]:g}] nm; the strength "
                "ratio is undefined"
            ),
        }])
    ratio_band = band_s / band_t

    # Non-negative least-squares scale: min_a ||a * ks_s - ks_t|| over the
    # band, a >= 0. K/S is non-negative, so the unconstrained solution is
    # already non-negative; the clip only guards the degenerate case.
    denom = float(np.dot(ks_s_band, ks_s_band))
    if denom <= KS_EPS:
        raise RequestValidationError([{
            "field": "sample.values",
            "code": "undefined_scaling",
            "reason": (
                "sample K/S is zero across the analysis band (the sample "
                "absorbs nothing there); no scaling factor can match the "
                "target strength"
            ),
        }])
    scale = max(0.0, float(np.dot(ks_s_band, ks_t_band)) / denom)

    # Corrected spectrum: rescale the sample K/S, invert to reflectance.
    r_corr = ks_to_reflectance(scale * ks_s)

    x_bar, y_bar, z_bar = cmf_on_grid(grid)
    per_ill: list[dict] = []
    for ill in illuminants:
        if ill.kind == "builtin":
            power = illuminant_on_grid(ill.ref_name, grid)
        else:
            power = interp_linear(ill.spectrum.wavelengths,
                                  ill.spectrum.values, grid)
            if trapz(power * y_bar, grid) <= 0:
                raise RequestValidationError([{
                    "field": f"illuminants:{ill.key}",
                    "code": "zero_normalisation_denominator",
                    "reason": (
                        "normalisation denominator integral "
                        f"S(lambda)y_bar(lambda) is zero for illuminant "
                        f"{ill.display_name!r}; the SPD must emit within the "
                        "visible (especially 450-650 nm) range"
                    ),
                }])
        white = tristimulus(np.ones_like(grid), power, grid,
                            x_bar, y_bar, z_bar)
        white_xyz = tuple(float(v) for v in white)
        lab_t = xyz_to_lab(
            tristimulus(r_t, power, grid, x_bar, y_bar, z_bar), white_xyz)
        lab_s = xyz_to_lab(
            tristimulus(r_s, power, grid, x_bar, y_bar, z_bar), white_xyz)
        lab_c = xyz_to_lab(
            tristimulus(r_corr, power, grid, x_bar, y_bar, z_bar), white_xyz)
        per_ill.append({
            "illuminant": ill.key,
            "illuminant_label": ill.display_name,
            "kind": ill.kind,
            "lab_target": {k: round(v, 4) for k, v in lab_t.as_dict().items()},
            "lab_sample": {k: round(v, 4) for k, v in lab_s.as_dict().items()},
            "lab_corrected": {k: round(v, 4) for k, v in lab_c.as_dict().items()},
            "delta_e00_before": round(ciede2000(lab_t, lab_s), 4),
            "delta_e00_after": round(ciede2000(lab_t, lab_c), 4),
        })

    residual_max = max(r["delta_e00_after"] for r in per_ill)
    worst_ill = max(per_ill, key=lambda r: r["delta_e00_after"])

    if ratio_integrated < 1.0 - strength_tolerance:
        verdict = "too_light"
        detail = (
            f"composite K/S strength ratio {ratio_integrated:.4f} is below "
            f"1 - tolerance ({1.0 - strength_tolerance:.4f}): the sample is "
            "under-strength (dyed too light); increase the dye concentration"
        )
    elif ratio_integrated > 1.0 + strength_tolerance:
        verdict = "too_dark"
        detail = (
            f"composite K/S strength ratio {ratio_integrated:.4f} is above "
            f"1 + tolerance ({1.0 + strength_tolerance:.4f}): the sample is "
            "over-strength (dyed too dark); reduce the dye concentration"
        )
    elif residual_max > residual_tolerance_de00 + 1e-9:
        verdict = "recipe_mismatch"
        detail = (
            f"strength is within tolerance but the residual Delta E00 after "
            f"K/S correction reaches {residual_max:.4f} under "
            f"{worst_ill['illuminant']} (tolerance "
            f"{residual_tolerance_de00:g}): the hue itself differs -- adjust "
            "the dye recipe, not the concentration"
        )
    else:
        verdict = "match"
        detail = (
            "composite strength and residual Delta E00 are both within "
            "tolerance"
        )

    return {
        "grid_nm": list(grid),
        "primary_wavelength_nm": round(primary_wl, 4),
        "primary_wavelength_source": source,
        "analysis_band_nm": [float(grid_band[0]), float(grid_band[-1])],
        "strength": {
            "ratio_at_wavelength": round(ratio_at, 6),
            "ratio_band": round(ratio_band, 6),
            "ratio_integrated": round(ratio_integrated, 6),
            "scale_factor": round(scale, 6),
            "tolerance": strength_tolerance,
            "within_tolerance": bool(
                abs(ratio_integrated - 1.0) <= strength_tolerance + 1e-12),
        },
        "target_reflectance": [round(float(v), 6) for v in r_t],
        "sample_reflectance": [round(float(v), 6) for v in r_s],
        "corrected_reflectance": [round(float(v), 6) for v in r_corr],
        "results_by_illuminant": per_ill,
        "residual_de00_max": round(residual_max, 4),
        "residual_tolerance_de00": residual_tolerance_de00,
        "residual_worst_illuminant": worst_ill["illuminant"],
        "verdict": verdict,
        "verdict_detail": detail,
    }
