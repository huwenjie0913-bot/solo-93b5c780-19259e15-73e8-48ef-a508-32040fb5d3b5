"""Tests for the Kubelka-Munk dye-strength endpoint
POST /api/v1/review/strength."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.strength import kubelka_munk_ks, ks_to_reflectance

from tests.fixtures_data import METAMER, TARGET, WL_10NM, spectrum

client = TestClient(app)

WL = np.asarray(WL_10NM, dtype=float)
KS_TARGET = kubelka_munk_ks(np.asarray(TARGET))


def scaled_sample(factor):
    """Sample whose K/S is exactly *factor* x the target K/S."""
    return ks_to_reflectance(factor * KS_TARGET).round(6).tolist()


def hue_shifted_sample():
    """Sample whose K/S shape differs from the target but integrates to the
    same visible-range strength (equal-area +/- bumps at 470/620 nm)."""
    bump = (0.15 * np.exp(-((WL - 470.0) / 12.0) ** 2)
            - 0.15 * np.exp(-((WL - 620.0) / 12.0) ** 2))
    return ks_to_reflectance(KS_TARGET + bump).round(6).tolist()


def strength_payload(**kw):
    payload = {
        "target": spectrum(TARGET),
        "sample": spectrum(scaled_sample(1.3)),
        "illuminants": [{"name": "D65"}],
    }
    payload.update(kw)
    return payload


def post(payload):
    return client.post("/api/v1/review/strength", json=payload)


def issues_for(payload):
    r = post(payload)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "validation_error"
    return body["issues"]


def find(issues, field_suffix, code=None):
    hits = [i for i in issues if i["field"].endswith(field_suffix)]
    if code:
        hits = [i for i in hits if i["code"] == code]
    assert hits, f"no issue for field ...{field_suffix} code={code}; got {issues}"
    return hits[0]


def test_ks_round_trip():
    r = np.linspace(0.01, 1.0, 50)
    ks = kubelka_munk_ks(r)
    assert np.all(ks >= 0.0)
    back = ks_to_reflectance(ks)
    assert np.allclose(back, r, atol=1e-12)
    # white stays white, and the inverse stays inside (0, 1]
    assert ks_to_reflectance(np.array([0.0]))[0] == 1.0
    assert np.all(ks_to_reflectance(np.array([0.0, 1.0, 100.0])) <= 1.0)
    assert np.all(ks_to_reflectance(np.array([0.0, 1.0, 100.0])) > 0.0)


def test_overstrength_sample_is_too_dark_and_corrected():
    r = post(strength_payload())
    assert r.status_code == 200, r.text
    body = r.json()

    # auto mode locates the target K/S peak (min reflectance of TARGET: 470 nm)
    assert body["primary_wavelength_nm"] == 470.0
    assert body["primary_wavelength_source"] == "auto"
    lo, hi = body["analysis_band_nm"]
    assert lo <= 470.0 <= hi

    s = body["strength"]
    assert s["ratio_integrated"] == pytest.approx(1.3, abs=1e-4)
    assert s["ratio_band"] == pytest.approx(1.3, abs=1e-4)
    assert s["ratio_at_wavelength"] == pytest.approx(1.3, abs=1e-4)
    assert s["scale_factor"] == pytest.approx(1.0 / 1.3, abs=1e-4)
    assert s["within_tolerance"] is False

    assert body["verdict"] == "too_dark"
    item = body["results_by_illuminant"][0]
    # correction removes essentially all of the colour difference
    assert item["delta_e00_before"] > 3.0
    assert item["delta_e00_after"] == pytest.approx(0.0, abs=1e-3)
    assert body["residual_de00_max"] == pytest.approx(0.0, abs=1e-3)
    # the corrected spectrum reproduces the target
    assert np.allclose(body["corrected_reflectance"], TARGET, atol=1e-5)
    assert all(0.0 < v <= 1.0 for v in body["corrected_reflectance"])


def test_understrength_sample_is_too_light():
    body = post(strength_payload(sample=spectrum(scaled_sample(0.8)))).json()
    assert body["strength"]["ratio_integrated"] == pytest.approx(0.8, abs=1e-4)
    assert body["strength"]["scale_factor"] == pytest.approx(1.25, abs=1e-4)
    assert body["verdict"] == "too_light"
    item = body["results_by_illuminant"][0]
    assert item["delta_e00_after"] < item["delta_e00_before"]
    assert np.allclose(body["corrected_reflectance"], TARGET, atol=1e-5)


def test_identical_pair_matches():
    body = post(strength_payload(sample=spectrum(TARGET))).json()
    assert body["verdict"] == "match"
    assert body["strength"]["ratio_integrated"] == pytest.approx(1.0, abs=1e-6)
    assert body["strength"]["scale_factor"] == pytest.approx(1.0, abs=1e-6)
    assert body["strength"]["within_tolerance"] is True
    for item in body["results_by_illuminant"]:
        assert item["delta_e00_before"] == 0.0
        assert item["delta_e00_after"] == 0.0


def test_hue_shift_with_matching_strength_is_recipe_mismatch():
    body = post(strength_payload(sample=spectrum(hue_shifted_sample()))).json()
    # depth is fine (integrated ratio ~ 1) but the K/S shape differs, so the
    # residual after correction stays above the 1.0 default tolerance
    assert body["strength"]["within_tolerance"] is True
    assert body["strength"]["ratio_integrated"] == pytest.approx(1.0, abs=0.02)
    assert body["residual_de00_max"] > 1.0
    assert body["verdict"] == "recipe_mismatch"
    assert body["residual_worst_illuminant"] == "D65"


def test_metamer_pair_reports_depth_problem_under_both_lights():
    body = post(strength_payload(
        sample=spectrum(METAMER),
        illuminants=[{"name": "D65"}, {"name": "A"}],
    )).json()
    by_ill = {x["illuminant"]: x for x in body["results_by_illuminant"]}
    # near-match under D65, far off under A before correction
    assert by_ill["D65"]["delta_e00_before"] < 0.25
    assert by_ill["A"]["delta_e00_before"] > 2.5
    # the K/S analysis attributes it to strength, not hue
    assert body["strength"]["ratio_integrated"] > 1.3
    assert body["verdict"] == "too_dark"


def test_requested_wavelength_scope():
    body = post(strength_payload(primary_wavelength_nm=550)).json()
    assert body["primary_wavelength_source"] == "requested_wavelength"
    assert body["primary_wavelength_nm"] == 550.0
    assert body["analysis_band_nm"] == [550.0, 550.0]
    s = body["strength"]
    assert s["ratio_at_wavelength"] == pytest.approx(1.3, abs=1e-4)
    assert s["ratio_band"] == pytest.approx(1.3, abs=1e-4)
    assert s["scale_factor"] == pytest.approx(1.0 / 1.3, abs=1e-4)


def test_requested_band_scope_reports_band_peak():
    body = post(strength_payload(analysis_band_nm=[500, 600])).json()
    assert body["primary_wavelength_source"] == "requested_band"
    assert body["analysis_band_nm"] == [500.0, 600.0]
    # target K/S peak inside [500, 600] sits at the 500 nm edge
    assert body["primary_wavelength_nm"] == 500.0
    assert body["strength"]["ratio_band"] == pytest.approx(1.3, abs=1e-4)
    assert body["strength"]["scale_factor"] == pytest.approx(1.0 / 1.3, abs=1e-4)


def test_band_is_clipped_to_common_support():
    wl = list(range(390, 771, 10))
    t = [v for w, v in zip(WL_10NM, TARGET) if w in wl]
    s = [v for w, v in zip(WL_10NM, scaled_sample(1.3)) if w in wl]
    body = post(strength_payload(
        target=spectrum(t, wl), sample=spectrum(s, wl),
        analysis_band_nm=[700, 780],
    )).json()
    # partial overlap: the band is clipped to the 390-770 grid, not an error
    assert body["analysis_band_nm"] == [700.0, 770.0]
    assert body["strength"]["ratio_band"] == pytest.approx(1.3, abs=1e-4)


def test_ambiguous_scope_rejected():
    issues = issues_for(strength_payload(
        primary_wavelength_nm=550, analysis_band_nm=[500, 600]))
    find(issues, "analysis_band_nm", "analysis_scope_ambiguous")


def test_band_not_increasing_rejected():
    issues = issues_for(strength_payload(analysis_band_nm=[600, 500]))
    find(issues, "analysis_band_nm", "band_not_increasing")


def test_band_without_common_support_rejected():
    wl = list(range(390, 771, 10))
    payload = strength_payload(
        target=spectrum([0.3] * len(wl), wl),
        sample=spectrum([0.4] * len(wl), wl),
        analysis_band_nm=[772, 778],
    )
    issues = issues_for(payload)
    issue = find(issues, "analysis_band_nm", "no_common_band")
    assert issue["grid_range_nm"] == [390.0, 770.0]


def test_near_zero_reflectance_rejected_with_locations():
    bad = list(TARGET)
    bad[10] = 0.0
    bad[20] = 0.0005
    issues = issues_for(strength_payload(sample=spectrum(bad)))
    issue = find(issues, "sample.values", "near_zero_reflectance")
    wls = sorted(v["wavelength_nm"] for v in issue["violations"])
    assert wls == [480.0, 580.0]


def test_white_target_has_no_absorption():
    issues = issues_for(strength_payload(target=spectrum([1.0] * len(WL_10NM))))
    find(issues, "target.values", "no_absorption")


def test_white_sample_scaling_undefined():
    issues = issues_for(strength_payload(sample=spectrum([1.0] * len(WL_10NM))))
    find(issues, "sample.values", "undefined_scaling")


def test_requested_wavelength_outside_support():
    wl = list(range(390, 771, 10))
    payload = strength_payload(
        target=spectrum([0.3] * len(wl), wl),
        sample=spectrum([0.4] * len(wl), wl),
        primary_wavelength_nm=775,
    )
    issues = issues_for(payload)
    find(issues, "primary_wavelength_nm", "wavelength_out_of_support")


def test_parameter_bounds_are_field_located():
    issues = issues_for(strength_payload(primary_wavelength_nm=300))
    find(issues, "primary_wavelength_nm")

    issues = issues_for(strength_payload(primary_wavelength_nm=900))
    find(issues, "primary_wavelength_nm")

    issues = issues_for(strength_payload(strength_tolerance=0))
    find(issues, "strength_tolerance")

    issues = issues_for(strength_payload(residual_tolerance_de00=0))
    find(issues, "residual_tolerance_de00")


def test_custom_illuminant_supported():
    spd = {"wavelengths": WL_10NM, "values": [100.0] * len(WL_10NM)}
    body = post(strength_payload(
        illuminants=[{"custom": spd, "label": "flat-E"}],
    )).json()
    item = body["results_by_illuminant"][0]
    assert item["kind"] == "custom"
    assert item["illuminant_label"] == "flat-E"
    assert item["delta_e00_after"] == pytest.approx(0.0, abs=1e-3)
