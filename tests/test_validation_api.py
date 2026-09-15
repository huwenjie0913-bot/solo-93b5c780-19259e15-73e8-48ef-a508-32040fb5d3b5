import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.validation import RequestValidationError, Spectrum
from app.engine import IlluminantInput, evaluate_pair
from app.colorimetry import make_grid

from tests.fixtures_data import METAMER, TARGET, WL_10NM, spectrum

client = TestClient(app)


def base(**over):
    payload = {
        "target": spectrum(TARGET),
        "sample": spectrum(TARGET),
        "illuminants": [{"name": "D65"}],
        "tolerance_de00": 2.0,
    }
    payload.update(over)
    return payload


def issues_for(payload, path="/api/v1/review"):
    r = client.post(path, json=payload)
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


def test_reflectance_out_of_range_reports_violations():
    bad = list(TARGET)
    bad[5] = 1.18
    bad[20] = -0.04
    issues = issues_for(base(sample=spectrum(bad)))
    issue = find(issues, "sample.values", "reflectance_out_of_range")
    assert issue["allowed_range"] == [0.0, 1.0]
    assert len(issue["violations"]) == 2
    wls = sorted(v["wavelength_nm"] for v in issue["violations"])
    assert wls == [430.0, 580.0]
    vals = {v["wavelength_nm"]: v["value"] for v in issue["violations"]}
    assert vals[430.0] == 1.18 and vals[580.0] == -0.04


def test_wavelength_gap_reports_exact_location():
    wl = [w for w in WL_10NM if w != 550]  # creates a 20 nm gap? -> still 20, allowed
    # create a 40 nm gap by removing two adjacent points
    wl = [w for w in WL_10NM if w not in (540, 550)]
    val = [v for w, v in zip(WL_10NM, TARGET) if w in wl]
    issues = issues_for(base(target=spectrum(val, wl)))
    issue = find(issues, "target.wavelengths", "wavelength_gap")
    assert issue["gap_nm"] == 30.0
    assert issue["gap_start_nm"] == 530.0
    assert issue["gap_end_nm"] == 560.0


def test_insufficient_coverage():
    wl = list(range(400, 701, 10))
    val = [0.3] * len(wl)
    issues = issues_for(base(sample=spectrum(val, wl)))
    issue = find(issues, "sample", "insufficient_coverage")
    assert issue["range_nm"] == [400.0, 700.0]
    assert issue["required_range_nm"] == [390.0, 770.0]


def test_length_mismatch():
    bad = {"wavelengths": WL_10NM, "values": TARGET[:-1]}
    issues = issues_for(base(target=bad))
    find(issues, "target", "length_mismatch")


def test_duplicate_wavelength():
    wl = list(WL_10NM)
    wl[5] = wl[4]
    issues = issues_for(base(sample=spectrum(TARGET, wl)))
    find(issues, "sample.wavelengths", "duplicate_wavelength")


def test_unknown_builtin_illuminant():
    issues = issues_for(base(illuminants=[{"name": "D90"}]))
    issue = find(issues, "illuminants[0].name", "unknown_illuminant")
    assert set(issue["allowed"]) == {"D65", "A", "F11"}


def test_illuminant_ambiguous_when_both_or_neither():
    issues = issues_for(base(illuminants=[{
        "name": "D65",
        "custom": {"wavelengths": WL_10NM, "values": [1.0] * len(WL_10NM)},
    }]))
    find(issues, "illuminants[0]", "illuminant_ambiguous")

    issues = issues_for(base(illuminants=[{}]))
    find(issues, "illuminants[0]", "illuminant_ambiguous")


def test_custom_zero_illuminant_rejected():
    issues = issues_for(base(illuminants=[{
        "custom": {"wavelengths": WL_10NM, "values": [0.0] * len(WL_10NM)},
    }]))
    find(issues, "illuminants[0].custom.values", "zero_illuminant_power")


def test_pydantic_tolerance_bounds():
    issues = issues_for(base(tolerance_de00=0))
    assert any("tolerance_de00" in i["field"] for i in issues)


def test_batch_sample_index_in_field_path():
    payload = {
        "target": spectrum(TARGET),
        "samples": [
            {"name": "ok", **spectrum(TARGET)},
            {"name": "bad", **spectrum([v + 1 for v in TARGET])},
        ],
        "illuminants": [{"name": "D65"}],
    }
    issues = issues_for(payload, "/api/v1/review/batch")
    issue = find(issues, "samples[1].values", "reflectance_out_of_range")
    assert issue is not None
    # the valid sample is not reported
    assert not any("samples[0]" in i["field"] for i in issues)


def test_custom_spd_runs_end_to_end():
    # flat E-like custom illuminant: neutral results, same XYZ shape as E
    spd = {"wavelengths": WL_10NM, "values": [100.0] * len(WL_10NM)}
    r = client.post("/api/v1/review", json=base(
        illuminants=[{"custom": spd, "label": "flat-E"}],
        sample=spectrum(METAMER),
    ))
    assert r.status_code == 200, r.text
    item = r.json()["results_by_illuminant"][0]
    assert item["kind"] == "custom"
    assert item["illuminant_label"] == "flat-E"
    assert item["white_xyz"][1] == pytest.approx(100.0, abs=1e-9)


def test_zero_normalisation_denominator_engine_level():
    """An SPD whose resampled power is zero on the grid: the defensive
    denominator check must name the illuminant and the physics reason."""
    grid = make_grid(10.0)
    zero_sp = Spectrum(
        wavelengths=np.asarray(WL_10NM, dtype=float),
        values=np.zeros(len(WL_10NM)),
        label="dead-lamp",
    )
    target_sp = Spectrum(
        wavelengths=np.asarray(WL_10NM, dtype=float),
        values=np.asarray(TARGET),
        label="target",
    )
    ill = IlluminantInput(key="custom:dead-lamp", display_name="dead-lamp",
                          kind="custom", spectrum=zero_sp)
    with pytest.raises(RequestValidationError) as exc:
        evaluate_pair(target_sp, target_sp, [ill],
                      step_nm=10.0, tolerance=2.0,
                      band_width_nm=40.0, top_bands=3)
    issue = exc.value.issues[0]
    assert issue["code"] == "zero_normalisation_denominator"
    assert "dead-lamp" in issue["reason"]
    assert "y_bar" in issue["reason"]


def test_multiple_issues_collected_at_once():
    bad_target = {"wavelengths": [400.0], "values": [0.5]}
    bad_sample = {"wavelengths": WL_10NM, "values": [2.0] * len(WL_10NM)}
    issues = issues_for(base(target=bad_target, sample=bad_sample,
                             illuminants=[{"name": "ZZ"}]))
    fields = {i["field"] for i in issues}
    assert any(f.startswith("target") for f in fields)
    assert any("sample.values" in f for f in fields)
    assert any("illuminants[0].name" in f for f in fields)
