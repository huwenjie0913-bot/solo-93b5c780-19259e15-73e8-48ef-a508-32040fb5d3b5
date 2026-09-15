"""Tests for the repeat-scan bootstrap endpoint
POST /api/v1/review/uncertainty."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures_data import (
    DEFAULT_ILLUMINANTS,
    METAMER,
    TARGET,
    WL_10NM,
    spectrum,
)

client = TestClient(app)

WL = np.asarray(WL_10NM, dtype=float)


def repeat_payload(base, n, *, noise=0.003, seed=0, sparse_first=False):
    """n jittered copies of *base*; the first may use a 20 nm grid to prove
    that scans on different wavelength grids are accepted."""
    rng = np.random.default_rng(seed)
    scans = []
    for i in range(n):
        wl = WL[::2] if (sparse_first and i == 0) else WL
        values = np.clip(
            np.interp(wl, WL, np.asarray(base)) + rng.normal(0, noise, wl.size),
            0.0, 1.0)
        scans.append(spectrum(values.tolist(), wl.tolist()))
    return scans


def uncertainty_payload(target=TARGET, sample=METAMER, **kw):
    payload = {
        "target": {"scans": repeat_payload(target, kw.pop("n_target", 6),
                                           seed=1, **kw.pop("target_kw", {}))},
        "sample": {"scans": repeat_payload(sample, kw.pop("n_sample", 6),
                                           seed=2, **kw.pop("sample_kw", {}))},
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": kw.pop("tolerance_de00", 2.0),
        "seed": kw.pop("seed", 1234),
        "draws": kw.pop("draws", 3000),
    }
    payload.update(kw)
    return payload


def post(payload):
    return client.post("/api/v1/review/uncertainty", json=payload)


def test_basic_response_shape_and_ordering():
    r = post(uncertainty_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["illuminant"] for x in body["results_by_illuminant"]] == \
        ["D65", "A", "F11"]
    for item in body["results_by_illuminant"]:
        assert 0.0 <= item["q025_de00"] <= item["median_de00"] <= item["q975_de00"]
        assert item["ci95_width_de00"] == pytest.approx(
            item["q975_de00"] - item["q025_de00"], abs=1.1e-4)
        assert 0.0 <= item["probability_over_tolerance"] <= 1.0
        assert item["classification"] in {
            "stable_pass", "critical", "stable_fail"}
        assert item["effective_draws"] >= 1
        assert item["tolerance_de00"] == 2.0
        assert {"illuminant", "illuminant_label", "kind"} <= set(item)
    rep = body["replicates"]
    assert rep == {
        "target_scans": 6, "sample_scans": 6, "possible_pairs": 36,
    }
    # effective_draws counts the finite Delta E00 samples that actually fed
    # the quantiles/probability for each illuminant, not distinct pairings
    for item in body["results_by_illuminant"]:
        assert item["effective_draws"] == body["draws"]
    assert body["summary"]["effective_draws"] == body["draws"]
    assert body["seed"] == 1234 and body["draws"] == 3000
    assert body["critical_probability_bound"] == 0.05


def test_stable_pass_and_stable_fail_classes():
    # Identical specimens with tiny measurement noise: never over tolerance.
    body = post(uncertainty_payload(
        target=TARGET, sample=TARGET,
        target_kw={"noise": 0.0005}, sample_kw={"noise": 0.0005},
        tolerance_de00=2.0)).json()
    for item in body["results_by_illuminant"]:
        assert item["classification"] == "stable_pass"
        assert item["probability_over_tolerance"] == 0.0
        assert item["q975_de00"] < 2.0
    assert body["summary"]["classification_any"] == "stable_pass"
    assert body["summary"]["probability_over_tolerance_any_illuminant"] == 0.0

    # Strongly separated pair under a tight tolerance: always over.
    bright = [min(1.0, v + 0.2) for v in TARGET]
    body = post(uncertainty_payload(
        target=TARGET, sample=bright,
        illuminants=[{"name": "D65"}], tolerance_de00=0.5)).json()
    item = body["results_by_illuminant"][0]
    assert item["classification"] == "stable_fail"
    assert item["probability_over_tolerance"] == 1.0
    assert item["q025_de00"] > 0.5
    assert body["summary"]["classification_any"] == "stable_fail"
    assert body["summary"]["probability_over_tolerance_any_illuminant"] == 1.0


def test_critical_pair_straddles_tolerance():
    # The METAMER pair under F11 sits near dE ~ 1; inflate the noise so the
    # bootstrap distribution straddles the 1.2 threshold on at least one light.
    body = post(uncertainty_payload(
        target_kw={"noise": 0.006}, sample_kw={"noise": 0.012},
        illuminants=[{"name": "F11"}],
        tolerance_de00=1.0, draws=8000)).json()
    item = body["results_by_illuminant"][0]
    assert item["q025_de00"] < 1.0 < item["q975_de00"]
    assert 0.0 < item["probability_over_tolerance"] < 1.0
    assert item["classification"] == "critical"
    assert body["summary"]["classification_any"] == "critical"


def test_seed_makes_results_reproducible():
    p1 = uncertainty_payload(seed=99, draws=2000)
    p2 = uncertainty_payload(seed=99, draws=2000)
    b1 = post(p1).json()
    b2 = post(p2).json()
    assert b1["results_by_illuminant"] == b2["results_by_illuminant"]
    assert b1["summary"] == b2["summary"]

    # A different seed may shift Monte Carlo quantiles slightly; with a lot of
    # noise and few scans the difference should show up somewhere.
    p3 = uncertainty_payload(seed=100, draws=2000,
                             target_kw={"noise": 0.01},
                             sample_kw={"noise": 0.01})
    b3 = post(p3).json()
    p3["seed"] = 777
    b4 = post(p3).json()
    vals_a = [x["q025_de00"] for x in b3["results_by_illuminant"]]
    vals_b = [x["q025_de00"] for x in b4["results_by_illuminant"]]
    assert vals_a != vals_b


def test_draws_control_resolution():
    base = uncertainty_payload(tolerance_de00=2.0, draws=1000)
    body = post(base).json()
    # probability granularity is 1/draws
    for item in body["results_by_illuminant"]:
        p = item["probability_over_tolerance"]
        assert abs(p * 1000 - round(p * 1000)) < 1e-9
    assert body["draws"] == 1000


def test_identical_repeats_give_zero_distribution():
    # Exactly identical repeat scans: every bootstrap draw is the same pair,
    # so the Delta E00 distribution collapses to zero for identical curves.
    # effective_draws must still equal the requested number of draws, because
    # every draw produced a finite Delta E00 that fed the statistics.
    scans = repeat_payload(TARGET, 4, noise=0.0)
    body = post({
        "target": {"scans": scans},
        "sample": {"scans": repeat_payload(TARGET, 3, noise=0.0, seed=99)},
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": 2.0,
        "seed": 5, "draws": 500,
    }).json()
    for item in body["results_by_illuminant"]:
        assert item["median_de00"] == 0.0
        assert item["q025_de00"] == 0.0 and item["q975_de00"] == 0.0
        assert item["ci95_width_de00"] == 0.0
        assert item["effective_draws"] == 500
    assert body["replicates"] == {
        "target_scans": 4, "sample_scans": 3, "possible_pairs": 12}
    assert body["summary"]["effective_draws"] == 500


def test_scans_with_different_wavelength_grids_are_aligned():
    # half of the target scans live on a 20 nm grid, half on 10 nm
    target_scans = repeat_payload(TARGET, 4, sparse_first=True, seed=3)
    target_scans[2] = spectrum(
        np.interp(WL[::2], WL, np.asarray(TARGET)).tolist(), WL[::2].tolist())
    body = post({
        "target": {"scans": target_scans},
        "sample": {"scans": repeat_payload(METAMER, 3, seed=4)},
        "illuminants": [{"name": "D65"}],
        "tolerance_de00": 2.0,
        "seed": 1, "draws": 1000,
    }).json()
    assert body["grid_nm"][0] == 380.0 and body["grid_nm"][-1] == 780.0
    assert len(body["grid_nm"]) == 41
    # medians stay close to the deterministic single-review dE00 (< 0.25 D65)
    assert body["results_by_illuminant"][0]["median_de00"] < 0.6


def test_mixed_grids_with_shrunk_common_range():
    # one scan covers only 390-770 (the minimum accepted reflectance
    # coverage): the integration grid snaps inward to that intersection,
    # exactly like the single-review support snapping.
    wl_short = [w for w in WL_10NM if 390 <= w <= 770]
    scans = repeat_payload(TARGET, 4, seed=5)
    scans[0] = spectrum(
        [v for w, v in zip(WL_10NM, TARGET) if w in wl_short], wl_short)
    body = post({
        "target": {"scans": scans},
        "sample": {"scans": repeat_payload(METAMER, 3, seed=6)},
        "illuminants": [{"name": "D65"}],
        "tolerance_de00": 2.0,
        "seed": 1, "draws": 1000,
    }).json()
    assert body["grid_nm"][0] == 390.0 and body["grid_nm"][-1] == 770.0


def test_any_illuminant_probability_and_least_stable_light():
    body = post(uncertainty_payload(
        target_kw={"noise": 0.008}, sample_kw={"noise": 0.012},
        draws=4000)).json()
    per = body["results_by_illuminant"]
    individual = [x["probability_over_tolerance"] for x in per]
    p_any = body["summary"]["probability_over_tolerance_any_illuminant"]
    # union bound: >= max individual, <= sum of individuals
    assert p_any >= max(individual) - 1e-12
    assert p_any <= min(1.0, sum(individual)) + 1e-12
    # A is over tolerance for essentially every bootstrap draw of this
    # metameric pair (deterministic dE00 ~ 3.2, threshold 2.0)
    by_ill = {x["illuminant"]: x for x in per}
    assert by_ill["A"]["probability_over_tolerance"] > 0.9
    assert p_any >= by_ill["A"]["probability_over_tolerance"] - 1e-12

    widths = {x["illuminant"]: x["ci95_width_de00"] for x in per}
    least = body["summary"]["least_stable_illuminant"]
    assert widths[least] == max(widths.values())
    assert body["summary"]["least_stable_ci95_width_de00"] == widths[least]
    assert body["summary"]["least_stable_illuminant_label"] == \
        next(x["illuminant_label"] for x in per if x["illuminant"] == least)


def test_custom_illuminant_and_adaptive_white():
    spd = {"wavelengths": WL_10NM, "values": [100.0] * len(WL_10NM)}
    body = post(uncertainty_payload(
        illuminants=[{"custom": spd, "label": "flat-E"}])).json()
    item = body["results_by_illuminant"][0]
    assert item["kind"] == "custom"
    assert item["illuminant"] == "custom:0"
    assert item["illuminant_label"] == "flat-E"


def test_configurable_probability_bound_changes_classification():
    # Pair straddling the threshold with some moderate over probability.
    p = uncertainty_payload(
        target_kw={"noise": 0.01}, sample_kw={"noise": 0.02},
        illuminants=[{"name": "F11"}],
        tolerance_de00=1.0, draws=8000,
        critical_probability_bound=0.2)
    body = post(p).json()
    item = body["results_by_illuminant"][0]
    p_over = item["probability_over_tolerance"]
    if 0.2 < p_over < 0.8:
        assert item["classification"] == "critical"
    elif p_over <= 0.2:
        assert item["classification"] == "stable_pass"
    else:
        assert item["classification"] == "stable_fail"


# ----------------------------- validation --------------------------------

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
    assert hits, f"no issue for ...{field_suffix} code={code}; got {issues}"
    return hits[0]


def test_requires_at_least_two_scans():
    payload = uncertainty_payload()
    payload["target"]["scans"] = payload["target"]["scans"][:1]
    issues = issues_for(payload)
    assert any("target.scans" in i["field"] for i in issues)


def test_more_than_fifty_scans_rejected():
    payload = uncertainty_payload()
    payload["target"]["scans"] = repeat_payload(TARGET, 51)
    issues = issues_for(payload)
    assert any("target.scans" in i["field"] for i in issues)


def test_bad_scan_uses_indexed_field_path():
    bad = list(TARGET)
    bad[10] = 1.5
    payload = uncertainty_payload()
    payload["sample"]["scans"][2] = spectrum(bad)
    issues = issues_for(payload)
    issue = find(issues, "sample.scans[2].values",
                 "reflectance_out_of_range")
    assert issue["allowed_range"] == [0.0, 1.0]
    # valid sibling scans are not reported as errors
    assert not any("sample.scans[0]" in i["field"] for i in issues)


def test_scan_missing_visible_coverage_is_indexed():
    # reflectance scans must individually cover 390-770 nm; a short scan is
    # rejected with its position inside target.scans[...]
    wl_short = list(range(390, 561, 10))
    vals = [v for w, v in zip(WL_10NM, TARGET) if w in wl_short]
    payload = uncertainty_payload()
    payload["target"]["scans"][0] = spectrum(vals, wl_short)
    issues = issues_for(payload)
    issue = find(issues, "target.scans[0]", "insufficient_coverage")
    assert issue["required_range_nm"] == [390.0, 770.0]


def test_invalid_draws_and_seed_and_bound():
    for field, bad in [("draws", 10), ("draws", 10 ** 9),
                       ("seed", -1), ("critical_probability_bound", 0.0),
                       ("critical_probability_bound", 0.5)]:
        payload = uncertainty_payload(**{field: bad})
        issues = issues_for(payload)
        assert any(field in i["field"] for i in issues), (field, issues)


def test_unknown_illuminant_still_structured():
    payload = uncertainty_payload(illuminants=[{"name": "D90"}])
    issues = issues_for(payload)
    find(issues, "illuminants[0].name", "unknown_illuminant")


def test_dark_pair_uncertainty_uses_correct_linear_branch():
    """Regression: the uncertainty entry shares xyz_to_lab_array, so dark
    non-black scans must produce the correct low-L* distribution (the broken
    kappa*t + 16/116 branch inflated L* by ~13)."""
    import math
    kappa = 24389.0 / 27.0
    r_t, r_s = 0.004, 0.006

    def flat_scans(r, seed):
        rng = np.random.default_rng(seed)
        return [spectrum(np.clip(
            r + rng.normal(0, 2e-5, WL.size), 0, 1).tolist())
            for _ in range(4)]

    body = post({
        "target": {"scans": flat_scans(r_t, 1)},
        "sample": {"scans": flat_scans(r_s, 2)},
        "illuminants": [{"name": "D65"}],
        "tolerance_de00": 50.0,
        "seed": 0, "draws": 2000,
    }).json()
    item = body["results_by_illuminant"][0]
    l1, l2 = kappa * r_t, kappa * r_s
    l_bar = 0.5 * (l1 + l2)
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / math.sqrt(
        20.0 + (l_bar - 50.0) ** 2)
    expected = abs(l2 - l1) / s_l
    # tight agreement with the closed-form oracle; the broken
    # kappa*t + 16/116 branch shifted this pair by ~0.18
    assert item["median_de00"] == pytest.approx(expected, abs=5e-3)
    assert item["effective_draws"] == 2000


def test_effective_draws_counts_finite_samples_per_illuminant():
    # With valid inputs every draw is finite: effective_draws equals draws for
    # every illuminant, regardless of how few distinct scans/pairings exist.
    body = post(uncertainty_payload(n_target=2, n_sample=2, draws=777)).json()
    for item in body["results_by_illuminant"]:
        assert item["effective_draws"] == 777
    assert body["summary"]["effective_draws"] == 777
    # pair combinatorics are reported separately, not as effective_draws
    assert body["replicates"]["possible_pairs"] == 4
    assert "unique_pairs_sampled" not in body["replicates"]


def test_single_and_batch_endpoints_unchanged():    # regression: the original single-review shape is untouched
    r = client.post("/api/v1/review", json={
        "target": spectrum(TARGET),
        "sample": spectrum(METAMER),
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": 2.0,
    })
    assert r.status_code == 200
    item = r.json()["results_by_illuminant"][0]
    assert "median_de00" not in item and "probability_over_tolerance" not in item
    assert "delta_e00" in item
