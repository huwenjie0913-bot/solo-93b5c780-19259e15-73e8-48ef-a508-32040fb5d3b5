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


def review_payload(target=TARGET, sample=METAMER, **kw):
    payload = {
        "target": spectrum(target),
        "sample": spectrum(sample),
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": kw.pop("tolerance_de00", 2.0),
    }
    payload.update(kw)
    return payload


def test_health_and_catalogue():
    assert client.get("/health").json() == {"status": "ok"}
    cat = client.get("/illuminants").json()
    names = [i["name"] for i in cat["built_in"]]
    assert names == ["D65", "A", "F11"]
    for item in cat["built_in"]:
        assert item["wavelength_range_nm"] == [380.0, 780.0]


def test_single_review_metameric_pair():
    r = client.post("/api/v1/review", json=review_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    by_ill = {x["illuminant"]: x for x in body["results_by_illuminant"]}

    # D65 near-exact match, A clearly off, F11 within tolerance.
    assert by_ill["D65"]["delta_e00"] < 0.25
    assert by_ill["A"]["delta_e00"] > 2.5
    assert by_ill["F11"]["delta_e00"] < 2.0

    assert by_ill["D65"]["pass"] is True
    assert by_ill["A"]["pass"] is False
    assert by_ill["F11"]["pass"] is True

    mr = body["metamerism_risk"]
    assert mr["flag"] is True
    assert set(mr["passes_under"]) == {"D65", "F11"}
    assert mr["fails_under"] == ["A"]
    assert mr["worst_illuminant"] == "A"
    assert mr["best_illuminant"] == "D65"
    assert mr["spread_de00"] == pytest.approx(
        by_ill["A"]["delta_e00"] - by_ill["D65"]["delta_e00"], abs=1e-3)
    assert body["pass_all"] is False

    # each per-illuminant result reports dominant bands, sorted desc
    for item in body["results_by_illuminant"]:
        bands = item["top_bands"]
        assert len(bands) == 3
        mags = [b["magnitude"] for b in bands]
        assert mags == sorted(mags, reverse=True)
        rng = bands[0]["wavelength_range_nm"]
        assert rng[0] < rng[1]
        assert "delta_xyz" in bands[0]


def test_identical_curves_pass_everywhere():
    body = client.post("/api/v1/review",
                       json=review_payload(sample=TARGET)).json()
    assert body["pass_all"] is True
    assert body["max_de00"] == 0.0
    assert body["metamerism_risk"]["flag"] is False
    for item in body["results_by_illuminant"]:
        assert item["delta_e00"] == 0.0
        assert item["pass"] is True
        # a perfect white has neutral Lab under its own illuminant white point
        assert abs(item["lab_target"]["L"] - item["lab_sample"]["L"]) < 1e-6


def test_grid_interpolation_alignment():
    # sparse target on 20 nm grid, dense sample on 10 nm grid, unsorted sample
    body = client.post("/api/v1/review", json=review_payload(
        target=TARGET, sample=[v + 0.01 for v in METAMER],
        grid_step_nm=10)).json()
    assert len(body["grid_nm"]) == 41
    assert len(body["target_reflectance"]) == 41
    assert len(body["sample_reflectance"]) == 41
    # endpoints are the common 380..780 grid
    assert body["grid_nm"][0] == 380.0 and body["grid_nm"][-1] == 780.0


def test_grid_shrinks_to_actual_support_no_500():
    # reflectance covering exactly the accepted 390-770 window but short of
    # the 380-780 reference tables: the integration grid must snap inward
    # instead of extrapolating (which would otherwise raise a 500).
    wl = list(range(390, 771, 10))
    t = [v for w, v in zip(WL_10NM, TARGET) if w in wl]
    s = [v for w, v in zip(WL_10NM, METAMER) if w in wl]
    payload = {
        "target": spectrum(t, wl),
        "sample": spectrum(s, wl),
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": 2.0,
    }
    r = client.post("/api/v1/review", json=payload)
    assert r.status_code == 200, r.text
    grid = r.json()["grid_nm"]
    assert grid[0] == 390.0 and grid[-1] == 770.0
    # results on the reduced grid stay close to the full-range computation
    full = client.post("/api/v1/review", json=review_payload()).json()
    for red, ful in zip(r.json()["results_by_illuminant"],
                        full["results_by_illuminant"]):
        assert abs(red["delta_e00"] - ful["delta_e00"]) < 0.25


def test_custom_spd_narrower_support_drives_grid():
    wl_led = list(range(410, 721, 10))
    import math
    led = [
        max(0.0, 0.05
            + math.exp(-((w - 455) ** 2) / (2 * 12 ** 2))
            + 1.1 * math.exp(-((w - 625) ** 2) / (2 * 13 ** 2)))
        for w in wl_led
    ]
    payload = review_payload(illuminants=[
        {"name": "D65"},
        {"custom": spectrum(led, wl_led), "label": "narrow-LED"},
    ])
    r = client.post("/api/v1/review", json=payload)
    assert r.status_code == 200, r.text
    grid = r.json()["grid_nm"]
    assert grid[0] == 410.0 and grid[-1] == 720.0


def test_batch_sort_filter_and_metamer_flag():
    samples = [
        {"name": "metamer", **spectrum(METAMER)},
        {"name": "identical", **spectrum(TARGET)},
        {"name": "bright", **spectrum([min(1.0, v + 0.25) for v in TARGET])},
    ]
    payload = {
        "target": spectrum(TARGET),
        "samples": samples,
        "illuminants": DEFAULT_ILLUMINANTS,
        "tolerance_de00": 2.0,
        "sort_by": "max_de00",
    }
    r = client.post("/api/v1/review/batch", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["submitted"] == 3 and body["returned"] == 3

    names = [c["name"] for c in body["candidates"]]
    maxes = [c["max_de00"] for c in body["candidates"]]
    # ascending worst-case colour difference
    assert maxes == sorted(maxes)
    assert names[0] == "identical"

    metamer_candidate = next(c for c in body["candidates"] if c["name"] == "metamer")
    assert metamer_candidate["metamerism_risk"]["flag"] is True
    assert metamer_candidate["pass_all"] is False

    # filter: only metameric
    payload["only_metameric"] = True
    body = client.post("/api/v1/review/batch", json=payload).json()
    assert [c["name"] for c in body["candidates"]] == ["metamer"]

    # filter: only passing all (identical sample)
    payload.pop("only_metameric")
    payload["only_passing_all"] = True
    body = client.post("/api/v1/review/batch", json=payload).json()
    assert [c["name"] for c in body["candidates"]] == ["identical"]


def test_batch_mean_sort_and_name_sort():
    samples = [
        {"name": "zeta", **spectrum(TARGET)},
        {"name": "alpha", **spectrum(METAMER)},
    ]
    payload = {"target": spectrum(TARGET), "samples": samples,
               "tolerance_de00": 2.0, "sort_by": "name"}
    body = client.post("/api/v1/review/batch", json=payload).json()
    assert [c["name"] for c in body["candidates"]] == ["alpha", "zeta"]
