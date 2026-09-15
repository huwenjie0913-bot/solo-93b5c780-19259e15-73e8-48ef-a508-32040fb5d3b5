"""Independent oracle test against colour-science (skipped if not installed).

Verifies the whole spectral pipeline (SPD + CMF trapezoidal integration,
per-illuminant adapted white, Lab, CIEDE2000) against a second implementation.
"""
import warnings

import numpy as np
import pytest

colour = pytest.importorskip("colour")
warnings.filterwarnings("ignore")

from colour import (  # noqa: E402
    SpectralDistribution,
    SpectralShape,
    XYZ_to_Lab,
    XYZ_to_xy,
    sd_to_XYZ,
)
from colour.colorimetry import MSDS_CMFS, SDS_ILLUMINANTS  # noqa: E402
from colour.difference import delta_E_CIE2000  # noqa: E402

from app.colorimetry import (  # noqa: E402
    ciede2000,
    cmf_on_grid,
    illuminant_on_grid,
    make_grid,
    tristimulus,
    xyz_to_lab,
)
from tests.fixtures_data import METAMER, TARGET  # noqa: E402

CMF = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
# our API name -> colour table key
ILL_NAME = {"D65": "D65", "A": "A", "F11": "FL11"}


def _colour_xyz(values, grid, colour_name):
    sd_r = SpectralDistribution(dict(zip(grid, values)))
    sd_ill = SDS_ILLUMINANTS[colour_name].copy().align(
        SpectralShape(380, 780, 5))
    cmf = CMF.copy().align(SpectralShape(380, 780, 5))
    return sd_to_XYZ(sd_r, cmfs=cmf, illuminant=sd_ill)


def _colour_white(grid, colour_name):
    sd_ill = SDS_ILLUMINANTS[colour_name].copy().align(
        SpectralShape(380, 780, 5))
    cmf = CMF.copy().align(SpectralShape(380, 780, 5))
    white_sd = SpectralDistribution(dict(zip(grid, np.ones_like(grid))))
    return sd_to_XYZ(white_sd, cmfs=cmf, illuminant=sd_ill)


@pytest.mark.parametrize("name", ["D65", "A", "F11"])
def test_xyz_lab_de000_match_colour_science(name):
    grid5 = make_grid(5.0)
    xb, yb, zb = cmf_on_grid(grid5)
    rt5 = np.interp(grid5, make_grid(10.0), TARGET)
    rs5 = np.interp(grid5, make_grid(10.0), METAMER)
    power = illuminant_on_grid(name, grid5)

    for ours_r, ref_r in ((rt5, TARGET), (rs5, METAMER)):
        xyz = tristimulus(ours_r, power, grid5, xb, yb, zb)
        white = tristimulus(np.ones_like(grid5), power, grid5, xb, yb, zb)
        ref = _colour_xyz(ours_r, grid5, ILL_NAME[name])
        ref_white = _colour_white(grid5, ILL_NAME[name])
        # independent XYZ integration: within 0.02 (10 nm->5 nm input rounding)
        assert np.abs(xyz - ref).max() < 0.02
        assert np.abs(white - ref_white).max() < 0.02

    xyz_t = tristimulus(rt5, power, grid5, xb, yb, zb)
    xyz_s = tristimulus(rs5, power, grid5, xb, yb, zb)
    white = tristimulus(np.ones_like(grid5), power, grid5, xb, yb, zb)
    lab_t = xyz_to_lab(xyz_t, tuple(float(v) for v in white))
    lab_s = xyz_to_lab(xyz_s, tuple(float(v) for v in white))
    de = ciede2000(lab_t, lab_s)

    clab_t = XYZ_to_Lab(_colour_xyz(rt5, grid5, ILL_NAME[name]) / 100.0,
                        XYZ_to_xy(_colour_white(grid5, ILL_NAME[name])))
    clab_s = XYZ_to_Lab(_colour_xyz(rs5, grid5, ILL_NAME[name]) / 100.0,
                        XYZ_to_xy(_colour_white(grid5, ILL_NAME[name])))
    ref_de = float(delta_E_CIE2000(clab_t, clab_s))
    assert abs(de - ref_de) < 0.002
