import numpy as np

from app.colorimetry import (
    LabColor,
    ciede2000,
    ciede2000_array,
    d65_white_xyz,
    xyz_to_lab,
    xyz_to_lab_array,
)
from tests.de2000_cases import CIEDE2000_CASES

# D65 white actually used by the app (X/Y/Z normalised to Y = 100).
_WHITE = d65_white_xyz()

# CIE XYZ -> Lab constants used by the independent reference below.
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _f_linear_reference(t):
    """Correct CIE low-value branch: (kappa*t + 16)/116."""
    return t ** (1.0 / 3.0) if t > _EPS else (_KAPPA * t + 16.0) / 116.0


def _lab_reference(xyz, white):
    fx = _f_linear_reference(xyz[0] / white[0])
    fy = _f_linear_reference(xyz[1] / white[1])
    fz = _f_linear_reference(xyz[2] / white[2])
    return np.array([
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    ])


def test_ciede2000_sharma_reference_rows():
    max_err = 0.0
    for (l1, a1, b1), (l2, a2, b2), expected in CIEDE2000_CASES:
        got = ciede2000(LabColor(l1, a1, b1), LabColor(l2, a2, b2))
        max_err = max(max_err, abs(got - expected))
        assert abs(got - expected) < 1e-3, (
            f"({l1},{a1},{b1}) vs ({l2},{a2},{b2}): got {got:.4f}, want {expected}")
    assert max_err < 5e-4


def test_ciede2000_identity_and_symmetry():
    lab = LabColor(55.1, -23.4, 41.7)
    assert ciede2000(lab, lab) < 1e-12
    other = LabColor(60.0, -20.0, 45.0)
    assert abs(ciede2000(lab, other) - ciede2000(other, lab)) < 1e-12


def test_lab_known_neutrals():
    black = xyz_to_lab(np.array([0.0, 0.0, 0.0]))
    assert abs(black.L) < 1e-9 and abs(black.a) < 1e-9 and abs(black.b) < 1e-9

    # D65 white point must map to L*=100, a*=b*=0.
    from app.colorimetry import d65_white_xyz
    white = xyz_to_lab(np.array(d65_white_xyz()))
    assert abs(white.L - 100.0) < 1e-9
    assert abs(white.a) < 1e-9 and abs(white.b) < 1e-9


def test_lab_known_red():
    # sRGB pure red (1,0,0) D65 -> roughly L*=53.24, a*=80.09, b*=67.20
    lab = xyz_to_lab(np.array([41.2456, 21.2673, 1.9334]))
    assert abs(lab.L - 53.24) < 0.05
    assert abs(lab.a - 80.09) < 0.1
    assert abs(lab.b - 67.20) < 0.1


def test_lab_nonzero_low_value_linear_branch():
    """Regression: the shared _f_lab low branch must be (kappa*t+16)/116,
    not kappa*t + 16/116 (an operator-precedence slip from vectorisation that
    made all dark, non-black samples too bright)."""
    # flat 0.5 % reflector under D65: X=Xw*r, Y=Yw*r, Z=Zw*r (all < eps)
    r = 0.005
    xyz = np.asarray(_WHITE) * r
    lab = xyz_to_lab(xyz)
    expected_L = 116.0 * ((_KAPPA * r + 16.0) / 116.0) - 16.0
    # the broken formula would return ~19.95 instead of the correct ~6.52
    assert abs(lab.L - expected_L) < 1e-9
    assert lab.L < 7.0
    # neutral reflectance stays neutral: a* = b* = 0 exactly
    assert abs(lab.a) < 1e-9 and abs(lab.b) < 1e-9

    # slightly different non-neutral dark ratios still exercise x/z branches
    xyz2 = np.array([0.1, 0.3, 0.9])
    ref = _lab_reference(xyz2, _WHITE)
    lab2 = xyz_to_lab(xyz2)
    assert abs(lab2.L - ref[0]) < 1e-9
    assert abs(lab2.a - ref[1]) < 1e-9
    assert abs(lab2.b - ref[2]) < 1e-9


def test_lab_array_matches_scalar_including_dark_samples():
    rng = np.random.default_rng(7)
    # span both branches: many ratios below eps, a few above
    xyz_rows = np.vstack([
        rng.uniform(0.0, 1.0, size=(50, 3)),
        rng.uniform(0.0, 110.0, size=(10, 3)),
    ])
    got = xyz_to_lab_array(xyz_rows)
    ref = np.vstack([_lab_reference(row, _WHITE) for row in xyz_rows])
    assert np.max(np.abs(got - ref)) < 1e-9


def test_ciede2000_array_matches_scalar_on_dark_pairs():
    rng = np.random.default_rng(11)
    lab1 = rng.uniform([0.0, -5.0, -5.0], [20.0, 5.0, 5.0], size=(64, 3))
    lab2 = rng.uniform([0.0, -5.0, -5.0], [20.0, 5.0, 5.0], size=(64, 3))
    vector = ciede2000_array(lab1, lab2)
    for row1, row2, expected in zip(lab1, lab2, vector):
        scalar = ciede2000(LabColor(*row1), LabColor(*row2))
        assert abs(scalar - float(expected)) < 1e-10
