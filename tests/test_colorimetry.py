import numpy as np

from app.colorimetry import (
    LabColor,
    ciede2000,
    xyz_to_lab,
)
from tests.de2000_cases import CIEDE2000_CASES


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
