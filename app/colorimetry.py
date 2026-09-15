"""CIE colorimetry: interpolation, XYZ, Lab and CIEDE2000.

All integration is trapezoidal over a common wavelength grid. Reference data
(CIE 1931 2-deg CMF and A/D65/F11 illuminant SPDs) are the 5 nm CIE 15:2004
tables shipped in app/data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# Visible range on which all reference tables are defined.
WL_MIN = 380.0
WL_MAX = 780.0
DEFAULT_STEP = 10.0

REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 1.0


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration (works on NumPy <2 and >=2)."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(y, x)

# CIE 1931 2-deg reference white point for D65 (normalised Y = 100).
D65_WHITE_XY = (0.31271, 0.32902)


@lru_cache(maxsize=1)
def cmf_table() -> dict[str, np.ndarray]:
    data = json.loads((DATA_DIR / "cmf_cie1931_2deg.json").read_text())
    return {
        "wavelengths": np.asarray(data["wavelengths_nm"], dtype=float),
        "x_bar": np.asarray(data["x_bar"], dtype=float),
        "y_bar": np.asarray(data["y_bar"], dtype=float),
        "z_bar": np.asarray(data["z_bar"], dtype=float),
    }


@lru_cache(maxsize=1)
def illuminant_table() -> dict:
    data = json.loads((DATA_DIR / "illuminants.json").read_text())
    wavelengths = np.asarray(data["wavelengths_nm"], dtype=float)
    return {
        "wavelengths": wavelengths,
        "power": {
            name: np.asarray(item["power"], dtype=float)
            for name, item in data["illuminants"].items()
        },
    }


BUILTIN_ILLUMINANTS = ("D65", "A", "F11")


@dataclass(frozen=True)
class LabColor:
    L: float
    a: float
    b: float

    def as_dict(self) -> dict[str, float]:
        return {"L": self.L, "a": self.a, "b": self.b}


def make_grid(step: float = DEFAULT_STEP,
              wl_min: float = WL_MIN,
              wl_max: float = WL_MAX) -> np.ndarray:
    """Common wavelength grid, inclusive of both ends."""
    n = int(round((wl_max - wl_min) / step)) + 1
    return np.round(wl_min + np.arange(n) * step, 4)


def interp_linear(wavelengths: np.ndarray, values: np.ndarray,
                  grid: np.ndarray) -> np.ndarray:
    """Linear interpolation; raises ValueError outside the support range."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(wavelengths)
    wavelengths, values = wavelengths[order], values[order]
    if grid[0] < wavelengths[0] - 1e-9 or grid[-1] > wavelengths[-1] + 1e-9:
        raise ValueError(
            f"requested grid [{grid[0]:g}, {grid[-1]:g}] nm is not covered by "
            f"data support [{wavelengths[0]:g}, {wavelengths[-1]:g}] nm"
        )
    return np.interp(grid, wavelengths, values)


def cmf_on_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the CIE 1931 CMFs (piecewise linear) onto *grid*."""
    t = cmf_table()
    w = t["wavelengths"]
    return (
        interp_linear(w, t["x_bar"], grid),
        interp_linear(w, t["y_bar"], grid),
        interp_linear(w, t["z_bar"], grid),
    )


def illuminant_on_grid(name: str, grid: np.ndarray) -> np.ndarray:
    t = illuminant_table()
    power = interp_linear(t["wavelengths"], t["power"][name], grid)
    if np.any(power < 0):
        raise ValueError(f"illuminant {name} has negative power after resampling")
    return power


def tristimulus(reflectance: np.ndarray, power: np.ndarray,
                grid: np.ndarray,
                x_bar: np.ndarray, y_bar: np.ndarray, z_bar: np.ndarray,
                ) -> np.ndarray:
    """XYZ (Y = 100 for a perfect white reflector)."""
    denom = trapz(power * y_bar, grid)
    if denom <= 0:
        raise ZeroDivisionError(
            "normalisation denominator integral(S(lambda) y_bar(lambda)) is zero"
        )
    x = 100.0 * trapz(power * x_bar * reflectance, grid) / denom
    y = 100.0 * trapz(power * y_bar * reflectance, grid) / denom
    z = 100.0 * trapz(power * z_bar * reflectance, grid) / denom
    return np.array([x, y, z], dtype=float)


def d65_white_xyz() -> tuple[float, float, float]:
    x, y = D65_WHITE_XY
    return (100.0 * x / y, 100.0, 100.0 * (1.0 - x - y) / y)


def xyz_to_lab(xyz: np.ndarray,
               white: tuple[float, float, float] = d65_white_xyz()) -> LabColor:
    """CIE XYZ -> CIE L*a*b* (D65 reference white by default)."""
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = xyz[0] / white[0]
    yr = xyz[1] / white[1]
    zr = xyz[2] / white[2]

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > eps else (kappa * t + 16.0) / 116.0

    fx, fy, fz = f(xr), f(yr), f(zr)
    return LabColor(L=116.0 * fy - 16.0, a=500.0 * (fx - fy), b=200.0 * (fy - fz))


def ciede2000(lab1: LabColor, lab2: LabColor,
              k_l: float = 1.0, k_c: float = 1.0,
              k_h: float = 1.0) -> float:
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005 formulation)."""
    l1, a1, b1 = lab1.L, lab1.a, lab1.b
    l2, a2, b2 = lab2.L, lab2.a, lab2.b

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p

    if c1p * c2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dhp_rad = np.radians(dhp)
    d_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(dhp_rad / 2.0)

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    if c1p * c2p == 0.0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        h_bar_p = 0.5 * (h1p + h2p)
    elif h1p + h2p < 360.0:
        h_bar_p = 0.5 * (h1p + h2p + 360.0)
    else:
        h_bar_p = 0.5 * (h1p + h2p - 360.0)

    t = (1.0
         - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
         + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0)))

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p ** 7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0 ** 7))
    s_l = (1.0
           + (0.015 * (l_bar_p - 50.0) ** 2)
           / np.sqrt(20.0 + (l_bar_p - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    term_l = dlp / (k_l * s_l)
    term_c = dcp / (k_c * s_c)
    term_h = d_hp / (k_h * s_h)
    return float(np.sqrt(
        term_l ** 2 + term_c ** 2 + term_h ** 2
        + r_t * term_c * term_h
    ))
