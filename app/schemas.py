"""Request/response Pydantic schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpectrumInput(BaseModel):
    wavelengths: list[float] = Field(
        ..., description="Wavelengths in nm, e.g. [400, 410, ..., 700].")
    values: list[float] = Field(
        ..., description="Reflectance factor in [0, 1] (or relative power).")


class NamedSample(SpectrumInput):
    name: str = Field(..., min_length=1, max_length=128)


class IlluminantSpec(BaseModel):
    """A built-in name or a custom uploaded SPD.

    Provide ``name`` for built-ins (D65, A, F11). For custom SPDs provide
    ``custom`` with the wavelength/power series and a label.
    """

    name: str | None = Field(None, description="Built-in illuminant: D65, A, F11.")
    custom: SpectrumInput | None = None
    label: str | None = Field(None, max_length=64)


class ReviewRequest(BaseModel):
    target: SpectrumInput
    sample: SpectrumInput
    illuminants: list[IlluminantSpec] = Field(
        default_factory=lambda: [
            IlluminantSpec(name="D65"),
            IlluminantSpec(name="A"),
            IlluminantSpec(name="F11"),
        ],
        min_length=1,
        max_length=8,
    )
    tolerance_de00: float = Field(
        2.0, gt=0, le=50,
        description="CIEDE2000 pass threshold; samples at/under it pass.",
    )
    grid_step_nm: float = Field(10.0, ge=5, le=20)
    band_width_nm: float = Field(
        20.0, ge=10, le=100,
        description="Aggregation width for the dominant-wavelength-band report.",
    )
    top_bands: int = Field(3, ge=1, le=20)


class BatchReviewRequest(BaseModel):
    target: SpectrumInput
    samples: list[NamedSample] = Field(..., min_length=1, max_length=100)
    illuminants: list[IlluminantSpec] = Field(
        default_factory=lambda: [
            IlluminantSpec(name="D65"),
            IlluminantSpec(name="A"),
            IlluminantSpec(name="F11"),
        ],
        min_length=1,
        max_length=8,
    )
    tolerance_de00: float = Field(2.0, gt=0, le=50)
    grid_step_nm: float = Field(10.0, ge=5, le=20)
    band_width_nm: float = Field(20.0, ge=10, le=100)
    top_bands: int = Field(3, ge=1, le=20)
    sort_by: Literal["max_de00", "mean_de00", "name"] = "max_de00"
    only_passing_all: bool = Field(
        False,
        description="If true, keep only samples passing under every requested illuminant.",
    )
    only_metameric: bool = Field(
        False,
        description="If true, keep only samples flagged with metamerism risk.",
    )


class ValidationIssue(BaseModel):
    field: str
    code: str
    reason: str


class ErrorResponse(BaseModel):
    error: Literal["validation_error"] = "validation_error"
    issues: list[dict[str, Any]]
