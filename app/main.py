"""Spectral colour-match review API.

Endpoints
----------
GET  /health
GET  /illuminants                 built-in illuminant catalogue
POST /api/v1/review               single target/sample pair, many illuminants
POST /api/v1/review/batch         one target, many samples; filter + sort
POST /api/v1/review/uncertainty   repeat-scan bootstrap of Delta E00 uncertainty
POST /api/v1/review/strength      Kubelka-Munk K/S dye-strength split
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError as FastAPIValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .colorimetry import BUILTIN_ILLUMINANTS, WL_MAX, WL_MIN, illuminant_table
from .engine import evaluate_pair, evaluate_uncertainty, resolve_illuminants
from .schemas import (
    BatchReviewRequest,
    ReviewRequest,
    StrengthAnalysisRequest,
    UncertaintyReviewRequest,
)
from .strength import evaluate_strength
from .validation import (
    RequestValidationError,
    parse_spectrum,
)

app = FastAPI(
    title="Spectral Colour-Match Review API",
    version="1.0.0",
    description=(
        "CIE 1931 2-degree XYZ/Lab/CIEDE2000 review of reflectance pairs under "
        "D65, A, F11 or uploaded SPDs, with metamerism-risk flagging and "
        "wavelength-band attribution."
    ),
)


@app.exception_handler(RequestValidationError)
async def _domain_validation_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "issues": exc.issues},
    )


def _format_pydantic_errors(errors) -> list[dict]:
    issues = []
    for err in errors:
        loc = tuple(err["loc"])
        # FastAPI wraps body errors as ("body", ...); drop the "body" prefix.
        if loc and loc[0] in ("body", "query", "path"):
            loc = loc[1:]
        field = ".".join(
            f"[{p}]" if isinstance(p, int) else str(p) for p in loc
        ) or "request"
        issues.append({
            "field": field,
            "code": err["type"],
            "reason": err["msg"],
        })
    return issues


@app.exception_handler(FastAPIValidationError)
async def _fastapi_validation_handler(_request: Request, exc: FastAPIValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error",
                 "issues": _format_pydantic_errors(exc.errors())},
    )


@app.exception_handler(ValidationError)
async def _pydantic_validation_handler(_request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error",
                 "issues": _format_pydantic_errors(exc.errors())},
    )


def _parse_pair(payload) -> tuple:
    issues: list[dict] = []
    target = parse_spectrum(
        payload.target.model_dump(), "target", "target",
        is_reflectance=True, issues=issues,
    )
    sample = parse_spectrum(
        payload.sample.model_dump(), "sample", "sample",
        is_reflectance=True, issues=issues,
    )
    illuminants = None
    try:
        illuminants = resolve_illuminants(payload.illuminants, "illuminants")
    except RequestValidationError as exc:
        issues.extend(exc.issues)
    if issues:
        raise RequestValidationError(issues)
    return target, sample, illuminants


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/illuminants")
async def list_illuminants():
    table = illuminant_table()
    wl = table["wavelengths"]
    return {
        "built_in": [
            {
                "name": name,
                "wavelength_range_nm": [float(wl.min()), float(wl.max())],
                "grid_step_nm": float(wl[1] - wl[0]),
                "points": int(wl.size),
            }
            for name in BUILTIN_ILLUMINANTS
        ],
        "custom": {
            "accepted": True,
            "required_coverage_nm": [WL_MIN, WL_MAX],
            "description": "relative spectral power; non-negative, zero integral rejected",
        },
        "observer": "CIE 1931 2-degree standard observer",
    }


@app.post("/api/v1/review")
async def review(req: ReviewRequest):
    target, sample, illuminants = _parse_pair(req)
    result = evaluate_pair(
        target, sample, illuminants,
        step_nm=req.grid_step_nm,
        tolerance=req.tolerance_de00,
        band_width_nm=req.band_width_nm,
        top_bands=req.top_bands,
    )
    return {
        "tolerance_de00": req.tolerance_de00,
        "grid_step_nm": req.grid_step_nm,
        **result,
    }


@app.post("/api/v1/review/uncertainty")
async def review_uncertainty(req: UncertaintyReviewRequest):
    issues: list[dict] = []

    target_scans = []
    for i, scan in enumerate(req.target.scans):
        sp = parse_spectrum(
            scan.model_dump(), f"target.scans[{i}]", f"target[{i}]",
            is_reflectance=True, issues=issues,
        )
        if sp is not None:
            target_scans.append(sp)
    sample_scans = []
    for i, scan in enumerate(req.sample.scans):
        sp = parse_spectrum(
            scan.model_dump(), f"sample.scans[{i}]", f"sample[{i}]",
            is_reflectance=True, issues=issues,
        )
        if sp is not None:
            sample_scans.append(sp)

    try:
        illuminants = resolve_illuminants(req.illuminants, "illuminants")
    except RequestValidationError as exc:
        issues.extend(exc.issues)

    if not target_scans:
        issues.append({
            "field": "target.scans",
            "code": "no_valid_scan",
            "reason": "at least one valid target reflectance scan is required",
        })
    if not sample_scans:
        issues.append({
            "field": "sample.scans",
            "code": "no_valid_scan",
            "reason": "at least one valid sample reflectance scan is required",
        })

    if issues:
        raise RequestValidationError(issues)

    return evaluate_uncertainty(
        target_scans, sample_scans, illuminants,
        step_nm=req.grid_step_nm,
        tolerance=req.tolerance_de00,
        seed=req.seed,
        draws=req.draws,
        prob_bound=req.critical_probability_bound,
    )


@app.post("/api/v1/review/strength")
async def review_strength(req: StrengthAnalysisRequest):
    target, sample, illuminants = _parse_pair(req)

    issues: list[dict] = []
    band = None
    if req.analysis_band_nm is not None:
        lo, hi = (float(v) for v in req.analysis_band_nm)
        if req.primary_wavelength_nm is not None:
            issues.append({
                "field": "analysis_band_nm",
                "code": "analysis_scope_ambiguous",
                "reason": (
                    "provide either 'analysis_band_nm' or "
                    "'primary_wavelength_nm', not both"
                ),
            })
        elif not lo < hi:
            issues.append({
                "field": "analysis_band_nm",
                "code": "band_not_increasing",
                "reason": f"analysis band must satisfy lo < hi, got [{lo:g}, {hi:g}]",
            })
        else:
            band = (lo, hi)
    if issues:
        raise RequestValidationError(issues)

    result = evaluate_strength(
        target, sample, illuminants,
        step_nm=req.grid_step_nm,
        primary_wavelength_nm=req.primary_wavelength_nm,
        analysis_band_nm=band,
        strength_tolerance=req.strength_tolerance,
        residual_tolerance_de00=req.residual_tolerance_de00,
    )
    return {
        "grid_step_nm": req.grid_step_nm,
        "strength_tolerance": req.strength_tolerance,
        "residual_tolerance_de00": req.residual_tolerance_de00,
        **result,
    }


@app.post("/api/v1/review/batch")
async def review_batch(req: BatchReviewRequest):
    issues: list[dict] = []
    target = parse_spectrum(
        req.target.model_dump(), "target", "target",
        is_reflectance=True, issues=issues,
    )
    parsed_samples = []
    for i, sample in enumerate(req.samples):
        sp = parse_spectrum(
            sample.model_dump(), f"samples[{i}]", sample.name,
            is_reflectance=True, issues=issues,
        )
        if sp is not None:
            parsed_samples.append((i, sample.name, sp))

    try:
        illuminants = resolve_illuminants(req.illuminants, "illuminants")
    except RequestValidationError as exc:
        issues.extend(exc.issues)

    if issues:
        raise RequestValidationError(issues)

    candidates = []
    for idx, name, sp in parsed_samples:
        result = evaluate_pair(
            target, sp, illuminants,
            step_nm=req.grid_step_nm,
            tolerance=req.tolerance_de00,
            band_width_nm=req.band_width_nm,
            top_bands=req.top_bands,
        )
        candidates.append({
            "index": idx,
            "name": name,
            "max_de00": result["max_de00"],
            "mean_de00": result["mean_de00"],
            "pass_all": result["pass_all"],
            "metamerism_risk": result["metamerism_risk"],
            "results_by_illuminant": result["results_by_illuminant"],
        })

    if req.only_passing_all:
        candidates = [c for c in candidates if c["pass_all"]]
    if req.only_metameric:
        candidates = [c for c in candidates if c["metamerism_risk"]["flag"]]

    if req.sort_by == "max_de00":
        candidates.sort(key=lambda c: c["max_de00"])
    elif req.sort_by == "mean_de00":
        candidates.sort(key=lambda c: c["mean_de00"])
    else:
        candidates.sort(key=lambda c: c["name"].lower())

    return {
        "tolerance_de00": req.tolerance_de00,
        "grid_step_nm": req.grid_step_nm,
        "submitted": len(req.samples),
        "returned": len(candidates),
        "candidates": candidates,
    }
