from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    PARSE_ERROR = "PARSE_ERROR"
    REGISTRY_ERROR = "REGISTRY_ERROR"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ToolSuccessResponse(BaseModel):
    status: str = "success"
    data: Any


class ToolErrorResponse(BaseModel):
    status: str = "error"
    error_code: ToolErrorCode
    message: str


# --- Input Schemas ---


class InitializeSandalwoodInput(BaseModel):
    max_order: int = Field(..., description="Maximum truncation order.")
    max_dimension: int = Field(..., description="Maximum number of variables.")
    implementation: str = Field("cosy", description="Backend ('cosy' or 'python').")
    session_id: str = Field("default", description="Session isolation namespace.")


class ParseExpressionInput(BaseModel):
    expression: str = Field(..., description="Math formula (e.g., 'x1**2 + sin(x2)').")
    dimension: int = Field(..., description="Number of variables.")
    max_order: int = Field(..., description="Maximum truncation order.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class CreateTaylorMapInput(BaseModel):
    expressions: List[str] = Field(..., description="List of component expressions.")
    dimension: int = Field(..., description="Number of input variables.")
    max_order: int = Field(..., description="Maximum truncation order.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class EvaluateTaylorMapInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference name of the TaylorMap.")
    point: List[float] = Field(..., description="Coordinates to evaluate at.")
    session_id: str = Field("default", description="Session isolation namespace.")


class EvaluateMtfInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference name of the MTF.")
    point: List[float] = Field(..., description="Coordinates to evaluate at.")
    session_id: str = Field("default", description="Session isolation namespace.")


class InvertTaylorMapInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference name of the TaylorMap.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ComputePartialDerivativeInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference name of the MTF.")
    var_index: int = Field(
        ..., description="Variable index to differentiate with respect to (1-indexed)."
    )
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class IntegrateMtfInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference name of the MTF.")
    var_index: int = Field(
        ..., description="Variable index to integrate with respect to (1-indexed)."
    )
    lower_limit: Optional[float] = Field(None, description="Lower integration limit.")
    upper_limit: Optional[float] = Field(None, description="Upper integration limit.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ComposeTaylorMapsInput(BaseModel):
    map_ref_1: str = Field(
        ..., description="Registry reference of the outer TaylorMap."
    )
    map_ref_2: str = Field(
        ..., description="Registry reference of the inner TaylorMap."
    )
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ComposeMtfsInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the outer MTF.")
    inner_mtf_refs: dict = Field(..., description="Dict of inner MTF refs.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class PerformMtfArithmeticInput(BaseModel):
    mtf_ref_1: str = Field(
        ..., description="Registry reference or scalar value (e.g. '1.5')."
    )
    mtf_ref_2: str = Field(..., description="Registry reference or scalar value.")
    op: str = Field(..., description="Operation: '+', '-', '*', '/', '**'.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class PerformComplexOperationInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    op: str = Field(
        ..., description="Complex operation: 'real_part', 'imag_part', 'conjugate'."
    )
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class MtfInfoInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    session_id: str = Field("default", description="Session isolation namespace.")


class TaylorMapInfoInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    session_id: str = Field("default", description="Session isolation namespace.")


class SubstituteVariableInMtfInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    var_index: int = Field(..., description="Variable index to substitute (1-indexed).")
    value: float = Field(..., description="Value to substitute.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class SubstituteInTaylorMapInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    variable_map: dict = Field(
        ..., description="Dict mapping string indices to numeric values."
    )
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class GetMtfCoefficientInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    exponents: List[int] = Field(..., description="Exponents list (e.g., [1, 0, 2]).")
    session_id: str = Field("default", description="Session isolation namespace.")


class TruncateObjectInput(BaseModel):
    ref: str = Field(..., description="Registry reference of the MTF or TaylorMap.")
    order: int = Field(..., description="Order to truncate to.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class AnalyzeTaylorMapInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    session_id: str = Field("default", description="Session isolation namespace.")


class EvaluateMtfBatchInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    points: List[List[float]] = Field(
        ..., description="List of coordinate points to evaluate."
    )
    session_id: str = Field("default", description="Session isolation namespace.")


class EvaluateTaylorMapBatchInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    points: List[List[float]] = Field(
        ..., description="List of coordinate points to evaluate."
    )
    session_id: str = Field("default", description="Session isolation namespace.")


class AnalyzeMtfDiagnosticsInput(BaseModel):
    mtf_ref: str = Field(..., description="Registry reference of the MTF.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ComputePoissonBracketInput(BaseModel):
    mtf_ref_1: str = Field(..., description="Registry reference of the first MTF.")
    mtf_ref_2: str = Field(..., description="Registry reference of the second MTF.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ComputeMapSensitivityInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    scaling_factors: List[float] = Field(..., description="Scaling factors list.")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")


class ExtractMapComponentInput(BaseModel):
    map_ref: str = Field(..., description="Registry reference of the TaylorMap.")
    index: int = Field(..., description="Component index to extract (0-indexed).")
    name: Optional[str] = Field(None, description="Optional custom registry name.")
    session_id: str = Field("default", description="Session isolation namespace.")
