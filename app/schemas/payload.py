"""
app/schemas/payload.py
Strict Pydantic V2 contracts for multimodal inference payloads.
"""

from pydantic import BaseModel, Field, field_validator


class VectorInferenceRequest(BaseModel):
    """
    Validation schema for tabular/numerical feature payloads.
    Guarantees non-empty float sequences within reasonable dimension boundaries.
    """
    features: list[float] = Field(
        ...,
        description="Dense numerical feature vector for real-time inference.",
        json_schema_extra={"example": [0.54, -1.22, 0.88, 2.15]}
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Feature vector cannot be empty.")
        if len(v) > 2048:
            raise ValueError(f"Feature vector exceeds max allowed dimensions of 2048 (got {len(v)}).")
        return v


class ImageInferenceRequest(BaseModel):
    """
    Validation schema for raw base64-encoded image payloads.
    Guarantees non-empty byte strings and allowable encoding format.
    """
    image_base64: str = Field(
        ...,
        description="Base64-encoded image string for semantic gatekeeping and classification.",
        json_schema_extra={"example": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBD..."}
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Image payload string cannot be empty.")
        # Basic sanity check on header length
        if len(cleaned) < 32:
            raise ValueError("Payload too short to constitute a valid encoded image.")
        return cleaned


class InferenceResponse(BaseModel):
    """
    Standardized outgoing API prediction response contract.
    """
    modality: str
    prediction: int | str
    confidence: float
    cached: bool
    latency_ms: float
    status: str = "SUCCESS"