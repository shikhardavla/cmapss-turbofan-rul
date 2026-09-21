from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """
    One raw cycle of readings, matching the CMAPSS column schema exactly
    (see src/data/loader.py's column_names).
    """
    time_cycles: int
    operational_setting_1: float
    operational_setting_2: float
    operational_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    dataset: Literal["FD001", "FD002", "FD003", "FD004"]
    model: Literal["lstm", "tcn", "transformer"] = "transformer"
    unit_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied identifier, echoed back in the response.",
    )
    cycles: List[SensorReading] = Field(
        ...,
        description=(
            "The engine's most recent cycles, in chronological order. "
            "Must contain exactly window_size cycles (see config.yaml)."
        ),
    )

    @field_validator("cycles")
    @classmethod
    def cycles_must_be_chronological(cls, cycles: List[SensorReading]) -> List[SensorReading]:
        cycle_numbers = [c.time_cycles for c in cycles]
        if cycle_numbers != sorted(cycle_numbers):
            raise ValueError("cycles must be sorted in ascending order of time_cycles")
        return cycles


class PredictionResponse(BaseModel):
    dataset: str
    model: str
    unit_id: Optional[str] = None
    predicted_rul: float
