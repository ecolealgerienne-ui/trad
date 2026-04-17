"""
Pydantic models for LLM output validation — simplified v5 schema.

Only 6 fields per asset. No global section.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class Action(str, Enum):
    buy = "buy"
    close = "close"
    hold = "hold"
    skip = "skip"


class AssetDecision(BaseModel):
    symbol: str
    action: Action
    conviction: int
    stop_mult: Optional[float] = None
    tp_mult: Optional[float] = None
    rationale: str = ""

    @field_validator("conviction")
    @classmethod
    def conviction_bounded(cls, v):
        if v < 0 or v > 10:
            raise ValueError(f"conviction must be 0-10, got {v}")
        return v

    @model_validator(mode="after")
    def check_null_rules(self):
        if self.action == Action.buy:
            missing = []
            if self.stop_mult is None:
                missing.append("stop_mult")
            if self.tp_mult is None:
                missing.append("tp_mult")
            if missing:
                raise ValueError(f"action=buy requires: {', '.join(missing)}")
        else:
            non_null = []
            if self.stop_mult is not None:
                non_null.append("stop_mult")
            if self.tp_mult is not None:
                non_null.append("tp_mult")
            if non_null:
                raise ValueError(f"action={self.action.value} requires null: {', '.join(non_null)}")
        return self


class QwenOutput(BaseModel):
    assets: List[AssetDecision] = Field(min_length=5, max_length=5)

    @classmethod
    def parse_raw_response(cls, data: dict) -> "QwenOutput":
        return cls.model_validate(data)
