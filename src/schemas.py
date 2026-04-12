"""
Pydantic models for Gemma LLM output validation.

Strict schema enforcement with:
- Enums for all categorical fields
- Bounded numerics (conviction 0-10, rank 1-5)
- Coherence rules: action=buy requires entry_zone/multipliers/horizon;
  action in (skip/hold/close) requires them null
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BtcRegime(str, Enum):
    trending_up = "trending_up"
    trending_down = "trending_down"
    range = "range"
    chaotic = "chaotic"


class MarketMode(str, Enum):
    risk_on = "risk_on"
    risk_off = "risk_off"
    rotation = "rotation"
    neutral = "neutral"


class Coherence(str, Enum):
    aligned = "aligned"
    diverging = "diverging"
    rotating = "rotating"


class SectorFlow(str, Enum):
    btc_led = "btc_led"
    alt_led = "alt_led"
    mixed = "mixed"
    defensive = "defensive"


class RiskAdjustment(str, Enum):
    normal = "normal"
    defensive = "defensive"
    aggressive = "aggressive"


class AssetRegime(str, Enum):
    trending_up = "trending_up"
    trending_down = "trending_down"
    range = "range"
    chaotic = "chaotic"


class SetupType(str, Enum):
    breakout = "breakout"
    pullback = "pullback"
    mean_reversion = "mean_reversion"
    none_ = "none"


class Action(str, Enum):
    buy = "buy"
    close = "close"
    hold = "hold"
    skip = "skip"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EntryZone(BaseModel):
    min: float
    max: float


class GlobalSection(BaseModel):
    btc_regime: BtcRegime
    market_mode: MarketMode
    market_coherence: Coherence
    sector_flow: SectorFlow
    relative_strength_ranking: List[str]
    risk_adjustment: RiskAdjustment
    max_concurrent_positions: int
    rationale: str

    @field_validator("relative_strength_ranking")
    @classmethod
    def ranking_must_have_5_unique(cls, v):
        if len(v) != 5:
            raise ValueError(f"ranking must have exactly 5 symbols, got {len(v)}")
        if len(set(v)) != 5:
            raise ValueError(f"ranking must have 5 unique symbols, got duplicates: {v}")
        return v

    @field_validator("max_concurrent_positions")
    @classmethod
    def max_pos_bounded(cls, v):
        if v < 1 or v > 5:
            raise ValueError(f"max_concurrent_positions must be 1-5, got {v}")
        return v


class AssetDecision(BaseModel):
    symbol: str
    regime: AssetRegime
    setup: SetupType
    action: Action
    conviction: int
    relative_strength_rank: int
    entry_zone: Optional[EntryZone] = None
    atr_stop_multiplier: Optional[float] = None
    atr_tp_multiplier: Optional[float] = None
    expected_horizon_hours: Optional[int] = None
    holistic_justification: str
    rationale: str

    @field_validator("conviction")
    @classmethod
    def conviction_bounded(cls, v):
        if v < 0 or v > 10:
            raise ValueError(f"conviction must be 0-10, got {v}")
        return v

    @field_validator("relative_strength_rank")
    @classmethod
    def rank_bounded(cls, v):
        if v < 1 or v > 5:
            raise ValueError(f"relative_strength_rank must be 1-5, got {v}")
        return v

    @model_validator(mode="after")
    def check_action_field_coherence(self):
        buy_fields = [
            self.entry_zone,
            self.atr_stop_multiplier,
            self.atr_tp_multiplier,
            self.expected_horizon_hours,
        ]
        if self.action == Action.buy:
            missing = []
            if self.entry_zone is None:
                missing.append("entry_zone")
            if self.atr_stop_multiplier is None:
                missing.append("atr_stop_multiplier")
            if self.atr_tp_multiplier is None:
                missing.append("atr_tp_multiplier")
            if self.expected_horizon_hours is None:
                missing.append("expected_horizon_hours")
            if missing:
                raise ValueError(
                    f"action=buy requires non-null: {', '.join(missing)}"
                )
        else:
            non_null = []
            if self.entry_zone is not None:
                non_null.append("entry_zone")
            if self.atr_stop_multiplier is not None:
                non_null.append("atr_stop_multiplier")
            if self.atr_tp_multiplier is not None:
                non_null.append("atr_tp_multiplier")
            if self.expected_horizon_hours is not None:
                non_null.append("expected_horizon_hours")
            if non_null:
                raise ValueError(
                    f"action={self.action.value} requires null: {', '.join(non_null)}"
                )
        return self


class Meta(BaseModel):
    analysis_confidence: int

    @field_validator("analysis_confidence")
    @classmethod
    def confidence_bounded(cls, v):
        if v < 0 or v > 10:
            raise ValueError(f"analysis_confidence must be 0-10, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------


class GemmaOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    global_: GlobalSection = Field(alias="global")
    assets: List[AssetDecision] = Field(min_length=5, max_length=5)
    meta: Meta

    @classmethod
    def parse_raw_response(cls, data: dict) -> "GemmaOutput":
        """Parse a raw dict from LLM. Accepts both 'global' and 'global_' keys."""
        return cls.model_validate(data)

    @model_validator(mode="after")
    def check_assets_count_and_symbols(self):
        if len(self.assets) != 5:
            raise ValueError(f"Must have exactly 5 assets, got {len(self.assets)}")

        asset_symbols = [a.symbol for a in self.assets]
        ranking_symbols = self.global_.relative_strength_ranking

        if set(asset_symbols) != set(ranking_symbols):
            raise ValueError(
                f"Asset symbols {asset_symbols} don't match "
                f"ranking symbols {ranking_symbols}"
            )
        return self
