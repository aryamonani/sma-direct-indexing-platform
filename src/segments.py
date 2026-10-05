"""
Customer segmentation: each segment pre-loads typical constraints and sets the
optimizer's priorities. This is how a real platform scales customization across
thousands of accounts: segments supply the template, individual client overrides
tweak from there.

Three segments:
- Institution (e.g. endowment): screens-heavy, often tax-exempt or lightly taxed,
  so optimizer weights tracking error high and harvesting low.
- Family office: idiosyncratic exclusions, high tax sensitivity, often arrives
  with a concentrated legacy position (the transition problem). Harvesting high.
- HNW individual: concentration-risk screen for own-sector exposure, high tax
  sensitivity, templated constraints. Harvesting high.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .screens import ScreenSet
from .tax import TaxRates


class Segment(str, Enum):
    INSTITUTION = "institution"
    FAMILY_OFFICE = "family_office"
    HNW = "hnw_individual"


@dataclass
class OptimizerPriorities:
    """
    How the optimizer weighs competing objectives for an account.

    harvest_weight: how aggressively to pursue tax-loss harvesting
    tracking_weight: how strictly to hold tracking error down
    turnover_weight: how much to penalize trading (lower = more trades)

    These sum informally; the engine uses relative magnitudes.
    """
    harvest_weight: float
    tracking_weight: float
    turnover_weight: float


@dataclass
class SegmentProfile:
    """Default template for a client segment."""
    segment: Segment
    label: str
    description: str
    default_screens: ScreenSet
    default_tax_rates: TaxRates
    default_priorities: OptimizerPriorities
    typical_aum_range: tuple[float, float]
    notes: list[str]


SEGMENT_PROFILES: dict[Segment, SegmentProfile] = {
    Segment.INSTITUTION: SegmentProfile(
        segment=Segment.INSTITUTION,
        label="Institution (Endowment / Foundation)",
        description=(
            "Tax-exempt or lightly taxed. Mandate-driven screens are common "
            "(fossil-fuel-free, tobacco-free, weapons-free). Priority is index "
            "tracking; tax harvesting delivers little benefit."
        ),
        default_screens=ScreenSet(
            thematic_screens={"fossil_fuels", "tobacco", "weapons"},
        ),
        default_tax_rates=TaxRates(
            short_term=0.0, long_term=0.0, state=0.0, niit=0.0
        ),
        default_priorities=OptimizerPriorities(
            harvest_weight=0.1,
            tracking_weight=1.0,
            turnover_weight=0.5,
        ),
        typical_aum_range=(5_000_000, 500_000_000),
        notes=[
            "Tax-exempt status means harvested losses carry no tax benefit; the "
            "optimizer should barely touch harvesting here.",
            "Investment policy statements (IPS) often lock the screens in; these "
            "are usually long-standing mandates, not optional preferences.",
        ],
    ),
    Segment.FAMILY_OFFICE: SegmentProfile(
        segment=Segment.FAMILY_OFFICE,
        label="Family Office (UHNW)",
        description=(
            "Highly idiosyncratic exclusions (specific competitor, legacy "
            "position, personal exclusion). High tax sensitivity. Transition "
            "problem is common: client arrives already holding concentrated stock."
        ),
        default_screens=ScreenSet(
            # Family offices typically specify these per-client; empty default.
        ),
        default_tax_rates=TaxRates(
            short_term=0.37, long_term=0.20, state=0.133, niit=0.038
        ),
        default_priorities=OptimizerPriorities(
            harvest_weight=1.0,
            tracking_weight=0.6,
            turnover_weight=0.3,
        ),
        typical_aum_range=(10_000_000, 100_000_000),
        notes=[
            "Transition management is often the first conversation: how to move "
            "from legacy holdings into the custom index while respecting a gain "
            "budget.",
            "Multiple accounts per client (grantor trust, GRAT, foundation) often "
            "share screens but differ in tax treatment.",
        ],
    ),
    Segment.HNW: SegmentProfile(
        segment=Segment.HNW,
        label="HNW Individual",
        description=(
            "High marginal tax rates. Common screen: exclude the sector of their "
            "own compensation (concentration risk, e.g. a tech executive loaded "
            "with company RSUs excludes Information Technology). Templated."
        ),
        default_screens=ScreenSet(),
        default_tax_rates=TaxRates(
            short_term=0.37, long_term=0.20, state=0.093, niit=0.038
        ),
        default_priorities=OptimizerPriorities(
            harvest_weight=1.0,
            tracking_weight=0.5,
            turnover_weight=0.3,
        ),
        typical_aum_range=(250_000, 10_000_000),
        notes=[
            "The 'concentration risk' screen is the biggest driver of adoption "
            "here: equity-rich employees reducing portfolio correlation with "
            "their employer.",
            "Minimum SMA account sizes have fallen from millions to ~$100K-$250K "
            "thanks to fractional-share trading and platform automation. This "
            "team's growth story.",
        ],
    ),
}


def get_profile(segment: Segment | str) -> SegmentProfile:
    if isinstance(segment, str):
        segment = Segment(segment)
    return SEGMENT_PROFILES[segment]
