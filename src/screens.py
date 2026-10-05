"""
Client screens: the rules that filter the investable universe for an account.

A screen is a predicate on a ticker. We combine screens with AND: a ticker
passes the client's full screen set if every screen accepts it.

Thematic screens (fossil-fuel-free, tobacco-free, etc.) are implemented as
sector / sub-industry / explicit-ticker lookups. In production, these would
be driven by licensed ESG classification data (MSCI, Sustainalytics). We note
that gap explicitly in the UI so a reviewer sees it as a deliberate scoping
choice, not a blind spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .data import SP500_UNIVERSE


# Thematic exclusion maps. These approximate what licensed ESG providers would
# flag at a higher resolution.
THEMATIC_EXCLUSIONS: dict[str, dict] = {
    "fossil_fuels": {
        "label": "Fossil-fuel-free",
        "description": "Excludes integrated oil & gas and upstream E&P names.",
        "sectors": {"Energy"},
        "sub_industries": set(),
        "tickers": set(),
    },
    "tobacco": {
        "label": "Tobacco-free",
        "description": "Excludes tobacco producers.",
        "sectors": set(),
        "sub_industries": {"Tobacco"},
        "tickers": set(),
    },
    "weapons": {
        "label": "Weapons-free",
        "description": "Excludes aerospace & defense names.",
        "sectors": set(),
        "sub_industries": {"Aerospace & Defense"},
        "tickers": set(),
    },
    "banks": {
        "label": "Bank-free",
        "description": "Excludes diversified banks and investment banks.",
        "sectors": set(),
        "sub_industries": {"Diversified Banks", "Investment Banking"},
        "tickers": set(),
    },
}


@dataclass
class ScreenSet:
    """
    The full set of exclusion rules for one client account.
    """
    excluded_sectors: set[str] = field(default_factory=set)
    excluded_sub_industries: set[str] = field(default_factory=set)
    excluded_tickers: set[str] = field(default_factory=set)
    thematic_screens: set[str] = field(default_factory=set)
    # Max weight per security (concentration cap). None = no cap.
    max_single_weight: float | None = None

    def is_allowed(self, ticker: str) -> bool:
        """Return True if `ticker` passes all screens."""
        if ticker not in SP500_UNIVERSE:
            return False
        meta = SP500_UNIVERSE[ticker]

        if ticker in self.excluded_tickers:
            return False
        if meta["sector"] in self.excluded_sectors:
            return False
        if meta["sub_industry"] in self.excluded_sub_industries:
            return False

        for theme in self.thematic_screens:
            rule = THEMATIC_EXCLUSIONS.get(theme)
            if rule is None:
                continue
            if ticker in rule["tickers"]:
                return False
            if meta["sector"] in rule["sectors"]:
                return False
            if meta["sub_industry"] in rule["sub_industries"]:
                return False
        return True

    def allowed_universe(self) -> list[str]:
        return [t for t in SP500_UNIVERSE if self.is_allowed(t)]

    def description(self) -> list[str]:
        """Human-readable description of each active screen."""
        parts = []
        if self.excluded_tickers:
            parts.append(f"Exclude tickers: {', '.join(sorted(self.excluded_tickers))}")
        if self.excluded_sectors:
            parts.append(f"Exclude sectors: {', '.join(sorted(self.excluded_sectors))}")
        if self.excluded_sub_industries:
            parts.append(f"Exclude sub-industries: {', '.join(sorted(self.excluded_sub_industries))}")
        for t in sorted(self.thematic_screens):
            if t in THEMATIC_EXCLUSIONS:
                parts.append(f"Theme: {THEMATIC_EXCLUSIONS[t]['label']}")
        if self.max_single_weight is not None:
            parts.append(f"Max single-security weight: {self.max_single_weight:.1%}")
        return parts or ["No screens applied (pure index replication)"]

    def to_dict(self) -> dict:
        return {
            "excluded_sectors": sorted(self.excluded_sectors),
            "excluded_sub_industries": sorted(self.excluded_sub_industries),
            "excluded_tickers": sorted(self.excluded_tickers),
            "thematic_screens": sorted(self.thematic_screens),
            "max_single_weight": self.max_single_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenSet":
        return cls(
            excluded_sectors=set(d.get("excluded_sectors", []) or []),
            excluded_sub_industries=set(d.get("excluded_sub_industries", []) or []),
            excluded_tickers=set(d.get("excluded_tickers", []) or []),
            thematic_screens=set(d.get("thematic_screens", []) or []),
            max_single_weight=d.get("max_single_weight"),
        )
