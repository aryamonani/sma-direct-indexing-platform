"""
Client and Book: the multi-account layer.

A Client bundles a Portfolio with its Segment profile, tax rates, and optimizer
priorities. The Book is the book of all clients; the dashboard operates on the
Book, and drill-ins operate on individual Clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from .portfolio import Portfolio
from .screens import ScreenSet
from .segments import OptimizerPriorities, Segment, SegmentProfile, get_profile
from .tax import TaxRates


@dataclass
class Client:
    client_id: str
    display_name: str
    segment: Segment
    aum: float
    screens: ScreenSet
    tax_rates: TaxRates
    priorities: OptimizerPriorities
    notes: list[str] = field(default_factory=list)
    portfolio: Optional[Portfolio] = None

    @classmethod
    def from_segment(
        cls,
        client_id: str,
        display_name: str,
        segment: Segment | str,
        aum: float,
        extra_screens: Optional[ScreenSet] = None,
        tax_overrides: Optional[dict] = None,
        notes: Optional[list[str]] = None,
    ) -> "Client":
        """Create a client from a segment profile, with optional overrides."""
        if isinstance(segment, str):
            segment = Segment(segment)
        profile = get_profile(segment)

        # Merge default + extra screens
        if extra_screens is None:
            screens = ScreenSet(
                excluded_sectors=set(profile.default_screens.excluded_sectors),
                excluded_sub_industries=set(profile.default_screens.excluded_sub_industries),
                excluded_tickers=set(profile.default_screens.excluded_tickers),
                thematic_screens=set(profile.default_screens.thematic_screens),
                max_single_weight=profile.default_screens.max_single_weight,
            )
        else:
            screens = ScreenSet(
                excluded_sectors=profile.default_screens.excluded_sectors | extra_screens.excluded_sectors,
                excluded_sub_industries=profile.default_screens.excluded_sub_industries | extra_screens.excluded_sub_industries,
                excluded_tickers=profile.default_screens.excluded_tickers | extra_screens.excluded_tickers,
                thematic_screens=profile.default_screens.thematic_screens | extra_screens.thematic_screens,
                max_single_weight=extra_screens.max_single_weight or profile.default_screens.max_single_weight,
            )

        tax = profile.default_tax_rates
        if tax_overrides:
            tax = TaxRates(
                short_term=tax_overrides.get("short_term", tax.short_term),
                long_term=tax_overrides.get("long_term", tax.long_term),
                state=tax_overrides.get("state", tax.state),
                niit=tax_overrides.get("niit", tax.niit),
            )

        return cls(
            client_id=client_id,
            display_name=display_name,
            segment=segment,
            aum=aum,
            screens=screens,
            tax_rates=tax,
            priorities=profile.default_priorities,
            notes=notes or [],
        )


@dataclass
class Book:
    """The full book of client accounts the platform manages."""
    clients: list[Client] = field(default_factory=list)

    def __iter__(self):
        return iter(self.clients)

    def __len__(self):
        return len(self.clients)

    def by_id(self, client_id: str) -> Optional[Client]:
        for c in self.clients:
            if c.client_id == client_id:
                return c
        return None

    def by_segment(self, segment: Segment | str) -> list[Client]:
        if isinstance(segment, str):
            segment = Segment(segment)
        return [c for c in self.clients if c.segment == segment]

    def total_aum(self) -> float:
        return sum(c.aum for c in self.clients)


# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------

def load_book(sample_dir: Path) -> Book:
    """Load all client YAML files under sample_clients/ into a Book."""
    book = Book()
    for path in sorted(sample_dir.glob("*.yaml")):
        with open(path) as f:
            spec = yaml.safe_load(f)
        screens = ScreenSet.from_dict(spec.get("screens", {}))
        client = Client.from_segment(
            client_id=spec["client_id"],
            display_name=spec["display_name"],
            segment=spec["segment"],
            aum=spec["aum"],
            extra_screens=screens,
            tax_overrides=spec.get("tax_overrides"),
            notes=spec.get("notes", []),
        )
        book.clients.append(client)
    return book
