"""Franchise codes, divisions and conferences (2002 alignment, unchanged through 2021)."""

from __future__ import annotations

# Schedules use the code a team had that season; play-by-play uses today's.
# Everything in this package uses today's franchise code.
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

DIVISIONS: dict[str, list[str]] = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}

TEAMS: list[str] = sorted(t for teams in DIVISIONS.values() for t in teams)
TEAM_INDEX = {t: i for i, t in enumerate(TEAMS)}
DIVISION_OF = {t: d for d, teams in DIVISIONS.items() for t in teams}
CONFERENCE_OF = {t: d.split()[0] for t, d in DIVISION_OF.items()}

# Names used for display and for matching free-text team input.
NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Washington",
}


def franchise(code: str | None) -> str | None:
    if code is None:
        return None
    return FRANCHISE.get(code, code)


def resolve_team(text: str) -> str:
    """'SEA', 'sea', 'Seahawks', 'SD' -> franchise code."""
    t = text.strip()
    up = franchise(t.upper())
    if up in TEAM_INDEX:
        return up
    for code, name in NAMES.items():
        if name.lower() == t.lower():
            return code
    raise ValueError(f"unknown team {text!r}")
