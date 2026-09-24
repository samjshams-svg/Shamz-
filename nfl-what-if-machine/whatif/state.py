"""Game state shared by the play-level and drive-level simulators."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

SCRIMMAGE = "scrimmage"   # offense has the ball
KICKOFF = "kickoff"       # `offense` is the KICKING team
PAT = "pat"               # `offense` just scored a TD and tries the conversion

HALF_SECONDS = 1800


def ot_length(season: int, playoff: bool) -> int:
    """Overtime period length: 10 min in the regular season from 2017, else 15."""
    return 600 if (season >= 2017 and not playoff) else 900


def modified_ot(season: int, playoff: bool) -> bool:
    """Possession rules (a first-drive FG doesn't end it): playoffs from 2010, all games from 2012."""
    return playoff or season >= 2012


@dataclass
class GameState:
    offense: str
    defense: str
    home: str = ""
    away: str = ""
    season: int = 2014
    playoff: bool = False
    phase: str = SCRIMMAGE
    yardline_100: int = 75      # yards from the offense's goal line
    down: int = 1
    ydstogo: int = 10
    half: int = 1               # 1, 2 = regulation halves; 3+ = overtime periods
    secs: float = HALF_SECONDS  # seconds left in the current half / OT period
    score: dict[str, int] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=dict)
    second_half_receiver: str = ""
    free_kick: bool = False     # kickoff after a safety
    mid_drive: bool = True      # True: finish this possession play by play
    forced_call: str | None = None  # "run" / "pass": next snap must be this play type
    ot_stage: int = 0           # 0 first possession, 1 answering a FG, 2 sudden death

    def copy(self) -> "GameState":
        return copy.deepcopy(self)

    @property
    def diff(self) -> int:
        return self.score[self.offense] - self.score[self.defense]

    @property
    def game_secs(self) -> float:
        """Seconds left in regulation (0 in overtime)."""
        if self.half == 1:
            return self.secs + HALF_SECONDS
        return self.secs if self.half == 2 else 0.0

    @property
    def elapsed_minutes(self) -> float:
        if self.half <= 2:
            return 60 - self.game_secs / 60
        return 60 + (self.half - 3) * ot_length(self.season, self.playoff) / 60 + \
            (ot_length(self.season, self.playoff) - self.secs) / 60

    @property
    def qtr(self) -> int:
        if self.half > 2:
            return self.half + 2
        return (self.half - 1) * 2 + (1 if self.secs > 900 else 2)

    @property
    def quarter_secs(self) -> float:
        if self.half > 2:
            return self.secs
        return self.secs - 900 if self.secs > 900 else self.secs

    def swap(self) -> None:
        self.offense, self.defense = self.defense, self.offense

    def spot(self) -> str:
        yl = self.yardline_100
        if yl == 50:
            return "50"
        return f"{self.defense} {yl}" if yl < 50 else f"{self.offense} {100 - yl}"

    def clock(self) -> str:
        q = self.qtr
        name = f"Q{q}" if q <= 4 else ("OT" if q == 5 else f"OT{q - 4}")
        s = int(round(self.quarter_secs))
        return f"{name} {s // 60}:{s % 60:02d}"

    def describe(self) -> str:
        a, b = self.offense, self.defense
        score = f"{a} {self.score[a]} - {b} {self.score[b]}"
        if self.phase == KICKOFF:
            where = f"{a} kicking off"
        elif self.phase == PAT:
            where = f"{a} try after touchdown"
        else:
            where = f"{a} ball, {self.down}&{self.ydstogo} at {self.spot()}"
            if self.forced_call:
                where += f" (next play: {self.forced_call})"
        tos = f"timeouts {a}:{self.timeouts[a]} {b}:{self.timeouts[b]}"
        return f"{score} | {self.clock()} | {where} | {tos}"


def game_mode(half: int, secs: float, diff: int) -> str:
    """Offensive tempo bucket shared by calibration and simulation."""
    if half == 1:
        return "hurry" if secs <= 120 else "normal"
    if half == 2:
        if diff < 0 and (secs <= 120 or (secs <= 300 and diff >= -24)):
            return "hurry"
        if diff == 0 and secs <= 120:
            return "hurry"
        if diff > 0 and secs <= 300:
            return "protect"
    return "normal"
