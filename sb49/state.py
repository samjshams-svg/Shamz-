"""Game state used by the simulator."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

# Phases: what happens on the next snap.
SCRIMMAGE = "scrimmage"   # offense has the ball: down / distance / yardline apply
KICKOFF = "kickoff"       # `offense` is the KICKING team
PAT = "pat"               # `offense` just scored a TD and tries the conversion

HALF_SECONDS = 1800
OT_SECONDS = 900


@dataclass
class GameState:
    offense: str
    defense: str
    phase: str = SCRIMMAGE
    yardline_100: int = 75      # yards from the offense's goal line
    down: int = 1
    ydstogo: int = 10
    half: int = 1               # 1, 2 = regulation halves; 3+ = overtime periods
    secs: float = HALF_SECONDS  # seconds left in the current half / OT period
    score: dict[str, int] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=dict)
    second_half_receiver: str = ""
    free_kick: bool = False     # kickoff after a safety (from the 20)
    # Overtime (2014 playoff rules). ot_stage: 0 = first possession,
    # 1 = other team answering a first-possession FG, 2 = sudden death.
    ot_stage: int = 0

    def copy(self) -> "GameState":
        return copy.deepcopy(self)

    @property
    def diff(self) -> int:
        """Offense score minus defense score."""
        return self.score[self.offense] - self.score[self.defense]

    @property
    def game_secs(self) -> float:
        """Seconds left in regulation (0 during overtime)."""
        if self.half == 1:
            return self.secs + HALF_SECONDS
        if self.half == 2:
            return self.secs
        return 0.0

    def swap(self) -> None:
        self.offense, self.defense = self.defense, self.offense

    def spot(self) -> str:
        yl = self.yardline_100
        if yl == 50:
            return "50"
        return f"{self.defense} {yl}" if yl < 50 else f"{self.offense} {100 - yl}"

    def describe(self) -> str:
        a, b = self.offense, self.defense
        score = f"{a} {self.score[a]} - {b} {self.score[b]}"
        clock = f"{'OT' if self.half > 2 else 'H' + str(self.half)} {int(self.secs) // 60}:{int(self.secs) % 60:02d}"
        if self.phase == KICKOFF:
            where = f"{a} kicking off"
        elif self.phase == PAT:
            where = f"{a} attempting the try after TD"
        else:
            where = f"{a} ball, {self.down}&{self.ydstogo} at {self.spot()}"
        tos = f"TO {a}:{self.timeouts[a]} {b}:{self.timeouts[b]}"
        return f"{score} | {clock} | {where} | {tos}"


def game_mode(half: int, secs: float, diff: int) -> str:
    """Offensive tempo bucket shared by the calibration data and the simulator."""
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
