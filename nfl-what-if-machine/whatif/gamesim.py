"""Monte Carlo game engine.

The possession in progress is finished play by play (so down, distance and
clock matter right after an edited play); every later possession is a
resampled real drive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .models import GameModel, punt_zone
from .state import (HALF_SECONDS, KICKOFF, PAT, SCRIMMAGE, GameState, game_mode, modified_ot,
                    ot_length)

MAX_STEPS = 600
KNEEL_SECS = 40


class GameOver(Exception):
    def __init__(self, winner: str | None):
        self.winner = winner  # None = tie


@dataclass
class SimResult:
    winner: str | None
    score: dict[str, int]
    path: list[tuple] = field(default_factory=list)  # (elapsed minutes, situation) at each step


class Engine:
    def __init__(self, model: GameModel, rng: random.Random):
        self.m = model
        self.rng = rng

    # ---- bookkeeping ------------------------------------------------------------
    def run_clock(self, s: GameState, secs: float, two_minute_warning: bool = True) -> None:
        if two_minute_warning and s.half <= 2 and s.secs > 120 and s.secs - secs < 120:
            s.secs = 120.0
        else:
            s.secs = max(0.0, s.secs - secs)

    def leader(self, s: GameState) -> str | None:
        a, b = s.offense, s.defense
        if s.score[a] == s.score[b]:
            return None
        return a if s.score[a] > s.score[b] else b

    def score(self, s: GameState, team: str, pts: int, kind: str) -> None:
        s.score[team] += pts
        if s.half <= 2:
            return
        if not modified_ot(s.season, s.playoff):
            raise GameOver(self.leader(s))
        if kind == "fg" and s.ot_stage == 0:
            s.ot_stage = 1
        elif kind == "fg" and s.ot_stage == 1 and self.leader(s) is None:
            s.ot_stage = 2
        else:
            raise GameOver(self.leader(s))

    def change_possession(self, s: GameState, new_yl: float) -> None:
        """Non-scoring change of possession (turnover, punt, downs, missed FG)."""
        if s.half > 2 and modified_ot(s.season, s.playoff):
            if s.ot_stage == 0:
                s.ot_stage = 2
            elif s.ot_stage == 1:
                raise GameOver(self.leader(s))
        s.swap()
        s.phase, s.mid_drive, s.forced_call = SCRIMMAGE, False, None
        s.yardline_100 = min(99, max(1, int(new_yl)))
        s.down, s.ydstogo = 1, min(10, s.yardline_100)

    def touchdown(self, s: GameState, team: str) -> None:
        if team != s.offense:
            s.swap()
        s.mid_drive, s.forced_call = False, None
        self.score(s, team, 6, "td")
        s.phase = PAT

    def safety(self, s: GameState) -> None:
        s.mid_drive, s.forced_call = False, None
        self.score(s, s.defense, 2, "safety")
        s.phase, s.free_kick = KICKOFF, True

    # ---- play results (also used to apply an edit) -------------------------------
    def gain(self, s: GameState, yards: int) -> None:
        new_yl = s.yardline_100 - yards
        if new_yl <= 0:
            self.touchdown(s, s.offense)
        elif new_yl >= 100:
            self.safety(s)
        elif yards >= s.ydstogo:
            s.yardline_100, s.down, s.ydstogo = new_yl, 1, min(10, new_yl)
        elif s.down == 4:
            self.change_possession(s, 100 - new_yl)
        else:
            s.yardline_100, s.down, s.ydstogo = new_yl, s.down + 1, s.ydstogo - yards

    def turnover(self, s: GameState, net: int = 0, defense_td: bool = False) -> None:
        """Interception / lost fumble; the defense takes over `net` yards past the line."""
        if defense_td:
            self.touchdown(s, s.defense)
            return
        new_yl = 100 - (s.yardline_100 - net)
        if new_yl >= 100:
            new_yl = 80  # touchback
        if new_yl <= 0:
            self.touchdown(s, s.defense)
            return
        self.change_possession(s, new_yl)

    def field_goal(self, s: GameState, made: bool) -> None:
        if made:
            s.mid_drive, s.forced_call = False, None
            self.score(s, s.offense, 3, "fg")
            s.phase = KICKOFF
        else:
            self.change_possession(s, 100 - max(20, s.yardline_100 + 7))

    def punt(self, s: GameState, net: int, return_td: bool = False) -> None:
        if return_td:
            self.change_possession(s, 50)
            self.touchdown(s, s.offense)
            return
        self.change_possession(s, 100 - (s.yardline_100 - net))

    def penalty(self, s: GameState, on_offense: bool, yards: int, automatic_first_down: bool = False) -> None:
        """Accepted penalty; the down is replayed unless it produces a first down."""
        yl = s.yardline_100
        if on_offense:
            own = 100 - yl
            moved = yards if yards < own / 2 else own // 2
            s.yardline_100 = yl + moved
            s.ydstogo += moved
        else:
            moved = yards if yards < yl / 2 else max(0, yl // 2)
            new_yl = max(1, yl - moved)
            gained = yl - new_yl
            s.yardline_100 = new_yl
            if automatic_first_down or gained >= s.ydstogo:
                s.down, s.ydstogo = 1, min(10, new_yl)
            else:
                s.ydstogo -= gained
        s.ydstogo = max(1, min(s.ydstogo, s.yardline_100))

    # ---- decisions --------------------------------------------------------------
    def should_kneel(self, s: GameState) -> bool:
        if s.half != 2 or s.diff <= 0 or s.yardline_100 >= 98:
            return False
        kneels = 5 - s.down
        return s.secs <= KNEEL_SECS * max(0, kneels - s.timeouts[s.defense]) + 2

    def fourth_down(self, s: GameState) -> str:
        yl, togo, diff, secs = s.yardline_100, s.ydstogo, s.diff, s.secs
        fg_range = yl <= 37
        late = s.half == 2 and secs <= 300
        if s.half == 2 and diff < -8 and secs <= 600:
            return "go"
        if late and diff < 0:
            return "fg" if -diff <= 3 and fg_range else "go"
        if fg_range:
            return "go" if togo <= 1 and yl <= 2 and diff < 0 else "fg"
        if (yl <= 40 and togo <= 3) or (togo <= 1 and yl <= 55):
            return "go"
        return "punt"

    # ---- main loop --------------------------------------------------------------
    def play_out(self, s: GameState, record: bool = False) -> SimResult:
        path: list[tuple] = []
        try:
            for _ in range(MAX_STEPS):
                if record:
                    path.append((s.elapsed_minutes, self.situation(s)))
                self.step(s)
            winner = self.leader(s)
        except GameOver as end:
            winner = end.winner
        if record:
            path.append((s.elapsed_minutes, ("END", winner)))
        return SimResult(winner, dict(s.score), path)

    @staticmethod
    def situation(s: GameState) -> tuple:
        """Coarse situation used to estimate win probability along simulated paths."""
        home_ball = s.offense == s.home
        if s.phase == SCRIMMAGE:
            where = "H" if home_ball else "A"
            fp = 0 if s.yardline_100 <= 20 else 1 if s.yardline_100 <= 50 else 2 if s.yardline_100 <= 80 else 3
        else:
            where = ("K" if s.phase == KICKOFF else "P") + ("H" if home_ball else "A")
            fp = 0
        return (s.score[s.home] - s.score[s.away], where, fp)

    def step(self, s: GameState) -> None:
        if s.phase == PAT:
            self.try_after_td(s)
        elif s.secs <= 0:
            self.end_period(s)
        elif s.phase == KICKOFF:
            self.kickoff(s)
        elif s.mid_drive:
            self.play(s)
        else:
            self.drive(s)

    def end_period(self, s: GameState) -> None:
        a, b = s.offense, s.defense
        s.mid_drive, s.forced_call = False, None
        if s.half == 1:
            recv = s.second_half_receiver or self.rng.choice([a, b])
            s.offense, s.defense = (b, a) if recv == a else (a, b)
            s.phase, s.free_kick = KICKOFF, False
            s.half, s.secs = 2, HALF_SECONDS
            s.timeouts = {a: 3, b: 3}
        elif s.half == 2:
            if self.leader(s) is not None:
                raise GameOver(self.leader(s))
            recv = self.rng.choice([a, b])  # coin toss
            s.offense, s.defense = (b, a) if recv == a else (a, b)
            s.phase, s.free_kick = KICKOFF, False
            s.half, s.secs, s.ot_stage = 3, ot_length(s.season, s.playoff), 0
            s.timeouts = {a: 3 if s.playoff else 2, b: 3 if s.playoff else 2}
        else:
            if self.leader(s) is not None or not s.playoff:
                raise GameOver(self.leader(s))  # regular-season OT can end tied
            s.half, s.secs = s.half + 1, ot_length(s.season, s.playoff)

    def try_after_td(self, s: GameState) -> None:
        go_for_two = s.half == 2 and s.secs <= 900 and s.diff in (-10, -5, -2, 1, 5, 12)
        if go_for_two:
            if self.rng.random() < self.m.two_pt_rate_for(s.season):
                s.score[s.offense] += 2
        elif self.rng.random() < self.m.pat_rate_for(s.season):
            s.score[s.offense] += 1
        s.phase, s.free_kick = KICKOFF, False

    def kickoff(self, s: GameState) -> None:
        onside = s.half == 2 and s.diff < 0 and (s.secs <= 150 or (s.diff < -8 and s.secs <= 300))
        if onside and not s.free_kick:
            self.run_clock(s, 4)
            if self.rng.random() < self.m.onside_recovery:
                s.phase, s.yardline_100, s.down, s.ydstogo, s.mid_drive = SCRIMMAGE, 55, 1, 10, False
            else:
                self.change_possession(s, 45)
            return
        yl = self.rng.choice(self.m.kickoffs)
        if s.free_kick:
            yl, s.free_kick = max(1, yl - 15), False
        s.swap()
        s.phase, s.mid_drive, s.forced_call = SCRIMMAGE, False, None
        s.yardline_100, s.down, s.ydstogo = int(yl), 1, 10

    def drive(self, s: GameState) -> None:
        """Resample a whole real drive from a similar situation."""
        pool = self.m.drive_pool(s.half, s.secs, s.diff, s.yardline_100)
        for _ in range(50):
            result, secs, next_yl = self.rng.choice(pool)
            # Keep drives that fit the time left: a scoring or punting drive must
            # finish before the half does, and "ran out the half" must be plausible.
            if (result == "END" and secs >= s.secs - 20) or (result != "END" and secs <= s.secs + 3):
                break
        else:
            result = "END"
        if result == "END":
            s.secs = 0.0
            return
        self.run_clock(s, secs, two_minute_warning=False)
        if result == "TD":
            self.touchdown(s, s.offense)
        elif result == "FG":
            self.field_goal(s, True)
        elif result == "OPP_TD":
            self.touchdown(s, s.defense)
        elif result == "SAFETY":
            self.safety(s)
        else:
            self.change_possession(s, next_yl)

    def play(self, s: GameState) -> None:
        """One snap of the possession in progress."""
        off, dfn = s.offense, s.defense
        call, s.forced_call = s.forced_call, None
        if call is None and self.should_kneel(s):
            if s.timeouts[dfn] > 0:
                s.timeouts[dfn] -= 1
                self.run_clock(s, 2)
            else:
                self.run_clock(s, KNEEL_SECS)
            self.gain(s, -1)
            return

        fg_range = s.yardline_100 <= 37
        mode = game_mode(s.half, s.secs, s.diff)
        if call is None:
            end_of_half_kick = fg_range and s.secs <= 8 and (s.half != 2 or -3 <= s.diff <= 0)
            decision = "fg" if end_of_half_kick else (self.fourth_down(s) if s.down == 4 else "go")
            if decision == "fg":
                self.run_clock(s, 5)
                self.field_goal(s, self.rng.random() < self.m.fg_prob(s.yardline_100))
                return
            if decision == "punt":
                net, td, secs = self.rng.choice(self.m.punts[punt_zone(s.yardline_100)])
                self.run_clock(s, secs)
                self.punt(s, net, bool(td))
                return

        pool = self.m.play_pool(s.down, s.ydstogo, s.yardline_100, mode, call)
        yards, turnover, net, def_td, stops = self.rng.choice(pool)
        durations = self.m.durations.get((mode, stops)) or self.m.durations[("normal", stops)]
        secs = self.rng.choice(durations)
        if not stops and not turnover:
            if mode == "hurry" and s.timeouts[off] > 0:
                s.timeouts[off] -= 1
                secs = min(secs, 6)
            elif mode == "protect" and s.timeouts[dfn] > 0 and s.secs <= 240:
                s.timeouts[dfn] -= 1
                secs = min(secs, 6)
        self.run_clock(s, secs)
        if turnover:
            self.turnover(s, net, bool(def_td))
        else:
            self.gain(s, int(yards))
