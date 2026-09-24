"""League-wide play outcome model calibrated on a season of nflfastR data.

The simulator does not use a parametric play model. Instead it resamples real
plays from the calibration season that happened in a similar situation
(down, distance, field zone, game tempo), and resamples clock run-off from
real plays with the same tempo and clock-stopping status. Special teams use
empirical kickoff and punt results and a logistic field-goal curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state import game_mode

SNAP_TYPES = {"run", "pass", "punt", "field_goal", "kickoff", "extra_point", "qb_kneel", "qb_spike"}
MIN_POOL = 40


def dist_bucket(ydstogo: float) -> int:
    if ydstogo <= 1:
        return 0
    if ydstogo <= 3:
        return 1
    if ydstogo <= 6:
        return 2
    if ydstogo <= 10:
        return 3
    return 4


def zone_bucket(yardline_100: float) -> int:
    if yardline_100 <= 5:
        return 0
    if yardline_100 <= 20:
        return 1
    if yardline_100 <= 60:
        return 2
    if yardline_100 <= 90:
        return 3
    return 4


def punt_zone(yardline_100: float) -> int:
    if yardline_100 <= 45:
        return 0
    if yardline_100 <= 60:
        return 1
    if yardline_100 <= 75:
        return 2
    return 3


def _half(qtr: float) -> int:
    return 1 if qtr <= 2 else (2 if qtr <= 4 else 3)


@dataclass
class LeagueModel:
    # key (down, dist, zone, mode) -> list of (yards, turnover, turnover_net, def_td, stops_clock)
    scrimmage: dict[tuple, list] = field(default_factory=dict)
    # key (mode, stops_clock) -> list of seconds
    durations: dict[tuple, list] = field(default_factory=dict)
    # list of (receiver_yardline_100, return_td, seconds)
    kickoffs: list = field(default_factory=list)
    onside_recovery: float = 0.2
    # punt zone -> list of (net, return_td, seconds)
    punts: dict[int, list] = field(default_factory=dict)
    fg_coef: tuple[float, float] = (6.0, -0.11)
    pat_rate: float = 0.993
    two_pt_rate: float = 0.48
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- sampling helpers used by the simulator -------------------------------
    def scrimmage_pool(self, down: int, ydstogo: int, yardline_100: int, mode: str) -> list:
        key = (min(down, 3), dist_bucket(ydstogo), zone_bucket(yardline_100), mode)
        pool = self._cache.get(key)
        if pool is None:
            d, t, z, m = key
            best: list = []
            for k in [(d, t, z, m), (d, t, z, None), (None, t, z, None), (d, t, None, None), (None, t, None, None)]:
                pool = self.scrimmage.get(k) or []
                if len(pool) >= MIN_POOL:
                    break
                best = max(best, pool, key=len)
            else:
                pool = best
            if not pool:
                raise ValueError(f"no calibration plays for situation {key}")
            self._cache[key] = pool
        return pool

    def fg_prob(self, yardline_100: int) -> float:
        a, b = self.fg_coef
        dist = yardline_100 + 17
        return 1.0 / (1.0 + math.exp(-(a + b * dist)))

    # ---- calibration -----------------------------------------------------------
    @classmethod
    def from_pbp(cls, pbp: pd.DataFrame, exclude_game: str | None = None) -> "LeagueModel":
        df = pbp
        if exclude_game:
            df = df[df["game_id"] != exclude_game]
        df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
        model = cls()
        snaps = _annotate_snaps(df)
        model._fit_scrimmage(snaps)
        model._fit_kickoffs(snaps)
        model._fit_punts(snaps)
        model._fit_kicks(df)
        return model

    def _fit_scrimmage(self, snaps: pd.DataFrame) -> None:
        plays = snaps[
            snaps["play_type"].isin(["run", "pass"])
            & snaps["down"].notna()
            & (snaps["two_point_attempt"].fillna(0) == 0)
            & (snaps["qb_spike"].fillna(0) == 0)
            & snaps["yardline_100"].notna()
        ]
        for r in plays.itertuples(index=False):
            half = _half(r.qtr)
            mode = game_mode(half, r.half_seconds_remaining, r.score_differential)
            turnover = int((r.interception or 0) == 1 or (r.fumble_lost or 0) == 1)
            yards = 0 if math.isnan(r.yards_gained) else int(r.yards_gained)
            if r.interception == 1:
                net = (0 if math.isnan(r.air_yards) else r.air_yards) - (0 if math.isnan(r.return_yards) else r.return_yards)
            else:
                net = yards
            def_td = int(turnover and r.return_touchdown == 1)
            stops = int(
                r.incomplete_pass == 1 or r.out_of_bounds == 1 or turnover == 1
                or yards >= r.yardline_100
            )
            outcome = (yards, turnover, int(net), def_td, stops)
            d, t, z = min(int(r.down), 3), dist_bucket(r.ydstogo), zone_bucket(r.yardline_100)
            for key in [(d, t, z, mode), (d, t, z, None), (None, t, z, None), (d, t, None, None), (None, t, None, None)]:
                self.scrimmage.setdefault(key, []).append(outcome)
            if r.clean_duration:
                self.durations.setdefault((mode, stops), []).append(float(r.duration))

    def _fit_kickoffs(self, snaps: pd.DataFrame) -> None:
        kicks = snaps[snaps["play_type"] == "kickoff"]
        onside_tries = onside_recovered = 0
        for r in kicks.itertuples(index=False):
            onside = (r.kick_distance or 99) <= 20
            if onside:
                onside_tries += 1
                onside_recovered += int(r.next_posteam == r.defteam)
                continue
            if r.return_touchdown == 1 and r.td_team == r.posteam:
                self.kickoffs.append((0, 1, max(r.duration, 8.0)))
            elif r.next_posteam == r.posteam and not math.isnan(r.next_yardline):
                self.kickoffs.append((int(r.next_yardline), 0, r.duration))
        if onside_tries >= 10:
            self.onside_recovery = onside_recovered / onside_tries

    def _fit_punts(self, snaps: pd.DataFrame) -> None:
        punts = snaps[(snaps["play_type"] == "punt") & snaps["yardline_100"].notna()]
        for r in punts.itertuples(index=False):
            zone = punt_zone(r.yardline_100)
            if r.return_touchdown == 1 and r.td_team == r.defteam:
                self.punts.setdefault(zone, []).append((0, 1, 12.0))
            elif r.next_posteam == r.defteam and not math.isnan(r.next_yardline):
                net = r.yardline_100 - (100 - r.next_yardline)
                self.punts.setdefault(zone, []).append((int(net), 0, r.duration))

    def _fit_kicks(self, df: pd.DataFrame) -> None:
        fg = df[(df["play_type"] == "field_goal") & df["kick_distance"].notna()]
        x = fg["kick_distance"].to_numpy(float)
        y = (fg["field_goal_result"] == "made").to_numpy(float)
        self.fg_coef = _logistic_fit(x, y)
        xp = df[df["extra_point_result"].notna()]
        if len(xp):
            self.pat_rate = float((xp["extra_point_result"] == "good").mean())
        tp = df[df["two_point_conv_result"].notna()]
        if len(tp):
            self.two_pt_rate = float((tp["two_point_conv_result"] == "success").mean())


def _annotate_snaps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real snaps and attach clock run-off + the next snap's possession."""
    df = df.copy()
    df["is_timeout"] = df["timeout"].fillna(0) == 1
    # Timeouts called between a snap and the next one truncate the run-off.
    df["to_group"] = df["play_type"].isin(SNAP_TYPES).groupby(df["game_id"]).cumsum()
    timeouts_after = df[df["is_timeout"]].groupby(["game_id", "to_group"]).size()

    snaps = df[df["play_type"].isin(SNAP_TYPES)].copy()
    g = snaps.groupby("game_id")
    snaps["next_gsr"] = g["game_seconds_remaining"].shift(-1)
    snaps["next_half_secs"] = g["half_seconds_remaining"].shift(-1)
    snaps["next_qtr"] = g["qtr"].shift(-1)
    snaps["next_posteam"] = g["posteam"].shift(-1)
    snaps["next_yardline"] = g["yardline_100"].shift(-1)
    snaps["duration"] = (snaps["game_seconds_remaining"] - snaps["next_gsr"]).clip(0, 45)

    had_timeout = pd.Series(
        [(gid, grp) in timeouts_after.index for gid, grp in zip(snaps["game_id"], snaps["to_group"])],
        index=snaps.index,
    )
    same_half = snaps["qtr"].map(_half) == snaps["next_qtr"].map(lambda q: _half(q) if q == q else -1)
    two_min_stop = (snaps["half_seconds_remaining"] > 120) & (snaps["next_half_secs"] == 120)
    snaps["clean_duration"] = (
        snaps["next_gsr"].notna() & same_half & ~had_timeout & ~two_min_stop
    )
    snaps["duration"] = snaps["duration"].fillna(5.0)
    return snaps


def _logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Newton-Raphson fit of P(y=1) = sigmoid(a + b x)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([5.0, -0.1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (y - p)
        hess = -(X.T * (p * (1 - p))) @ X
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])
