"""Empirical game model built from 2010s play-by-play.

* Play level (used only to finish the drive in progress when the edit happens):
  resample a real play from the same down / distance / field zone / tempo, and
  a real clock run-off from the same tempo and clock-stopping status.
* Drive level (every later possession): resample a real drive that started
  in a similar spot: field position, and late in a half also time left and
  score margin. A drive gives a result (TD, FG, punt, turnover, ...), how long
  it took, and where the opponent's next drive started.
* Kicks: logistic FG curve on distance; PAT / two-point / onside rates by era.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .state import game_mode

MODEL_VERSION = 6
SNAP_TYPES = {"run", "pass", "punt", "field_goal", "kickoff", "extra_point", "qb_kneel", "qb_spike"}
MIN_PLAY_POOL = 40
MIN_DRIVE_POOL = 25

DRIVE_RESULTS = {
    "Touchdown": "TD", "Field goal": "FG", "Punt": "PUNT", "Turnover": "TO",
    "Turnover on downs": "DOWNS", "Missed field goal": "MISSED_FG", "End of half": "END",
    "Opp touchdown": "OPP_TD", "Safety": "SAFETY",
}
CHANGE_OF_POSSESSION = {"PUNT", "TO", "DOWNS", "MISSED_FG"}


# ---- buckets --------------------------------------------------------------------
def dist_bucket(ydstogo: float) -> int:
    return 0 if ydstogo <= 1 else 1 if ydstogo <= 3 else 2 if ydstogo <= 6 else 3 if ydstogo <= 10 else 4


def zone_bucket(yl: float) -> int:
    return 0 if yl <= 5 else 1 if yl <= 20 else 2 if yl <= 60 else 3 if yl <= 90 else 4


def punt_zone(yl: float) -> int:
    return 0 if yl <= 45 else 1 if yl <= 60 else 2 if yl <= 75 else 3


def fp_bucket(yl: float) -> int:
    return 0 if yl <= 20 else 1 if yl <= 40 else 2 if yl <= 60 else 3 if yl <= 70 else 4 if yl <= 80 else 5


def fp3_bucket(yl: float) -> int:
    return 0 if yl <= 40 else 1 if yl <= 70 else 2


def time_bucket(half_secs: float) -> int:
    for i, cut in enumerate((30, 60, 120, 240, 480)):
        if half_secs <= cut:
            return i
    return 5


def diff_bucket(diff: float) -> int:
    for i, cut in enumerate((-9, -4, -1, 0, 3, 8)):
        if diff <= cut:
            return i
    return 6


def drive_keys(half: int, half_secs: float, diff: float, yl: float) -> list[tuple]:
    """Most specific first. Overtime uses the late-second-half buckets."""
    tb = time_bucket(half_secs)
    if half == 1 and tb <= 2:
        return [("h1", tb, fp3_bucket(yl)), ("h1", tb), ("early", fp_bucket(yl))]
    if half >= 2 and tb <= 4:
        db = diff_bucket(diff)
        return [("h2", tb, db, fp3_bucket(yl)), ("h2", tb, db), ("h2", tb), ("early", fp_bucket(yl))]
    return [("early", fp_bucket(yl))]


def _half(qtr: float) -> int:
    return 1 if qtr <= 2 else (2 if qtr <= 4 else 3)


def default_window(season: int) -> tuple[int, int]:
    """Calibration seasons for a game: +/- 2 seasons, kept inside 2010-2019."""
    lo = max(data.FIRST_SEASON, min(season - 2, data.LAST_SEASON - 4))
    return lo, lo + 4



@dataclass
class GameModel:
    seasons: tuple[int, int]
    scrimmage: dict[tuple, list] = field(default_factory=dict)
    durations: dict[tuple, list] = field(default_factory=dict)
    punts: dict[int, list] = field(default_factory=dict)
    kickoffs: list = field(default_factory=list)
    drives: dict[tuple, list] = field(default_factory=dict)
    fg_coef: tuple[float, float] = (6.0, -0.11)
    pat_rate: float = 0.99
    two_pt_rate: float = 0.48
    onside_recovery: float = 0.2
    pat_by_season: dict[int, float] = field(default_factory=dict)
    two_pt_by_season: dict[int, float] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- sampling ---------------------------------------------------------------
    def play_pool(self, down: int, ydstogo: int, yl: int, mode: str, kind: str | None = None) -> list:
        key = (min(down, 3), dist_bucket(ydstogo), zone_bucket(yl), mode, kind)
        pool = self._cache.get(key)
        if pool is None:
            d, t, z, m, k = key
            best: list = []
            for cand in [(d, t, z, m, k), (d, t, z, None, k), (None, t, z, None, k),
                         (d, t, None, None, k), (None, t, None, None, k), (None, None, None, None, k)]:
                pool = self.scrimmage.get(cand) or []
                if len(pool) >= MIN_PLAY_POOL:
                    break
                best = max(best, pool, key=len)
            else:
                pool = best
            if not pool:
                raise ValueError(f"no calibration plays for {key}")
            self._cache[key] = pool
        return pool

    def drive_pool(self, half: int, half_secs: float, diff: float, yl: float) -> list:
        keys = drive_keys(half, half_secs, diff, yl)
        pool = self._cache.get(keys[0])
        if pool is None:
            for k in keys:
                pool = self.drives.get(k) or []
                if len(pool) >= MIN_DRIVE_POOL:
                    break
            self._cache[keys[0]] = pool
        return pool

    def pat_rate_for(self, season: int) -> float:
        return self.pat_by_season.get(season, self.pat_rate)

    def two_pt_rate_for(self, season: int) -> float:
        return self.two_pt_by_season.get(season, self.two_pt_rate)

    def fg_prob(self, yl: float) -> float:
        a, b = self.fg_coef
        return 1.0 / (1.0 + math.exp(-(a + b * (yl + 17))))

    # ---- calibration ------------------------------------------------------------
    @classmethod
    def fit(cls, pbp: pd.DataFrame, seasons: tuple[int, int]) -> "GameModel":
        df = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
        m = cls(seasons=seasons)
        snaps = _annotate_snaps(df)
        m._fit_plays(snaps)
        m._fit_special(snaps, df)
        m._fit_drives(df)
        return m

    def _fit_plays(self, snaps: pd.DataFrame) -> None:
        plays = snaps[snaps["play_type"].isin(["run", "pass"]) & snaps["down"].notna()
                      & (snaps["two_point_attempt"].fillna(0) == 0) & (snaps["qb_spike"].fillna(0) == 0)
                      & snaps["yardline_100"].notna()]
        for r in plays.itertuples(index=False):
            mode = game_mode(_half(r.qtr), r.half_seconds_remaining, r.score_differential)
            yards = 0 if math.isnan(r.yards_gained) else int(r.yards_gained)
            turnover = int(r.interception == 1 or r.fumble_lost == 1)
            if r.interception == 1:
                net = (0 if math.isnan(r.air_yards) else r.air_yards) - (0 if math.isnan(r.return_yards) else r.return_yards)
            else:
                net = yards
            def_td = int(turnover and r.return_touchdown == 1)
            stops = int(r.incomplete_pass == 1 or r.out_of_bounds == 1 or turnover == 1 or yards >= r.yardline_100)
            outcome = (yards, turnover, int(net), def_td, stops)
            d, t, z = min(int(r.down), 3), dist_bucket(r.ydstogo), zone_bucket(r.yardline_100)
            for kind in (r.play_type, None):
                for key in [(d, t, z, mode, kind), (d, t, z, None, kind), (None, t, z, None, kind),
                            (d, t, None, None, kind), (None, t, None, None, kind), (None, None, None, None, kind)]:
                    self.scrimmage.setdefault(key, []).append(outcome)
            if r.clean_duration:
                self.durations.setdefault((mode, stops), []).append(float(r.duration))

    def _fit_special(self, snaps: pd.DataFrame, df: pd.DataFrame) -> None:
        punts = snaps[(snaps["play_type"] == "punt") & snaps["yardline_100"].notna()]
        for r in punts.itertuples(index=False):
            zone = punt_zone(r.yardline_100)
            if r.return_touchdown == 1 and r.td_team == r.defteam:
                self.punts.setdefault(zone, []).append((0, 1, 12.0))
            elif r.next_posteam == r.defteam and not math.isnan(r.next_yardline):
                self.punts.setdefault(zone, []).append((int(r.yardline_100 - (100 - r.next_yardline)), 0, r.duration))
        kicks = snaps[snaps["play_type"] == "kickoff"]
        onside = kicks[kicks["kick_distance"].fillna(99) <= 20]
        if len(onside) >= 10:
            self.onside_recovery = float((onside["next_posteam"] == onside["defteam"]).mean())
        normal = kicks[(kicks["kick_distance"].fillna(99) > 20) & (kicks["next_posteam"] == kicks["posteam"])]
        self.kickoffs = [int(y) for y in normal["next_yardline"].dropna()]
        fg = df[(df["play_type"] == "field_goal") & df["kick_distance"].notna()]
        self.fg_coef = _logistic(np.column_stack([np.ones(len(fg)), fg["kick_distance"].to_numpy(float)]),
                                 (fg["field_goal_result"] == "made").to_numpy(float))
        self.fg_coef = (float(self.fg_coef[0]), float(self.fg_coef[1]))
        xp = df[df["extra_point_result"].notna()]
        self.pat_rate = float((xp["extra_point_result"] == "good").mean())
        tp = df[df["two_point_conv_result"].notna()]
        self.two_pt_rate = float((tp["two_point_conv_result"] == "success").mean())
        # PAT distance moved back in 2015, so these are kept per season.
        self.pat_by_season = {int(k): float(v) for k, v in
                              xp.groupby("season")["extra_point_result"].apply(lambda r: (r == "good").mean()).items()}
        self.two_pt_by_season = {int(k): float(v) for k, v in
                                 tp.groupby("season")["two_point_conv_result"].apply(lambda r: (r == "success").mean()).items()}

    def _fit_drives(self, df: pd.DataFrame) -> None:
        for d in build_drives(df).itertuples(index=False):
            if d.result == "END" or d.result not in CHANGE_OF_POSSESSION:
                sample = (d.result, float(d.duration), 0)
            elif d.next_same_half and d.next_is_opponent:
                sample = (d.result, float(d.duration), int(d.next_yl))
            else:
                sample = ("END", float(d.duration), 0)
            keys = drive_keys(d.half, d.half_secs, d.diff, d.yl)
            # The last key is the general fallback pool; late-half drives stay out of it.
            native = keys if len(keys) == 1 else keys[:-1]
            for key in native:
                self.drives.setdefault(key, []).append(sample)


def build_drives(df: pd.DataFrame) -> pd.DataFrame:
    """One row per drive: start situation, result, duration and next drive start."""
    snaps = df[df["posteam"].notna() & df["down"].notna() & df["fixed_drive"].notna()
               & df["yardline_100"].notna()]
    first = snaps.groupby(["game_id", "fixed_drive"], sort=True).first().reset_index()
    first["half"] = first["qtr"].map(_half)
    g = first.groupby("game_id")
    first["next_half"] = g["half"].shift(-1)
    first["next_half_secs"] = g["half_seconds_remaining"].shift(-1)
    first["next_posteam"] = g["posteam"].shift(-1)
    first["next_yl"] = g["yardline_100"].shift(-1)
    same = first["next_half"] == first["half"]
    out = pd.DataFrame({
        "game_id": first["game_id"],
        "season": first["season"],
        "posteam": first["posteam"],
        "half": first["half"],
        "half_secs": first["half_seconds_remaining"],
        "diff": first["score_differential"],
        "yl": first["yardline_100"],
        "result": first["fixed_drive_result"].map(DRIVE_RESULTS),
        "duration": np.where(same, first["half_seconds_remaining"] - first["next_half_secs"],
                             first["half_seconds_remaining"]).clip(0, None),
        "next_same_half": same,
        "next_is_opponent": first["next_posteam"] != first["posteam"],
        "next_yl": first["next_yl"],
    })
    return out[out["result"].notna()].reset_index(drop=True)


def _annotate_snaps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real snaps; attach clock run-off to the next snap and its possession."""
    df = df.copy()
    is_snap = df["play_type"].isin(SNAP_TYPES)
    df["grp"] = is_snap.groupby(df["game_id"]).cumsum()
    timeout_groups = set(zip(df.loc[df["timeout"] == 1, "game_id"], df.loc[df["timeout"] == 1, "grp"]))
    snaps = df[is_snap].copy()
    g = snaps.groupby("game_id")
    snaps["next_gsr"] = g["game_seconds_remaining"].shift(-1)
    snaps["next_half_secs"] = g["half_seconds_remaining"].shift(-1)
    snaps["next_qtr"] = g["qtr"].shift(-1)
    snaps["next_posteam"] = g["posteam"].shift(-1)
    snaps["next_yardline"] = g["yardline_100"].shift(-1)
    snaps["duration"] = (snaps["game_seconds_remaining"] - snaps["next_gsr"]).clip(0, 45).fillna(5.0)
    had_timeout = pd.Series([k in timeout_groups for k in zip(snaps["game_id"], snaps["grp"])], index=snaps.index)
    same_half = snaps["qtr"].map(_half) == snaps["next_qtr"].map(lambda q: _half(q) if q == q else -1)
    two_min = (snaps["half_seconds_remaining"] > 120) & (snaps["next_half_secs"] == 120)
    snaps["clean_duration"] = snaps["next_gsr"].notna() & same_half & ~had_timeout & ~two_min & (snaps["qtr"] <= 4)
    return snaps


def _logistic(X: np.ndarray, y: np.ndarray, iters: int = 60) -> np.ndarray:
    """Newton-Raphson logistic regression with a tiny ridge for stability."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (y - p) - 1e-6 * w
        hess = -(X.T * (p * (1 - p))) @ X - 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-9:
            break
    return w


def load_model(season: int, refresh: bool = False) -> GameModel:
    """Game model calibrated on the seasons around `season` (cached)."""
    lo, hi = default_window(season)
    path = data.cache_dir() / f"game_model_v{MODEL_VERSION}_{lo}_{hi}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    model = GameModel.fit(data.load_pbp(range(lo, hi + 1)), (lo, hi))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model
