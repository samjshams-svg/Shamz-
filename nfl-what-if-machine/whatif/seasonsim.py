"""Part 2: replay history from the changed game and simulate the next seasons.

Simulations are vectorized: every array has one row per simulated history.
Ratings update after every simulated game ("hot" simulation, as 538 did), so
an upset in week 3 carries into week 4 and into next season.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import elo as E
from .teams import CONFERENCE_OF, DIVISION_OF, DIVISIONS, TEAM_INDEX, TEAMS

N_TEAMS = len(TEAMS)
CONF = np.array([0 if CONFERENCE_OF[t] == "AFC" else 1 for t in TEAMS])
DIV_ID = {d: i for i, d in enumerate(DIVISIONS)}
DIV = np.array([DIV_ID[DIVISION_OF[t]] for t in TEAMS])
DIV_TEAMS = [[TEAM_INDEX[t] for t in DIVISIONS[d]] for d in DIVISIONS]
CONF_DIVS = [[i for i, d in enumerate(DIVISIONS) if d.startswith(c)] for c in ("AFC", "NFC")]
ROUNDS = ["WC", "DIV", "CON", "SB"]


def playoff_teams(season: int) -> int:
    return 7 if season >= 2020 else 6


@dataclass
class Adjustment:
    """Hand-set Elo bump for one team from a point in time, with a note."""
    team: str
    season: int
    elo: float
    note: str = ""
    week: int = 1
    world: str = "alternate"          # "alternate", "baseline" or "both"
    if_winner: str | None = None      # only in histories where this team won the changed game

    def applies_to(self, world: str) -> bool:
        return self.world in ("both", world)


@dataclass
class Change:
    game_id: str
    season: int
    game_type: str
    home: str
    away: str
    margins: np.ndarray               # home minus away, one per simulated history


@dataclass
class SeasonOutcome:
    season: int
    wins: np.ndarray                  # (n, 32)
    playoffs: np.ndarray              # (n, 32) bool
    division: np.ndarray
    conference: np.ndarray
    champion: np.ndarray


@dataclass
class WorldResult:
    world: str
    seasons: list[SeasonOutcome]
    timeline: list[tuple[int, int]] = field(default_factory=list)   # (season, week) after each week
    elo_mean: list[np.ndarray] = field(default_factory=list)          # (32,) per timeline point
    elo_p10: list[np.ndarray] = field(default_factory=list)
    elo_p90: list[np.ndarray] = field(default_factory=list)


# ---- standings and seeding ------------------------------------------------------------
class Standings:
    """Doubled win counts (a tie = 1) so everything stays integer."""

    def __init__(self, n: int, games: pd.DataFrame):
        self.W2 = np.zeros((n, N_TEAMS), dtype=np.int16)
        self.DW2 = np.zeros_like(self.W2)
        self.CW2 = np.zeros_like(self.W2)
        self.HW2 = np.zeros((n, N_TEAMS, N_TEAMS), dtype=np.int8)
        G = np.zeros((N_TEAMS, N_TEAMS), dtype=np.int16)
        for h, a in zip(games["h"], games["a"]):
            G[h, a] += 1
            G[a, h] += 1
        self.G = G
        self.games = G.sum(1)
        same_div = DIV[:, None] == DIV[None, :]
        same_conf = CONF[:, None] == CONF[None, :]
        self.div_games = (G * same_div).sum(1)
        self.conf_games = (G * same_conf).sum(1)

    def add(self, h: int, a: int, margin: np.ndarray) -> None:
        hw = np.where(margin > 0, 2, np.where(margin == 0, 1, 0)).astype(np.int16)
        aw = 2 - hw
        self.W2[:, h] += hw
        self.W2[:, a] += aw
        if DIV[h] == DIV[a]:
            self.DW2[:, h] += hw
            self.DW2[:, a] += aw
        if CONF[h] == CONF[a]:
            self.CW2[:, h] += hw
            self.CW2[:, a] += aw
        self.HW2[:, h, a] += hw.astype(np.int8)
        self.HW2[:, a, h] += aw.astype(np.int8)

    def seed(self, season: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        """Return (seeds (n, 2, k) team indices, division winners (n, 32) bool)."""
        n = self.W2.shape[0]
        k = playoff_teams(season)
        g = np.maximum(self.games, 1)
        pct = self.W2 / (2 * g)
        dpct = self.DW2 / (2 * np.maximum(self.div_games, 1))
        cpct = self.CW2 / (2 * np.maximum(self.conf_games, 1))
        beat = self.HW2.astype(np.float32)
        sov = (beat * pct[:, None, :]).sum(2) / np.maximum(beat.sum(2), 1)
        sos = (self.G[None].astype(np.float32) * pct[:, None, :]).sum(2) / np.maximum(self.G.sum(1), 1)
        seeds = np.zeros((n, 2, k), dtype=np.int64)
        div_win = np.zeros((n, N_TEAMS), dtype=bool)
        G = self.G.tolist()
        for s in range(n):
            ctx = _SeedContext(self.W2[s].tolist(), dpct[s].tolist(), cpct[s].tolist(), sov[s].tolist(),
                               sos[s].tolist(), self.HW2[s].tolist(), G, rng)
            for c in (0, 1):
                order, winners = ctx.conference(c, k)
                seeds[s, c] = order
                div_win[s, winners] = True
        return seeds, div_win


class _SeedContext:
    def __init__(self, w2, dpct, cpct, sov, sos, hw, G, rng):
        self.w2, self.dpct, self.cpct, self.sov, self.sos, self.hw, self.G = w2, dpct, cpct, sov, sos, hw, G
        self.rng = rng

    def order(self, teams: list[int], division: bool) -> list[int]:
        teams = sorted(teams, key=lambda t: -self.w2[t])
        out, i = [], 0
        while i < len(teams):
            j = i
            while j + 1 < len(teams) and self.w2[teams[j + 1]] == self.w2[teams[i]]:
                j += 1
            group = teams[i:j + 1]
            out.extend(group if len(group) == 1 else self._break(group, division))
            i = j + 1
        return out

    def _break(self, group: list[int], division: bool) -> list[int]:
        """Order teams with the same record using the NFL procedure: find the best
        team, then restart with the rest. Steps (division ties): head-to-head,
        division record, common games, conference record, strength of victory,
        strength of schedule, coin flip. Wild-card ties: head-to-head only for a
        sweep, conference record, common games (at least 4), SOV, SOS, coin flip.
        Net-points steps are skipped."""
        out, rest = [], list(group)
        while len(rest) > 1:
            best = self._best(rest, division)
            out.append(best)
            rest.remove(best)
        return out + rest

    def _best(self, group: list[int], division: bool) -> int:
        if len(group) == 1:
            return group[0]
        steps = [self._h2h] + ([self._div] if division else []) + \
            ([self._common, self._conf] if division else [self._conf, self._common]) + \
            [self._sov, self._sos]
        for step in steps:
            vals = step(group, division)
            if vals is None:
                continue
            top = max(vals.values())
            leaders = [t for t in group if vals[t] == top]
            if len(leaders) == 1:
                return leaders[0]
            if len(leaders) < len(group):
                return self._best(leaders, division)  # restart with the survivors
        return self.rng.choice(group)

    def _h2h(self, group, division):
        if len(group) == 2 or division:
            vals = {}
            for t in group:
                gp = sum(self.G[t][o] for o in group if o != t)
                if gp == 0:
                    return None
                vals[t] = sum(self.hw[t][o] for o in group if o != t) / (2 * gp)
            return vals
        # Wild-card ties of 3+: only a sweep (beat everyone / lost to everyone) counts.
        for t in group:
            others = [o for o in group if o != t]
            if all(self.G[t][o] > 0 and self.hw[t][o] == 2 * self.G[t][o] for o in others):
                return {x: float(x == t) for x in group}
        for t in group:
            others = [o for o in group if o != t]
            if all(self.G[t][o] > 0 and self.hw[t][o] == 0 for o in others):
                return {x: float(x != t) for x in group}
        return None

    def _div(self, group, division):
        return {t: self.dpct[t] for t in group}

    def _conf(self, group, division):
        return {t: self.cpct[t] for t in group}

    def _common(self, group, division):
        opps = [o for o in range(len(self.G)) if o not in group and all(self.G[t][o] > 0 for t in group)]
        games = {t: sum(self.G[t][o] for o in opps) for t in group}
        if not opps or (not division and min(games.values()) < 4):
            return None
        return {t: sum(self.hw[t][o] for o in opps) / (2 * games[t]) for t in group}

    def _sov(self, group, division):
        return {t: self.sov[t] for t in group}

    def _sos(self, group, division):
        return {t: self.sos[t] for t in group}

    def conference(self, c: int, k: int) -> tuple[list[int], list[int]]:
        div_orders = [self.order(DIV_TEAMS[d], division=True) for d in CONF_DIVS[c]]
        winners = [o[0] for o in div_orders]
        seeds = self.order(winners, division=False)
        rest = [o[1:] for o in div_orders]
        for _ in range(k - 4):
            # Only the best remaining team of each division is eligible at each step.
            cands = [r[0] for r in rest if r]
            best = self.order(cands, division=False)[0]
            seeds.append(best)
            for r in rest:
                if r and r[0] == best:
                    r.pop(0)
        return seeds, winners


# ---- the simulator ------------------------------------------------------------------------
class HistorySim:
    def __init__(self, schedules: pd.DataFrame, params: E.EloParams, n: int, seed: int = 1):
        self.params = params
        self.n = n
        self.seed = seed
        g = schedules.copy()
        g["neutral"] = g["location"] == "Neutral"
        g["playoff"] = g["game_type"].isin(E.PLAYOFF_TYPES)
        g["margin"] = g["home_score"] - g["away_score"]
        g["h"] = g["home_team"].map(TEAM_INDEX)
        g["a"] = g["away_team"].map(TEAM_INDEX)
        self.games = g
        # Margin pools by Elo gap and whether the favorite won: favorites win by
        # more, which is what 538's margin-of-victory damping expects. Drawing
        # margins without this makes simulated ratings drift toward the mean.
        _, log = E.run_history(schedules, params)
        d = (log["home_elo"] - log["away_elo"] + np.where(log["game_type"] == "SB", 0, params.hfa)).to_numpy()
        m = log["margin"].to_numpy()
        keep = m != 0
        d, m = d[keep], m[keep]
        fav_won = np.sign(d) == np.sign(m)
        gap = self._gap_bucket(np.abs(d))
        self.margin_pools = {(b, fw): np.abs(m[(gap == b) & (fav_won == fw)])
                             for b in range(len(self.GAP_EDGES) + 1) for fw in (False, True)}

    # ---- helpers ------------------------------------------------------------------------
    def season_games(self, season: int) -> pd.DataFrame:
        return self.games[self.games["season"] == season]

    GAP_EDGES = (40, 80, 130, 200)

    def _gap_bucket(self, gap: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.array(self.GAP_EDGES), gap)

    def _sample_margin(self, rng: np.random.Generator, diff: np.ndarray, fav_won: np.ndarray) -> np.ndarray:
        gap = self._gap_bucket(np.abs(diff))
        u = rng.random(self.n)
        out = np.empty(self.n)
        for (b, fw), pool in self.margin_pools.items():
            mask = (gap == b) & (fav_won == fw)
            if mask.any():
                out[mask] = pool[(u[mask] * len(pool)).astype(int)]
        return out

    def _play(self, elo: np.ndarray, h, a, neutral: bool, playoff: bool, rng: np.random.Generator,
              margin: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(self.n)
        eh, ea = elo[rows, h], elo[rows, a]
        if margin is None:
            d = E.game_diff(eh, ea, neutral, playoff, self.params)
            home_win = rng.random(self.n) < E.win_prob(d)
            m = self._sample_margin(rng, d, (d >= 0) == home_win)
            margin = np.where(home_win, m, -m)
        shift, _ = E.update(eh, ea, margin, neutral, playoff, self.params)
        elo[rows, h] += shift
        elo[rows, a] -= shift
        return margin

    # ---- worlds ------------------------------------------------------------------------------
    def run(self, start: dict[str, float], change: Change, world: str, n_seasons: int = 2,
            adjustments: list[Adjustment] | None = None) -> WorldResult:
        rng = np.random.default_rng(self.seed)          # same random stream in both worlds
        pyrng = random.Random(self.seed)
        elo = np.tile(np.array([start[t] for t in TEAMS], dtype=np.float64), (self.n, 1))
        result = WorldResult(world, [])
        adj = [a for a in (adjustments or []) if a.applies_to(world)]
        winner_is = {t: change.margins > 0 if t == change.home else change.margins < 0
                     for t in (change.home, change.away)}
        pending = sorted(adj, key=lambda a: (a.season, a.week))

        def apply_adjustments(season: int, week: int) -> None:
            while pending and (pending[0].season, pending[0].week) <= (season, week):
                a = pending.pop(0)
                mask = winner_is.get(a.if_winner, np.ones(self.n, bool)) if a.if_winner else np.ones(self.n, bool)
                elo[mask, TEAM_INDEX[a.team]] += a.elo

        def snapshot(season: int, week: int) -> None:
            result.timeline.append((season, week))
            result.elo_mean.append(elo.mean(0))
            result.elo_p10.append(np.percentile(elo, 10, axis=0))
            result.elo_p90.append(np.percentile(elo, 90, axis=0))

        S = change.season
        games = self.season_games(S)
        order = list(games["game_id"])
        i0 = order.index(change.game_id)
        before, after = games.iloc[:i0], games.iloc[i0 + 1:]
        reg = games[~games["playoff"]]
        st = Standings(self.n, reg)
        for g in before[~before["playoff"]].itertuples(index=False):
            st.add(g.h, g.a, np.full(self.n, g.margin))

        cg = games.iloc[i0]
        self._play(elo, cg.h, cg.a, bool(cg.neutral), bool(cg.playoff), rng, change.margins.astype(float))
        if not cg.playoff:
            st.add(cg.h, cg.a, change.margins)
        apply_adjustments(S, int(cg.week))
        snapshot(S, int(cg.week))

        if not cg.playoff:
            self._regular_season(elo, st, after[~after["playoff"]], S, rng, apply_adjustments, snapshot)
            seeds, div_win = st.seed(S, pyrng)
            known = {}
        else:
            seeds, div_win = st.seed(S, pyrng)  # real standings: the same in every history
            known = {frozenset((g.h, g.a)): (g.h, g.margin) for g in before[before["playoff"]].itertuples(index=False)}
            known[frozenset((cg.h, cg.a))] = (cg.h, change.margins.astype(float))
        result.seasons.append(self._playoffs(elo, st, seeds, div_win, S, rng, known))
        snapshot(S, 23)

        for season in range(S + 1, S + 1 + n_seasons):
            elo[:] = E.MEAN + (1 - self.params.regress) * (elo - E.MEAN)
            games = self.season_games(season)
            reg = games[~games["playoff"]]
            st = Standings(self.n, reg)
            apply_adjustments(season, 0)
            snapshot(season, 0)
            self._regular_season(elo, st, reg, season, rng, apply_adjustments, snapshot)
            seeds, div_win = st.seed(season, pyrng)
            result.seasons.append(self._playoffs(elo, st, seeds, div_win, season, rng, {}))
            snapshot(season, 23)
        return result

    def _regular_season(self, elo, st: Standings, games: pd.DataFrame, season: int, rng, apply_adjustments,
                        snapshot) -> None:
        week = None
        for g in games.itertuples(index=False):
            if g.week != week:
                if week is not None:
                    snapshot(season, int(week))
                week = g.week
                apply_adjustments(season, int(week))
            margin = self._play(elo, g.h, g.a, bool(g.neutral), False, rng)
            st.add(g.h, g.a, margin)
        if week is not None:
            snapshot(season, int(week))

    def _playoffs(self, elo, st: Standings, seeds: np.ndarray, div_win: np.ndarray, season: int, rng,
                  known: dict) -> SeasonOutcome:
        n, k = self.n, seeds.shape[2]
        rows = np.arange(n)
        made = np.zeros((n, N_TEAMS), dtype=bool)
        for c in (0, 1):
            made[rows[:, None], seeds[:, c, :]] = True
        conf_champ = np.zeros((n, N_TEAMS), dtype=bool)
        champs = []

        def game(home: np.ndarray, away: np.ndarray, neutral: bool) -> np.ndarray:
            """Return winner indices. Games already played (before the change, and the
            changed game itself) keep their result and are already in the ratings."""
            if np.all(home == home[0]) and np.all(away == away[0]):
                key = frozenset((int(home[0]), int(away[0])))
                if key in known:
                    real_home, m = known[key]
                    m = np.broadcast_to(np.asarray(m, dtype=float), (n,))
                    margin = m if real_home == home[0] else -m
                    return np.where(margin > 0, home, away)
            margin = self._play(elo, home, away, neutral, True, rng)
            return np.where(margin > 0, home, away)

        for c in (0, 1):
            seed_no = np.tile(np.arange(1, k + 1), (n, 1))       # seed numbers still alive
            team = seeds[:, c, :]
            byes = 2 if k == 6 else 1
            pairs = [(2, 5), (3, 4)] if k == 6 else [(1, 6), (2, 5), (3, 4)]
            alive = [seed_no[:, i] for i in range(byes)]
            for hi, lo in pairs:
                w = game(team[:, hi], team[:, lo], False)
                alive.append(np.where(w == team[:, hi], hi + 1, lo + 1))
            alive = np.sort(np.stack(alive, 1), 1)              # (n, 4) seed numbers, best first
            div_w = []
            for x, y in ((0, 3), (1, 2)):
                h = team[rows, alive[:, x] - 1]
                a = team[rows, alive[:, y] - 1]
                w = game(h, a, False)
                div_w.append(np.where(w == h, alive[:, x], alive[:, y]))
            fin = np.sort(np.stack(div_w, 1), 1)
            h, a = team[rows, fin[:, 0] - 1], team[rows, fin[:, 1] - 1]
            champ = game(h, a, False)
            conf_champ[rows, champ] = True
            champs.append(champ)
        sb = game(champs[0], champs[1], True)
        champion = np.zeros((n, N_TEAMS), dtype=bool)
        champion[rows, sb] = True
        return SeasonOutcome(season, st.W2 / 2.0, made, div_win, conf_champ, champion)


# ---- what actually happened --------------------------------------------------------------------
def actual_outcomes(schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    g = schedules[schedules["season"] == season].copy()
    g["h"] = g["home_team"].map(TEAM_INDEX)
    g["a"] = g["away_team"].map(TEAM_INDEX)
    g["margin"] = g["home_score"] - g["away_score"]
    reg = g[g["game_type"] == "REG"]
    st = Standings(1, reg)
    for r in reg.itertuples(index=False):
        if pd.notna(r.margin):
            st.add(r.h, r.a, np.array([r.margin]))
    _, div_win = st.seed(season, random.Random(0))
    po = g[g["game_type"] != "REG"]
    playoff_teams_ = set(po["home_team"]) | set(po["away_team"])
    con = po[po["game_type"] == "CON"]
    con_w = {r.home_team if r.margin > 0 else r.away_team for r in con.itertuples()}
    sb = po[po["game_type"] == "SB"]
    sb_w = {r.home_team if r.margin > 0 else r.away_team for r in sb.itertuples()}
    return pd.DataFrame({
        "team": TEAMS,
        "wins": st.W2[0] / 2.0,
        "playoffs": [t in playoff_teams_ for t in TEAMS],
        "division": div_win[0],
        "conference": [t in con_w for t in TEAMS],
        "champion": [t in sb_w for t in TEAMS],
    }).set_index("team")


def real_seeds(schedules: pd.DataFrame, season: int) -> tuple[list[str], list[str]]:
    """Seeds from real standings with this module's tiebreakers (AFC, NFC)."""
    g = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")].copy()
    g["h"] = g["home_team"].map(TEAM_INDEX)
    g["a"] = g["away_team"].map(TEAM_INDEX)
    st = Standings(1, g)
    for r in g.itertuples(index=False):
        st.add(r.h, r.a, np.array([r.home_score - r.away_score]))
    seeds, _ = st.seed(season, random.Random(0))
    return [TEAMS[i] for i in seeds[0, 0]], [TEAMS[i] for i in seeds[0, 1]]
