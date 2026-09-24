"""Matplotlib charts shared by the CLI and the web page."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
REAL = "#2a78d6"      # what actually happened
ALT = "#eb6834"       # the rewritten history
BASE = "#52514e"      # simulated, no change (neutral so it recedes)


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    })


def _style(a) -> None:
    a.set_facecolor(SURFACE)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.tick_params(length=0)


def _clock_label(x: float) -> str:
    if x >= 60:
        return "End" if x < 60.01 else f"OT {x - 60:.0f}m"
    q = int(x // 15) + 1
    left = round((15 * q - x) * 60)
    return f"Q{q} {left // 60}:{left % 60:02d}"


def path_wp(mc, start: float, end: float, n_grid: int = 160, smooth: float = 20.0):
    """Home win probability along each simulated path, estimated from the
    simulations themselves: at time t, a path's WP is the share of simulated
    games in the same situation (score, possession, field position) that the
    home team went on to win, shrunk toward games with the same score margin."""
    grid = np.linspace(start, end, n_grid)
    y = mc.home_outcome
    fine: dict = {}
    coarse: dict = {}
    K = np.zeros((len(mc.paths), n_grid), dtype=np.int64)
    C = np.zeros_like(K)
    for i, p in enumerate(mc.paths):
        t = np.array([x for x, _ in p])
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        sit = [p[j][1] for j in idx]
        K[i] = [fine.setdefault(k, len(fine)) for k in sit]
        C[i] = [coarse.setdefault(k if k[0] == "END" else k[0], len(coarse)) for k in sit]
    out = np.zeros(K.shape)
    overall = y.mean()
    for g in range(n_grid):
        cn = np.bincount(C[:, g], minlength=len(coarse))
        cs = np.bincount(C[:, g], weights=y, minlength=len(coarse))
        c_wp = (cs + smooth * overall) / (cn + smooth)
        fn = np.bincount(K[:, g], minlength=len(fine))
        fs = np.bincount(K[:, g], weights=y, minlength=len(fine))
        prior = c_wp[C[:, g]]
        out[:, g] = (fs[K[:, g]] + smooth * prior) / (fn[K[:, g]] + smooth)
    # Finished games are certain.
    for i, p in enumerate(mc.paths):
        t_end = p[-1][0]
        out[i, grid >= t_end] = y[i]
    return grid, out


def win_probability(rw, team: str | None = None, path: str | None = None):
    """Actual (nflfastR) vs alternate win probability for one team."""
    _setup()
    team = team or rw.pre.offense
    opp = rw.away if team == rw.home else rw.home
    flip = (lambda p: p) if team == rw.home else (lambda p: 1 - p)

    fig = plt.figure(figsize=(13, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], height_ratios=[3, 1.3], hspace=0.45, wspace=0.12, top=0.84)
    ax, zx, hx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a in (ax, zx, hx):
        _style(a)

    x0 = rw.play_minutes
    ends = np.array([p[-1][0] for p in rw.alt.paths]) if rw.alt.paths else np.array([60.0])
    # Regulation only, unless the real game went to overtime.
    end = max(60.0, float(rw.actual_wp["minutes"].max()))
    ot_share = float((ends > 60.01).mean())
    band = None
    if rw.alt.paths:
        grid, wp = path_wp(rw.alt, x0, max(end, x0 + 0.1))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(wp), size=min(40, len(wp)), replace=False)
        band = (grid, wp[pick])
    p_alt = rw.alt.prob(team)
    lo, hi = rw.alt.ci95(team)
    p_real_after = flip(rw.nflfastr_after)

    def draw(a, labels: bool):
        a.axhline(50, color=GRID, lw=1, zorder=0)
        a.plot(rw.actual_wp["minutes"], flip(rw.actual_wp["home_wp"].to_numpy()) * 100, color=REAL, lw=2,
               label="Actual game (nflfastR win probability)")
        if band is not None:
            g, sample = band
            for k, row in enumerate(sample):
                a.plot(g, flip(row) * 100, color=ALT, lw=1, alpha=0.25,
                       label=f"Alternate game: {len(sample)} of {rw.alt.n:,} simulated paths" if k == 0 else None)
        a.axvline(x0, color=TEXT_2, lw=1, ls=(0, (2, 3)), zorder=1)
        a.errorbar([x0], [p_alt * 100], yerr=[[(p_alt - lo) * 100], [(hi - p_alt) * 100]], fmt="D", ms=9,
                   color=ALT, mec=SURFACE, mew=2, ecolor=ALT, elinewidth=2, zorder=5,
                   label=f"Monte Carlo after the edit ({rw.alt.n:,} games)")
        a.plot([x0], [p_real_after * 100], "o", ms=9, color=REAL, mec=SURFACE, mew=2, zorder=5,
               label="nflfastR after the real play")
        a.set_ylim(0, 100)
        a.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        if labels:
            ys = {"alt": p_alt * 100, "real": p_real_after * 100}
            if abs(ys["alt"] - ys["real"]) < 12:
                top = "alt" if ys["alt"] >= ys["real"] else "real"
                bot = "real" if top == "alt" else "alt"
                mid = (ys["alt"] + ys["real"]) / 2
                ys[top], ys[bot] = min(94, mid + 6), max(6, mid - 6)
            xa, xb = a.get_xlim()
            for key, text, color in (("alt", f"With the edit: {p_alt:.1%}", ALT),
                                     ("real", f"What happened: {p_real_after:.1%}", REAL)):
                a.annotate(text, (x0, ys[key]), xytext=(x0 - 0.04 * (xb - xa), min(94, max(6, ys[key]))),
                           textcoords="data", ha="right", va="center", fontsize=9.5, color=TEXT,
                           bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=color, lw=1))

    for q in (15, 30, 45, 60):
        ax.axvline(q, color=GRID, lw=1, zorder=0)
    draw(ax, labels=False)
    ax.set_xlim(0, end + 0.5)
    ticks = [0, 15, 30, 45, 60]
    ax.set_xticks(ticks, ["Kickoff", "Q2", "Q3", "Q4", "OT" if end > 60.5 else "End"])
    ax.set_ylabel(f"{team} win probability")
    ax.set_title("Whole game", loc="left", fontsize=10.5, color=TEXT_2)

    left = end - x0
    if left < 9:
        z0, z1 = max(0.0, x0 - min(6.0, max(1.0, 3 * left))), end + 0.03 * max(left, 1)
    else:
        z0, z1 = max(0.0, x0 - 2), x0 + 10
    ax.axvspan(z0, z1, color=GRID, alpha=0.45, lw=0, zorder=0)
    zx.set_xlim(z0, z1)
    draw(zx, labels=True)
    t = np.linspace(z0, min(z1, end), 4)
    zx.set_xticks(t, [_clock_label(v) for v in t])
    zx.set_yticklabels([])
    zx.set_title("Zoom: shaded window", loc="left", fontsize=10.5, color=TEXT_2)

    fig.suptitle(f"{rw.away} at {rw.home} ({rw.game_id}): {team} win probability, actual vs. alternate",
                 x=0.125, y=0.985, ha="left", fontsize=14, color=TEXT)
    fig.text(0.125, 0.94, f"{rw.pre.clock()}, {rw.pre.offense} {rw.pre.down}&{rw.pre.ydstogo} at {rw.pre.spot()}. "
             f"Edit: {rw.edit.describe()}. Real play: {_short(rw.play_desc)}", ha="left", fontsize=9.5, color=TEXT_2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.12, 0.925), ncol=3, frameon=False, fontsize=9)

    m = rw.alt.margins if team == rw.home else -rw.alt.margins
    real_margin = rw.final_score[team] - rw.final_score[opp]
    lo_b, hi_b = int(min(m.min(), real_margin)) - 1, int(max(m.max(), real_margin)) + 2
    hx.hist(m, bins=np.arange(lo_b, hi_b) - 0.5, color=ALT, edgecolor=SURFACE, linewidth=1.5,
            weights=np.full(len(m), 100 / len(m)))
    hx.axvline(0, color=TEXT_2, lw=1)
    hx.axvline(real_margin, color=REAL, lw=2)
    hx.text(real_margin, hx.get_ylim()[1] * 0.95, f"  real final ({real_margin:+d})", color=TEXT, fontsize=9, va="top")
    tie = f", tie {rw.alt.tie_prob:.1%}" if rw.alt.ties else ""
    if ot_share >= 0.005:
        tie += f" ({ot_share:.0%} went to overtime)"
    hx.set_title(f"Final margin in the alternate games ({team} minus {opp}): {team} wins {p_alt:.1%}, "
                 f"{opp} wins {rw.alt.prob(opp):.1%}{tie}", loc="left", fontsize=10.5, color=TEXT)
    hx.set_xlabel("Final point margin")
    hx.set_ylabel("% of simulations")
    hx.grid(axis="y", color=GRID, lw=1)
    hx.set_axisbelow(True)
    return _finish(fig, path)


def _short(desc: str, n: int = 90) -> str:
    import re
    desc = re.sub(r"^\([\d:]*\)\s*(\([^)]*\)\s*)*", "", str(desc))
    return desc if len(desc) <= n else desc[: n - 1] + "…"


def _finish(fig, path: str | None):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return fig
