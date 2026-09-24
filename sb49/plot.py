"""Real (nflfastR) vs simulated (Monte Carlo) win-probability chart."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
REAL = "#2a78d6"   # categorical slot 1: nflfastR / what actually happened
SIM = "#eb6834"    # categorical slot 2: this project's Monte Carlo


@dataclass
class ChartData:
    team: str
    opponent: str
    real_x: np.ndarray          # minutes elapsed
    real_wp: np.ndarray         # nflfastR win prob for `team`
    sim_x: np.ndarray
    sim_wp: np.ndarray          # Monte Carlo win prob for `team`, at the real game states
    play_x: float               # minutes elapsed at the changed play
    play_label: str             # short text for the changed play
    new_label: str              # short text for the new result
    actual_post_wp: float       # nflfastR win prob after what really happened
    cf_wp: float                # Monte Carlo win prob after the changed result
    cf_ci: tuple[float, float]
    cf_margins: np.ndarray      # final margins (team - opponent) in the counterfactual sims
    actual_margin: int
    n_sims: int


def minutes_elapsed(game_seconds_remaining: pd.Series | np.ndarray) -> np.ndarray:
    return 60 - np.asarray(game_seconds_remaining, dtype=float) / 60


def _clock_label(x: float, _pos=None) -> str:
    """Minutes elapsed -> 'Q4 2:00' style game clock."""
    if x >= 60:
        return "Final"
    qtr = int(x // 15) + 1
    left = round((15 * qtr - x) * 60)
    return f"Q{qtr} {left // 60}:{left % 60:02d}"


def _style(a) -> None:
    a.set_facecolor(SURFACE)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.tick_params(length=0)


def _wp_lines(a, d: ChartData, with_labels: bool) -> None:
    a.axhline(50, color=GRID, lw=1, zorder=0)
    a.plot(d.real_x, d.real_wp * 100, color=REAL, lw=2, label="nflfastR win probability (real game)")
    if len(d.sim_x):
        a.plot(d.sim_x, d.sim_wp * 100, color=SIM, lw=2,
               label="Monte Carlo win probability (from each real game state)")
    a.axvline(d.play_x, color=TEXT_2, lw=1, ls=(0, (2, 3)), zorder=1)
    lo, hi = d.cf_ci
    a.errorbar([d.play_x], [d.cf_wp * 100], yerr=[[(d.cf_wp - lo) * 100], [(hi - d.cf_wp) * 100]],
               fmt="D", ms=9, color=SIM, mec=SURFACE, mew=2, ecolor=SIM, elinewidth=2, capsize=0, zorder=5,
               label=f"Monte Carlo after the changed play (95% CI, n={d.n_sims:,})")
    a.plot([d.play_x], [d.actual_post_wp * 100], "o", ms=9, color=REAL, mec=SURFACE, mew=2, zorder=5,
           label="nflfastR after the real play")
    a.set_ylim(0, 100)
    a.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    if not with_labels:
        return
    # Direct labels, pushed apart if the two points are close.
    ys = {"cf": d.cf_wp * 100, "real": d.actual_post_wp * 100}
    if abs(ys["cf"] - ys["real"]) < 12:
        hi_key = "cf" if ys["cf"] >= ys["real"] else "real"
        lo_key = "real" if hi_key == "cf" else "cf"
        mid = (ys["cf"] + ys["real"]) / 2
        ys[hi_key], ys[lo_key] = min(96, mid + 6), max(4, mid - 6)
    for key, text, color in (("cf", f"{d.new_label}: {d.cf_wp:.1%}", SIM),
                             ("real", f"What really happened: {d.actual_post_wp:.1%}", REAL)):
        y_pt = d.cf_wp * 100 if key == "cf" else d.actual_post_wp * 100
        x0, x1 = a.get_xlim()
        a.annotate(text, (d.play_x, y_pt), xytext=(d.play_x - 0.05 * (x1 - x0), min(95, max(5, ys[key]))),
                   textcoords="data", ha="right", va="center", fontsize=9.5, color=TEXT,
                   bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=color, lw=1))


def render(d: ChartData, path: str) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    })
    fig = plt.figure(figsize=(13, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], height_ratios=[3, 1.3], hspace=0.45, wspace=0.12, top=0.84)
    ax, zx, hx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a in (ax, zx, hx):
        _style(a)

    # ---- full game ----
    for q in (15, 30, 45):
        ax.axvline(q, color=GRID, lw=1, zorder=0)
    _wp_lines(ax, d, with_labels=False)
    z0, z1 = max(0.0, d.play_x - 6), min(60.4, d.play_x + 3)
    ax.axvspan(z0, z1, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xlim(0, 60.5)
    ax.set_xticks([0, 15, 30, 45, 60], ["Kickoff", "Q2", "Q3", "Q4", "Final"])
    ax.set_ylabel(f"{d.team} win probability")
    ax.set_title("Whole game", loc="left", fontsize=10.5, color=TEXT_2)

    # ---- zoom around the changed play ----
    zx.set_xlim(z0, z1 + 0.15)
    _wp_lines(zx, d, with_labels=True)
    ticks = np.linspace(z0, min(60, z1), 4)
    zx.set_xticks(ticks, [_clock_label(t) for t in ticks])
    zx.set_yticklabels([])
    zx.set_title("Zoom: shaded window", loc="left", fontsize=10.5, color=TEXT_2)

    fig.suptitle(f"Super Bowl XLIX: {d.team} win probability, real vs. simulated", x=0.125, y=0.985,
                 ha="left", fontsize=14, color=TEXT)
    fig.text(0.125, 0.94, f"Changed play: {d.play_label}", ha="left", fontsize=10, color=TEXT_2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.12, 0.925), ncol=2, frameon=False, fontsize=9)

    # ---- final margin distribution after the changed play ----
    m = d.cf_margins
    lo_b, hi_b = int(min(m.min(), d.actual_margin)) - 1, int(max(m.max(), d.actual_margin)) + 2
    bins = np.arange(lo_b, hi_b) - 0.5
    hx.hist(m, bins=bins, color=SIM, edgecolor=SURFACE, linewidth=2, weights=np.full(len(m), 100 / len(m)))
    hx.axvline(0, color=TEXT_2, lw=1)
    hx.axvline(d.actual_margin, color=REAL, lw=2)
    ymax = hx.get_ylim()[1]
    hx.text(d.actual_margin, ymax * 0.95, f"  real final ({d.actual_margin:+d})", color=TEXT, fontsize=9, va="top")
    hx.set_title(f"Final margin after the changed play ({d.team} minus {d.opponent}): "
                 f"{d.team} wins {d.cf_wp:.1%}, {d.opponent} wins {1 - d.cf_wp:.1%}",
                 loc="left", fontsize=10.5, color=TEXT)
    hx.set_xlabel("Final point margin")
    hx.set_ylabel("% of simulations")
    hx.grid(axis="y", color=GRID, lw=1)
    hx.set_axisbelow(True)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
