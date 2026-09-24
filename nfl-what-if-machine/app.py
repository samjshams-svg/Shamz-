"""NFL What If Machine: local web page.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from whatif import charts, data, elo
from whatif.models import GameModel
from whatif.rewrite import OUTCOMES, Edit, apply_edit, real_runoff, state_from_row
from whatif.scenario import METRICS, Scenario, format_odds, run_scenario
from whatif.seasonsim import Adjustment
from whatif.teams import NAMES, TEAMS

st.set_page_config(page_title="NFL What If Machine", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Loading schedules…")
def schedules() -> pd.DataFrame:
    return data.load_schedules()


@st.cache_data(show_spinner="Loading play-by-play…")
def game_pbp(game_id: str) -> pd.DataFrame:
    return data.load_game(game_id)


def game_label(r) -> str:
    score = f"{int(r.away_score)}-{int(r.home_score)}" if pd.notna(r.home_score) else ""
    kind = "" if r.game_type == "REG" else f" [{r.game_type}]"
    return f"{r.away_team} @ {r.home_team} {score}{kind} ({r.gameday})"


# ---- sidebar: pick a game ---------------------------------------------------------------------
sched = schedules()
st.sidebar.title("🏈 NFL What If Machine")
st.sidebar.caption("Pick a game, click a play, change it, rewrite history.")
seasons = list(range(data.FIRST_SEASON, data.LAST_SEASON + 1))
season = st.sidebar.selectbox("Season", seasons, index=seasons.index(2014))
season_games = sched[sched["season"] == season]
weeks = sorted(season_games["week"].unique())
week_names = {w: (f"Week {w}" if w <= 17 else {18: "Wild card", 19: "Divisional", 20: "Conference",
                                                21: "Super Bowl"}.get(w, f"Week {w}")) for w in weeks}
week = st.sidebar.selectbox("Week", weeks, index=len(weeks) - 1 if season == 2014 else 0,
                            format_func=week_names.get)
team_filter = st.sidebar.selectbox("Team (optional)", ["any"] + TEAMS,
                                   format_func=lambda t: "any" if t == "any" else f"{t} {NAMES[t]}")
games = data.find_games(sched, season, week, None if team_filter == "any" else team_filter)
if games.empty:
    st.warning("No games match.")
    st.stop()
game_id = st.sidebar.radio("Game", list(games["game_id"]),
                           format_func=lambda gid: game_label(games[games["game_id"] == gid].iloc[0]))

with st.sidebar.expander("How it works"):
    st.markdown(
        "- **Part 1**: the drive in progress is finished play by play, and every later possession is a real "
        "2010s drive resampled from a similar situation. At least 10,000 games per run.\n"
        "- **Part 2**: a FiveThirtyEight-style Elo replays history from the changed game and simulates the "
        "next two seasons on the real schedules, playoffs and tiebreakers included.\n\n"
        + elo.calibration_report().replace("\n", "  \n"))

# ---- the plays ------------------------------------------------------------------------------------
game = game_pbp(game_id)
g = games[games["game_id"] == game_id].iloc[0]
st.title(f"{g.away_team} @ {g.home_team}, {season} {week_names[week]}")
st.caption(f"Final: {g.away_team} {int(g.away_score)}, {g.home_team} {int(g.home_score)} · {g.gameday} · {game_id}")

c1, c2, c3 = st.columns([1, 2, 1])
qtrs = c1.multiselect("Quarter", sorted(int(q) for q in game["qtr"].dropna().unique()))
search = c2.text_input("Search the play description", placeholder="e.g. INTERCEPTED, Lynch, TOUCHDOWN")
min_wpa = c3.slider("Only swings of at least (win prob.)", 0.0, 0.5, 0.0, 0.05)
plays = data.find_plays(game, search=search or None, min_abs_wpa=min_wpa or None)
plays = plays[plays["play_type"].isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"])]
if qtrs:
    plays = plays[plays["qtr"].isin(qtrs)]
table = data.play_summary(plays)
st.markdown("**Click a play to edit it.**")
picked = st.dataframe(table, hide_index=True, width="stretch", height=300, on_select="rerun",
                      selection_mode="single-row", key=f"plays_{game_id}",
                      column_config={"wpa": st.column_config.NumberColumn("WPA", format="%.3f"),
                                     "desc": st.column_config.TextColumn("play", width="large")})
rows = picked.selection.rows
default_play = 4205 if game_id == "2014_21_NE_SEA" else None
if rows:
    play_id = int(table.iloc[rows[0]]["play_id"])
elif default_play is not None and default_play in set(table["play_id"]):
    play_id = default_play
else:
    st.info("Select a play in the table above.")
    st.stop()

idx = int(game.index[game["play_id"] == play_id][0])
row = game.loc[idx]
pre = state_from_row(row, game)
st.subheader(f"Play {play_id}")
st.write(f"**What happened:** {row['desc']}")
st.write(f"**Before the snap:** {pre.describe()}")

# ---- the edit -------------------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.markdown("#### Change the play")
    outcome = st.selectbox("New result", list(OUTCOMES), format_func=lambda o: f"{o.replace('_', ' ')}: {OUTCOMES[o]}",
                           index=0, key=f"outcome_{play_id}")
    yards = None
    penalty_on, auto_fd = "defense", False
    if outcome in ("gain", "sack", "punt", "penalty", "interception", "fumble_lost"):
        default = {"gain": 5, "sack": 7, "punt": 40, "penalty": 5, "interception": 0, "fumble_lost": 0}[outcome]
        yards = st.number_input("Yards", -99, 99, default, key=f"yards_{play_id}_{outcome}")
    if outcome == "penalty":
        penalty_on = st.radio("Penalty on", ["defense", "offense"], horizontal=True)
        auto_fd = st.checkbox("Automatic first down")
    clock_secs = None
    if outcome not in ("run", "pass", "custom") and st.checkbox("Set how long the play takes", key=f"cs_{play_id}"):
        clock_secs = st.number_input("Seconds off the clock", 0, 60, 6)
    edit = Edit(outcome=outcome, yards=yards, penalty_on=penalty_on, automatic_first_down=auto_fd,
                clock_secs=clock_secs)
    try:
        preview, winner, over = apply_edit(pre, edit, GameModel(seasons=(0, 0)), real_runoff(game, idx))
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.write("**After the edit:** " + (f"game over, {winner or 'tie'}" if over else preview.describe()))

    overrides = {}
    if not over:
        with st.expander("Set down, distance, yard line, score, clock by hand"):
            k = f"{play_id}_{outcome}_{yards}"
            teams = [preview.offense, preview.defense]
            poss = st.selectbox("Possession", teams, key=f"poss_{k}")
            a, b = st.columns(2)
            down = a.number_input("Down", 1, 4, preview.down, key=f"down_{k}")
            togo = b.number_input("Distance", 1, 99, preview.ydstogo, key=f"togo_{k}")
            yl = st.text_input("Yard line (e.g. 'NE 1', 'own 20')", preview.spot(), key=f"yl_{k}")
            a, b = st.columns(2)
            s_home = a.number_input(f"{pre.home} score", 0, 99, preview.score[pre.home], key=f"sh_{k}")
            s_away = b.number_input(f"{pre.away} score", 0, 99, preview.score[pre.away], key=f"sa_{k}")
            clock = st.text_input("Clock (e.g. 'Q4 0:20')", preview.clock(), key=f"clk_{k}")
            a, b = st.columns(2)
            t_home = a.number_input(f"{pre.home} timeouts", 0, 3, preview.timeouts[pre.home], key=f"th_{k}")
            t_away = b.number_input(f"{pre.away} timeouts", 0, 3, preview.timeouts[pre.away], key=f"ta_{k}")
            if poss != preview.offense:
                overrides["possession"] = poss
            if down != preview.down:
                overrides["down"] = int(down)
            if togo != preview.ydstogo:
                overrides["ydstogo"] = int(togo)
            if yl != preview.spot():
                overrides["yardline"] = yl
            if (s_home, s_away) != (preview.score[pre.home], preview.score[pre.away]):
                overrides["score"] = {pre.home: int(s_home), pre.away: int(s_away)}
            if clock != preview.clock():
                overrides["clock"] = clock
            if (t_home, t_away) != (preview.timeouts[pre.home], preview.timeouts[pre.away]):
                overrides["timeouts"] = {pre.home: int(t_home), pre.away: int(t_away)}
            if overrides:
                st.caption("Changed: " + ", ".join(overrides))
    for key, value in overrides.items():
        setattr(edit, key, value)

with right:
    st.markdown("#### Narrative adjustments (optional)")
    st.caption("Elo bumps you set by hand, e.g. a player doesn't retire. 25 Elo ≈ 1 point of spread. "
               "`if_winner` limits a bump to histories where that team won the changed game.")
    default_adj = pd.DataFrame([{"team": pre.offense, "season": season + 1, "week": 1, "elo": 0.0,
                                 "note": "", "if_winner": "", "world": "alternate"}])
    adj_df = st.data_editor(
        default_adj, num_rows="dynamic", hide_index=True, key=f"adj_{game_id}", width="stretch",
        column_config={
            "team": st.column_config.SelectboxColumn("team", options=TEAMS, required=True),
            "season": st.column_config.NumberColumn("season", min_value=season, max_value=season + 2, step=1),
            "week": st.column_config.NumberColumn("week", min_value=0, max_value=22, step=1),
            "elo": st.column_config.NumberColumn("Elo", min_value=-200, max_value=200, step=5),
            "note": st.column_config.TextColumn("note", width="large"),
            "if_winner": st.column_config.SelectboxColumn("if_winner", options=[""] + [g.home_team, g.away_team]),
            "world": st.column_config.SelectboxColumn("world", options=["alternate", "baseline", "both"]),
        })
    st.markdown("#### Simulation")
    a, b = st.columns(2)
    game_sims = a.number_input("Game simulations", 10_000, 50_000, 10_000, 5_000)
    season_sims = b.number_input("Season simulations", 1_000, 20_000, 10_000, 1_000)
    report_teams = st.multiselect("Teams in the report", TEAMS, default=[pre.offense, pre.defense])

adjustments = [Adjustment(team=r.team, season=int(r.season), elo=float(r.elo), note=str(r.note or ""),
                          week=int(r.week or 1), world=r.world or "alternate", if_winner=r.if_winner or None)
               for r in adj_df.dropna(subset=["team", "season", "elo"]).itertuples() if float(r.elo) != 0]

# ---- run ------------------------------------------------------------------------------------------
if st.button("✍️ Rewrite history", type="primary", width="stretch"):
    scn = Scenario(name=f"{game_id} play {play_id}: {edit.describe()}", game_id=game_id, play_id=play_id,
                   edit=edit, game_sims=int(game_sims), season_sims=int(season_sims),
                   teams=report_teams or [pre.offense, pre.defense], adjustments=adjustments)
    with st.status("Rewriting history…", expanded=True) as status:
        try:
            res = run_scenario(scn, progress=st.write)
        except ValueError as e:
            status.update(label="Could not run this edit", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Done", state="complete", expanded=False)
    st.session_state["result"] = res

res = st.session_state.get("result")
if res is None or res.game.play_id != play_id or res.game.game_id != game_id:
    st.stop()

rw = res.game
st.divider()
st.header("Part 1: the rewritten game")
cols = st.columns(3)
for col, team in zip(cols, (rw.pre.offense, rw.pre.defense)):
    real_after = rw.nflfastr_after if team == rw.home else 1 - rw.nflfastr_after
    col.metric(f"{team} win probability", f"{rw.alt.prob(team):.1%}",
               f"{(rw.alt.prob(team) - real_after) * 100:+.1f} pts vs. what happened")
cols[2].metric("Simulated games", f"{rw.alt.n:,}", f"ties {rw.alt.tie_prob:.1%}" if rw.alt.ties else None)
st.pyplot(charts.win_probability(rw, team=res.teams[0]), clear_figure=True)
with st.expander("Details"):
    st.code(rw.summary())

st.header("Part 2: the ripple effect")
st.caption(f"{res.scenario.season_sims:,} simulated histories with the edit and {res.scenario.season_sims:,} "
           f"without it, from the changed game through {res.alternate.seasons[-1].season}. "
           f"Elo: {res.elo_params.label()}.")
odds = res.odds_table()
show = []
for r in odds.to_dict("records"):
    for world, label in (("actual", "Actual"), ("base", "No change"), ("alt", "With the edit")):
        entry = {"Season": r["season"], "Team": r["team"], "World": label}
        for key, name in METRICS:
            v = r[f"{world}_{key}"]
            entry[name] = (f"{v:.1f}" if key == "wins" else ("yes" if v >= 0.5 else "no") if world == "actual"
                           else f"{v:.1%}")
        show.append(entry)
st.dataframe(pd.DataFrame(show), hide_index=True, width="stretch")
st.pyplot(charts.elo_trajectory(res), clear_figure=True)
mv = res.movers()
if not mv.empty:
    st.markdown("**Other teams whose playoff odds moved most**")
    st.dataframe(mv.assign(base_playoffs=mv["base_playoffs"].map("{:.1%}".format),
                           alt_playoffs=mv["alt_playoffs"].map("{:.1%}".format),
                           change=mv["change"].map("{:+.1%}".format)), hide_index=True)
if res.scenario.adjustments:
    st.markdown("**Narrative adjustments used**")
    for a in res.scenario.adjustments:
        st.write(f"- {a.team} {a.elo:+g} Elo from {a.season} week {a.week} ({a.world}"
                 f"{', only if ' + a.if_winner + ' won' if a.if_winner else ''}): {a.note}")
st.download_button("Download the report (Markdown)", res.text_report(), file_name="what_if_report.md")
st.download_button("Download the odds (CSV)", odds.to_csv(index=False), file_name="what_if_odds.csv")
with st.expander("Plain-text odds table"):
    st.code(format_odds(odds))
