import random

import pytest

from whatif.gamesim import Engine, GameOver
from whatif.models import GameModel
from whatif.rewrite import Edit, apply_edit, parse_game_clock, parse_yardline
from whatif.state import KICKOFF, PAT, SCRIMMAGE, GameState


@pytest.fixture
def model():
    m = GameModel(seasons=(2014, 2014))
    plays = [(3, 0, 0, 0, 0), (0, 0, 0, 0, 1), (8, 0, 0, 0, 0)] * 20
    for kind in ("run", "pass", None):
        m.scrimmage[(None, None, None, None, kind)] = plays
    for mode in ("normal", "hurry", "protect"):
        m.durations[(mode, 0)] = [30.0]
        m.durations[(mode, 1)] = [5.0]
    m.kickoffs = [75]
    m.punts = {z: [(40, 0, 8.0)] for z in range(4)}
    early = [("TD", 180.0, 0), ("PUNT", 150.0, 70), ("FG", 160.0, 0)] * 10
    for fp in range(6):
        m.drives[("early", fp)] = early
    return m


def state(**kw):
    base = dict(offense="SEA", defense="NE", home="SEA", away="NE", season=2014, playoff=True,
                score={"SEA": 24, "NE": 28}, timeouts={"SEA": 1, "NE": 2}, half=2, secs=26,
                yardline_100=1, down=2, ydstogo=1, second_half_receiver="SEA")
    base.update(kw)
    return GameState(**base)


def eng(model, seed=0):
    return Engine(model, random.Random(seed))


def test_gain_first_down_td_safety_downs(model):
    e = eng(model)
    s = state(yardline_100=50, down=1, ydstogo=10)
    e.gain(s, 12)
    assert (s.yardline_100, s.down, s.ydstogo) == (38, 1, 10)
    s = state()
    e.gain(s, 1)
    assert s.phase == PAT and s.score["SEA"] == 30
    s = state(yardline_100=99, down=1, ydstogo=10)
    e.gain(s, -2)
    assert s.score["NE"] == 30 and s.phase == KICKOFF and s.free_kick
    s = state(down=4)
    e.gain(s, 0)
    assert s.offense == "NE" and s.yardline_100 == 99 and not s.mid_drive


def test_penalties(model):
    e = eng(model)
    s = state(yardline_100=4, down=3, ydstogo=4)
    e.penalty(s, on_offense=False, yards=15)  # half the distance
    assert s.yardline_100 == 2 and s.down == 3 and s.ydstogo == 2
    e.penalty(s, on_offense=False, yards=5, automatic_first_down=True)
    assert s.down == 1 and s.ydstogo == 1
    s = state(yardline_100=30, down=2, ydstogo=6)
    e.penalty(s, on_offense=True, yards=10)
    assert s.yardline_100 == 40 and s.down == 2 and s.ydstogo == 16


def test_turnovers(model):
    e = eng(model)
    s = state(yardline_100=10)
    e.turnover(s, net=15)
    assert s.offense == "NE" and s.yardline_100 == 80
    s = state(yardline_100=40)
    e.turnover(s, defense_td=True)
    assert s.offense == "NE" and s.phase == PAT


def test_modified_overtime(model):
    e = eng(model)
    s = state(half=3, secs=900, score={"SEA": 28, "NE": 28}, yardline_100=20, down=4, ot_stage=0)
    e.field_goal(s, True)
    assert s.ot_stage == 1
    s.swap()
    s.phase, s.yardline_100 = SCRIMMAGE, 70
    with pytest.raises(GameOver) as end:
        e.punt(s, 40)
    assert end.value.winner == "SEA"


def test_sudden_death_before_2012(model):
    e = eng(model)
    s = state(season=2011, playoff=False, half=3, secs=900, score={"SEA": 28, "NE": 28}, yardline_100=20)
    with pytest.raises(GameOver) as end:
        e.field_goal(s, True)
    assert end.value.winner == "SEA"


def test_regular_season_overtime_can_tie(model):
    e = eng(model)
    s = state(season=2019, playoff=False, half=3, secs=0, score={"SEA": 20, "NE": 20})
    with pytest.raises(GameOver) as end:
        e.end_period(s)
    assert end.value.winner is None


def test_games_always_finish(model):
    e = eng(model)
    for seed in range(30):
        e.rng = random.Random(seed)
        s = state(phase=KICKOFF, half=1, secs=1800, score={"SEA": 0, "NE": 0}, mid_drive=False)
        r = e.play_out(s, record=True)
        assert r.winner in ("SEA", "NE") and r.path[-1][1][0] == "END"


def test_drive_end_of_half_needs_time_to_run_out(model):
    m = model
    m.drives[("early", 4)] = [("END", 30.0, 0)] * 30 + [("PUNT", 150.0, 70)]
    m._cache.clear()
    e = eng(m)
    s = state(half=1, secs=1500, yardline_100=75, mid_drive=False)
    e.drive(s)
    assert s.secs > 0 and s.offense == "NE"


def test_parse_helpers():
    assert parse_yardline("NE 1", "SEA", "NE") == 1
    assert parse_yardline("SEA 20", "SEA", "NE") == 80
    assert parse_yardline("own 20", "SEA", "NE") == 80
    assert parse_yardline("opp 35", "SEA", "NE") == 35
    assert parse_yardline(50, "SEA", "NE") == 50
    assert parse_game_clock("Q4 0:20") == (4, 20)
    assert parse_game_clock("4 :26") == (4, 26)
    assert parse_game_clock("OT 3:00") == (5, 180)


def test_edit_outcomes_and_overrides(model):
    pre = state()
    post, winner, over = apply_edit(pre, Edit("touchdown"), model, 6.0)
    assert not over and post.phase == PAT and post.score["SEA"] == 30 and post.secs == 20

    post, _, _ = apply_edit(pre, Edit("run"), model, 6.0)
    assert post.forced_call == "run" and post.mid_drive and post.secs == 26

    post, _, _ = apply_edit(pre, Edit("incomplete", clock_secs=4, timeouts={"SEA": 0}), model, 6.0)
    assert post.down == 3 and post.secs == 22 and post.timeouts["SEA"] == 0

    post, _, _ = apply_edit(pre, Edit("custom", possession="NE", yardline="NE 20", down=1, ydstogo=10,
                                      score={"SEA": 31, "NE": 28}, clock="Q4 0:20"), model, 6.0)
    assert post.offense == "NE" and post.yardline_100 == 80 and post.score["SEA"] == 31 and post.secs == 20

    with pytest.raises(ValueError):
        apply_edit(pre, Edit("gain"), model, 6.0)


def test_edit_that_ends_overtime(model):
    pre = state(half=3, secs=300, score={"SEA": 28, "NE": 28}, yardline_100=10)
    _, winner, over = apply_edit(pre, Edit("touchdown"), model, 6.0)
    assert over and winner == "SEA"
