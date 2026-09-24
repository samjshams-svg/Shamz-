import random

import pytest

from sb49.game import Override, apply_override
from sb49.model import LeagueModel
from sb49.simulate import Engine, GameOver
from sb49.state import KICKOFF, PAT, SCRIMMAGE, GameState, game_mode


@pytest.fixture
def model():
    m = LeagueModel()
    plays = [(3, 0, 0, 0, 0), (0, 0, 0, 0, 1), (8, 0, 0, 0, 0)] * 20
    for t in range(5):
        m.scrimmage[(None, t, None, None)] = plays
    for mode in ("normal", "hurry", "protect"):
        m.durations[(mode, 0)] = [30.0]
        m.durations[(mode, 1)] = [5.0]
    m.kickoffs = [(75, 0, 5.0)]
    m.punts = {z: [(40, 0, 8.0)] for z in range(4)}
    return m


def state(**kw):
    base = dict(offense="SEA", defense="NE", score={"SEA": 24, "NE": 28}, timeouts={"SEA": 1, "NE": 2},
                half=2, secs=26, yardline_100=1, down=2, ydstogo=1, second_half_receiver="SEA")
    base.update(kw)
    return GameState(**base)


def engine(model):
    return Engine(model, random.Random(0))


def test_gain_first_down_td_safety_downs(model):
    e = engine(model)
    s = state(yardline_100=50, down=1, ydstogo=10)
    e.gain(s, 12)
    assert (s.yardline_100, s.down, s.ydstogo) == (38, 1, 10)
    e.gain(s, 3)
    assert (s.down, s.ydstogo) == (2, 7)

    s = state()
    e.gain(s, 1)
    assert s.phase == PAT and s.score["SEA"] == 30

    s = state(yardline_100=99, down=1, ydstogo=10)
    e.gain(s, -2)
    assert s.score["NE"] == 30 and s.phase == KICKOFF and s.offense == "SEA" and s.free_kick

    s = state(down=4)
    e.gain(s, 0)
    assert s.offense == "NE" and s.yardline_100 == 99 and s.down == 1


def test_turnovers(model):
    e = engine(model)
    s = state(yardline_100=10)
    e.turnover(s, net=15)  # intercepted 5 yards deep in the end zone -> touchback
    assert s.offense == "NE" and s.yardline_100 == 80

    s = state(yardline_100=40)
    e.turnover(s, net=0, defense_td=True)
    assert s.offense == "NE" and s.phase == PAT and s.score["NE"] == 34


def test_two_minute_warning(model):
    e = engine(model)
    s = state(secs=130)
    e.run_clock(s, 30)
    assert s.secs == 120


def test_kneel_out_when_leading(model):
    e = engine(model)
    s = state(offense="NE", defense="SEA", score={"SEA": 24, "NE": 28}, yardline_100=80, down=1, ydstogo=10,
              secs=60, timeouts={"SEA": 1, "NE": 2})
    assert e.should_kneel(s)
    assert e.play_out(s).winner == "NE"


def test_overtime_fg_then_stop_wins(model):
    e = engine(model)
    s = state(half=3, secs=900, score={"SEA": 28, "NE": 28}, yardline_100=20, down=4, ydstogo=5, ot_stage=0)
    e.field_goal(s, True)
    assert s.ot_stage == 1 and s.phase == KICKOFF
    s.swap()  # NE now has the ball
    s.phase = SCRIMMAGE
    s.yardline_100, s.down = 70, 4
    with pytest.raises(GameOver) as end:
        e.punt(s, 40)
    assert end.value.winner == "SEA"


def test_tied_game_goes_to_overtime_and_ends(model):
    e = engine(model)
    for seed in range(20):
        e.rng = random.Random(seed)
        s = state(score={"SEA": 28, "NE": 28}, secs=0, phase=KICKOFF)
        result = e.play_out(s)
        assert result.winner in ("SEA", "NE")
        assert result.score["SEA"] != result.score["NE"]


def test_override_touchdown(model):
    post, winner = apply_override(state(), Override("touchdown"), 6.0, model)
    assert winner is None and post.phase == PAT and post.score["SEA"] == 30 and post.secs == 20


def test_override_requires_yards(model):
    with pytest.raises(ValueError):
        apply_override(state(), Override("gain"), 6.0, model)


def test_game_mode():
    assert game_mode(1, 100, 0) == "hurry"
    assert game_mode(2, 250, -4) == "hurry"
    assert game_mode(2, 250, 4) == "protect"
    assert game_mode(2, 1000, -4) == "normal"
