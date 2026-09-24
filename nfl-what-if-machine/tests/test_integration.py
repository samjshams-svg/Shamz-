"""Uses the real 2010s data (downloaded on first run, then cached)."""

import os

import pytest

from whatif import data
from whatif.rewrite import Edit, rewrite_game

pytestmark = pytest.mark.skipif(os.environ.get("WHATIF_OFFLINE") == "1", reason="needs nflverse data")


def test_find_the_butler_interception():
    game = data.load_game("2014_21_NE_SEA")
    hits = data.find_plays(game, qtr=4, clock="0:26", search="INTERCEPTED")
    assert list(hits["play_id"]) == [4205]


def test_sb49_lynch_handoff():
    rw = rewrite_game("2014_21_NE_SEA", 4205, Edit("run"), n=4000, compare=False, record=False)
    assert rw.pre.describe().startswith("SEA 24 - NE 28 | Q4 0:26 | SEA ball, 2&1 at NE 1")
    assert 0.65 < rw.alt.prob("SEA") < 0.9
    assert rw.alt.prob("SEA") + rw.alt.prob("NE") + rw.alt.tie_prob == pytest.approx(1)
