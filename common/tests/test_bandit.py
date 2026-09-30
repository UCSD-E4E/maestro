import numpy as np
import pytest

from maestro_common.bandit import BetaBandit


def test_fractional_update_uses_r():
    b = BetaBandit(["a"], r=2.0)
    b.update("a", 0.75)
    assert (b.arms["a"].alpha, b.arms["a"].beta) == pytest.approx((1 + 1.5, 1 + 0.5))


def test_reward_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        BetaBandit(["a"]).update("a", 1.2)


def test_discount_pulls_toward_prior():
    b = BetaBandit(["a"], gamma=0.5)
    b.arms["a"].alpha, b.arms["a"].beta = 11.0, 3.0
    b.discount()
    assert (b.arms["a"].alpha, b.arms["a"].beta) == pytest.approx((6.0, 2.0))


def test_select_returns_distinct_arms_and_counts_plays():
    b = BetaBandit(["a", "b", "c"])
    chosen = b.select(np.random.default_rng(0), n=2)
    assert len(set(chosen)) == 2
    assert sum(p.plays for p in b.arms.values()) == 2


def test_thompson_converges_on_the_better_arm():
    rng = np.random.default_rng(0)
    b = BetaBandit(["good", "bad"])
    means = {"good": 0.8, "bad": 0.4}
    picks = []
    for _ in range(300):
        (arm,) = b.select(rng)
        picks.append(arm)
        b.update(arm, float(np.clip(rng.normal(means[arm], 0.1), 0, 1)))
    assert picks[-100:].count("good") > 90


def test_save_load_round_trip(tmp_path):
    b = BetaBandit(["a", "b"], r=3.0, gamma=0.9)
    b.update("a", 0.5)
    path = tmp_path / "state" / "bandit.json"
    b.save(path)
    loaded = BetaBandit.load(path)
    assert loaded.to_dict() == b.to_dict()
    assert not list(path.parent.glob(".bandit.json.*"))  # no temp file left behind


def test_sync_arms_adds_new_at_prior_and_keeps_existing():
    b = BetaBandit(["a", "b"])
    b.update("a", 1.0)
    b.sync_arms(["a", "c"])
    assert set(b.arms) == {"a", "c"}
    assert b.arms["a"].alpha == 2.0 and b.arms["c"].alpha == 1.0
