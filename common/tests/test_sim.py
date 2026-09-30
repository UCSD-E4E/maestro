import numpy as np
import pytest

from maestro_common.arms import Arm
from maestro_common.sim import (
    LearningCurve,
    SimConfig,
    SyntheticEnvironment,
    SyntheticFamily,
    default_environment,
    run,
)


def env_with(*families, efficiency=None, seed=0):
    return SyntheticEnvironment(
        list(families),
        strategy_efficiency=efficiency or {"random": 1.0, "uncertainty": 1.0},
        rng=np.random.default_rng(seed),
    )


def test_learning_curve_starts_at_start_and_approaches_plateau():
    c = LearningCurve(start=0.2, plateau=0.8, rate=100)
    assert c(0) == pytest.approx(0.2)
    assert c(100) < c(1000) < 0.8
    assert c(10_000) == pytest.approx(0.8, abs=1e-3)


def test_playing_an_arm_adds_labels_and_retrains_only_its_family():
    env = env_with(
        SyntheticFamily("a", LearningCurve(0.2, 0.8, 100)),
        SyntheticFamily("b", LearningCurve(0.2, 0.8, 100)),
    )
    env.play(Arm("a", "random"), batch_size=50)
    assert env.labels == 50
    assert env.quality("a") > env.quality("b") == pytest.approx(0.2)


def test_untrainable_family_never_changes():
    env = env_with(SyntheticFamily("zero-shot", LearningCurve(0.5, 0.5, 1), trainable=False))
    env.play(Arm("zero-shot", "random"), batch_size=500)
    assert env.quality("zero-shot") == pytest.approx(0.5)


def test_efficient_strategy_learns_faster_from_the_same_labels():
    curve = LearningCurve(0.2, 0.8, 200)
    fast = env_with(SyntheticFamily("m", curve), efficiency={"uncertainty": 1.5, "random": 1.0})
    slow = env_with(SyntheticFamily("m", curve), efficiency={"uncertainty": 1.5, "random": 1.0})
    fast.play(Arm("m", "uncertainty"), 100)
    slow.play(Arm("m", "random"), 100)
    assert fast.quality("m") > slow.quality("m")


def test_audit_rewards_are_in_unit_interval_around_quality():
    env = env_with(SyntheticFamily("m", LearningCurve(0.6, 0.6, 1)))
    rewards = env.audit_rewards("m", 2000)
    assert rewards.min() >= 0 and rewards.max() <= 1
    assert rewards.mean() == pytest.approx(0.6, abs=0.02)


def test_fixed_arm_run_records_every_round():
    config = SimConfig(rounds=5, batch_size=40, audit_fraction=0.25)
    history = run(default_environment(np.random.default_rng(0)), config, np.random.default_rng(1), fixed_arm="yolo-seg+random")
    assert [h.round for h in history] == [1, 2, 3, 4, 5]
    assert history[-1].labels == 200
    assert {h.arm for h in history} == {"yolo-seg+random"}


def test_bandit_run_is_deterministic_for_a_seed():
    config = SimConfig(rounds=10)
    a = run(default_environment(np.random.default_rng(0)), config, np.random.default_rng(1))
    b = run(default_environment(np.random.default_rng(0)), config, np.random.default_rng(1))
    assert [h.arm for h in a] == [h.arm for h in b]


def test_bandit_settles_on_a_clearly_dominant_arm():
    env = env_with(
        SyntheticFamily("good", LearningCurve(0.8, 0.9, 50)),
        SyntheticFamily("bad", LearningCurve(0.2, 0.3, 50)),
    )
    history = run(env, SimConfig(rounds=40, batch_size=40), np.random.default_rng(3))
    late = [h.arm for h in history[-15:]]
    assert sum(a.startswith("good+") for a in late) >= 13


def test_history_tracks_best_quality_shown_to_labelers():
    history = run(default_environment(np.random.default_rng(0)), SimConfig(rounds=8), np.random.default_rng(0))
    assert all(0 <= h.best_quality <= 1 for h in history)
    assert all(set(h.posterior_means) == set(history[0].posterior_means) for h in history)


def test_compare_summarizes_bandit_and_every_fixed_arm():
    from maestro_common.sim import compare

    results = compare(SimConfig(rounds=6), seeds=3, target=0.6)
    arms = {a.name for a in default_environment(np.random.default_rng(0)).arms()}
    assert set(results) == {"bandit"} | arms
    bandit = results["bandit"]
    assert bandit.curve.shape == (6,)  # mean best quality per round, over seeds
    assert bandit.labels.tolist() == [50 * t for t in range(1, 7)]
    assert 0 <= bandit.final_quality <= 1


def test_cli_writes_csv_and_prints_summary(tmp_path, capsys):
    from maestro_common.sim_cli import main

    main(["--rounds", "5", "--seeds", "2", "--out", str(tmp_path), "--no-plot"])
    out = capsys.readouterr().out
    assert "bandit" in out and "illustrative" in out
    lines = (tmp_path / "curves.csv").read_text().splitlines()
    assert lines[0].startswith("policy,round,labels,best_quality")
    assert len(lines) == 1 + 10 * 5  # bandit + 9 fixed arms, 5 rounds each
