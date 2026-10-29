"""WallclockStop: the fork's stop rule and the deadline resolution."""

from __future__ import annotations

import pytest

from marl_workflows.trainer import wallclock as wallclock_mod
from marl_workflows.trainer.wallclock import WallclockStop


def test_wallclock_should_stop():
    # Port of the fork's test_wallclock_should_stop.
    state = WallclockStop(deadline=10_000.0, margin_s=900.0)
    assert not state.has_completed_step
    assert not state.should_stop(now=0.0)
    state.record(step_time=1000.0, iter_time=1100.0)
    assert state.has_completed_step
    # 1.25 * 1100 + 900 = 2275
    assert not state.should_stop(now=7700.0)
    assert state.should_stop(now=7800.0)


def test_record_ema_and_sample_choice():
    state = WallclockStop(deadline=0.0)
    state.record(step_time=None, iter_time=100.0)
    assert state.ema == 100.0 and state.last_iter == 100.0
    state.record(step_time=200.0, iter_time=150.0)  # sample = max(step, iter) = 200
    assert state.ema == pytest.approx(0.3 * 200 + 0.7 * 100)
    assert state.last_iter == 150.0


def test_from_config_disabled():
    assert WallclockStop.from_config(None, env={}) is None
    assert WallclockStop.from_config({}, env={"SLURM_JOB_END_TIME": "123"}) is None
    assert WallclockStop.from_config({"enable": False}, env={"SLURM_JOB_END_TIME": "123"}) is None


def test_from_config_deadline_sources(monkeypatch: pytest.MonkeyPatch):
    stop = WallclockStop.from_config({"enable": True, "deadline_epoch": 5000, "margin_s": 60}, env={})
    assert stop is not None and stop.deadline == 5000.0 and stop.margin_s == 60.0

    stop = WallclockStop.from_config({"enable": True}, env={"SLURM_JOB_END_TIME": "7000"})
    assert stop is not None and stop.deadline == 7000.0 and stop.margin_s == 900.0

    calls = []

    def fake_squeue(job_id: str) -> float:
        calls.append(job_id)
        return 9000.0

    monkeypatch.setattr(wallclock_mod, "_squeue_end_time", fake_squeue)
    stop = WallclockStop.from_config({"enable": True}, env={"SLURM_JOB_END_TIME": "", "SLURM_JOB_ID": "42"})
    assert stop is not None and stop.deadline == 9000.0 and calls == ["42"]


def test_from_config_no_deadline_disables(capsys: pytest.CaptureFixture[str]):
    assert WallclockStop.from_config({"enable": True}, env={}) is None
    assert "disabled" in capsys.readouterr().out


def test_from_omegaconf():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"enable": True, "deadline_epoch": 1.5e9})
    stop = WallclockStop.from_config(cfg, env={})
    assert stop is not None and stop.deadline == 1.5e9
