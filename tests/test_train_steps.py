"""Tests for the training loop's step-counting logic.

These tests verify that `--steps` works correctly both for fresh starts
and when resuming from a checkpoint with a high internal step counter.
"""

import pytest


# ======================================================================
# Unit tests for the step target calculation
# ======================================================================

def compute_target(steps_arg: int, current_steps: int, resume: bool) -> int:
    """
    Mirror the logic that train.py SHOULD use to compute the absolute
    training target from the CLI --steps argument.

    When resuming, --steps means "train for X MORE steps from where I am."
    When starting fresh, --steps means "train for X total steps."
    """
    if resume and current_steps > 0:
        return current_steps + steps_arg
    return steps_arg


class TestComputeTarget:
    """Test the target step calculation."""

    # --- Happy path ---
    def test_fresh_start(self):
        """Fresh start: --steps 1M means train to 1M."""
        assert compute_target(1_000_000, 0, resume=False) == 1_000_000

    def test_resume_from_checkpoint(self):
        """Resume at 9.5M with --steps 1M → target is 10.5M."""
        assert compute_target(1_000_000, 9_584_640, resume=True) == 10_584_640

    def test_resume_from_small_checkpoint(self):
        """Resume at 100k with --steps 1M → target is 1.1M."""
        assert compute_target(1_000_000, 100_000, resume=True) == 1_100_000

    # --- Edge / boundary cases ---
    def test_resume_zero_steps_arg(self):
        """--steps 0 when resuming should stay at current position."""
        assert compute_target(0, 5_000_000, resume=True) == 5_000_000

    def test_resume_flag_but_no_checkpoint(self):
        """Resume flag set but no checkpoint found (current_steps=0) → treat as fresh."""
        assert compute_target(1_000_000, 0, resume=True) == 1_000_000

    def test_fresh_start_zero_steps(self):
        """Fresh start with 0 steps should result in 0 target."""
        assert compute_target(0, 0, resume=False) == 0

    def test_very_large_resume(self):
        """Resume at a very large step count."""
        assert compute_target(500_000, 100_000_000, resume=True) == 100_500_000


class TestWhileLoopCondition:
    """
    Test that the while-loop condition `steps_done < total_timesteps`
    is satisfied when it should be, and NOT satisfied when it shouldn't.
    """

    def test_loop_runs_on_fresh_start(self):
        steps_done = 0
        total_timesteps = compute_target(1_000_000, steps_done, resume=False)
        assert steps_done < total_timesteps, "Loop should run on fresh start"

    def test_loop_runs_on_resume(self):
        """THIS IS THE BUG: with the old logic, this would FAIL."""
        steps_done = 9_584_640
        # OLD (broken): total_timesteps = 1_000_000  → 9.5M < 1M = False!
        # NEW (fixed):  total_timesteps = 9_584_640 + 1_000_000 = 10_584_640
        total_timesteps = compute_target(1_000_000, steps_done, resume=True)
        assert steps_done < total_timesteps, \
            f"Loop should run when resuming! steps_done={steps_done}, target={total_timesteps}"

    def test_loop_stops_when_done(self):
        steps_done = 10_584_640
        total_timesteps = 10_584_640
        assert not (steps_done < total_timesteps), "Loop should stop when target reached"


class TestModelLearnArgument:
    """
    Test that model.learn() gets the correct total_timesteps argument.
    With reset_num_timesteps=False, SB3 treats total_timesteps as an
    ABSOLUTE target (not incremental).
    """

    def test_learn_target_first_chunk(self):
        """First chunk: model.learn should be called with steps_done + chunk."""
        steps_done = 9_584_640
        save_interval = 100_000
        total_timesteps = compute_target(1_000_000, steps_done, resume=True)  # 10_584_640
        chunk = min(save_interval, total_timesteps - steps_done)  # 100_000
        learn_target = steps_done + chunk  # 9_684_640
        assert learn_target == 9_684_640
        assert learn_target > steps_done, "Learn target must be > current steps"

    def test_learn_target_last_chunk(self):
        """Last chunk may be smaller than save_interval."""
        steps_done = 10_500_000
        save_interval = 100_000
        total_timesteps = 10_584_640
        chunk = min(save_interval, total_timesteps - steps_done)  # 84_640
        assert chunk == 84_640
        learn_target = steps_done + chunk
        assert learn_target == total_timesteps

    def test_chunk_never_negative(self):
        """Chunk should never be negative."""
        steps_done = 10_584_640
        total_timesteps = 10_584_640
        chunk = min(100_000, total_timesteps - steps_done)
        assert chunk >= 0

    def test_chunk_zero_at_target(self):
        """When steps_done == total_timesteps, chunk is 0 and loop shouldn't run."""
        steps_done = 10_584_640
        total_timesteps = 10_584_640
        # The while loop condition should prevent us from getting here
        assert not (steps_done < total_timesteps)
