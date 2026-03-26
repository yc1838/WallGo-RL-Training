"""
End-to-end test suite for the WallGo training loop.

Tests cover:
  1. Step target computation (unit)
  2. SB3 model.learn behavior with reset_num_timesteps=False (integration)
  3. Full training loop: checkpoint saving, eval triggering (e2e)
  4. Resume scenario: loop enters and advances (regression)
  5. Edge cases: zero steps, already-past target, boundary values

Run:  python tests/test_training_e2e.py
"""

import os
import sys
import re
import shutil
import tempfile
import time

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from train import SelfPlayEnv, WallGoCNN, _mask_fn

# ======================================================================
# Test infrastructure
# ======================================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            print(f"  ✅ {name}")
            self.passed += 1
        else:
            msg = f"  ❌ FAIL: {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)
            self.failed += 1
            self.errors.append(name)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


def _create_tiny_model():
    """Create a tiny MaskablePPO model for testing (fast to train)."""
    env = ActionMasker(SelfPlayEnv(reward_shaping=True), _mask_fn)
    policy_kwargs = dict(
        features_extractor_class=WallGoCNN,
        features_extractor_kwargs=dict(features_dim=32),
        net_arch=[32]
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        n_steps=64,      # very small for speed
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        verbose=0,
        device="cpu",     # CPU for test reliability
    )
    return model, env


# ======================================================================
# TEST 1: Step target computation
# ======================================================================

def test_step_target_computation(R: TestResult):
    """Test the logic that computes the absolute training target."""
    print("\n--- TEST 1: Step Target Computation ---")

    # The fixed logic from train.py:
    # if resume and current_steps > 0:
    #     total_timesteps = steps_done + total_timesteps

    def compute_target(steps_arg, current_steps, resume):
        if resume and current_steps > 0:
            return current_steps + steps_arg
        return steps_arg

    # Happy path
    R.check("fresh start: 1M → 1M",
            compute_target(1_000_000, 0, False) == 1_000_000)
    R.check("resume 9.5M + 1M → 10.5M",
            compute_target(1_000_000, 9_584_640, True) == 10_584_640)
    R.check("resume 100k + 500k → 600k",
            compute_target(500_000, 100_000, True) == 600_000)

    # Edge cases
    R.check("resume + 0 steps → stay put",
            compute_target(0, 5_000_000, True) == 5_000_000)
    R.check("resume flag but no checkpoint → fresh",
            compute_target(1_000_000, 0, True) == 1_000_000)
    R.check("fresh + 0 steps → 0",
            compute_target(0, 0, False) == 0)

    # Boundary: while loop condition
    steps_done = 9_584_640
    old_target = 1_000_000  # BUG: absolute target
    new_target = compute_target(1_000_000, steps_done, True)  # FIX: additive
    R.check("OLD logic: loop would NOT run",
            not (steps_done < old_target),
            f"{steps_done:,} < {old_target:,} should be False")
    R.check("NEW logic: loop DOES run",
            steps_done < new_target,
            f"{steps_done:,} < {new_target:,} should be True")


# ======================================================================
# TEST 2: SB3 model.learn behavior
# ======================================================================

def test_model_learn_advances_steps(R: TestResult):
    """Does model.learn actually increment num_timesteps?"""
    print("\n--- TEST 2: model.learn Step Advancement ---")

    model, env = _create_tiny_model()

    # 2a: Fresh model starts at 0
    R.check("fresh model starts at 0",
            model.num_timesteps == 0,
            f"got {model.num_timesteps}")

    # 2b: Learn with reset=True advances steps
    model.learn(total_timesteps=128, reset_num_timesteps=True)
    steps_after_first = model.num_timesteps
    R.check("learn(128, reset=True) advances steps",
            steps_after_first > 0,
            f"num_timesteps={steps_after_first}")

    # 2c: Learn with reset=False and target > current SHOULD advance
    before = model.num_timesteps
    target = before + 128
    print(f"    [info] Calling model.learn(total_timesteps={target}, reset=False), current={before}")
    model.learn(total_timesteps=target, reset_num_timesteps=False)
    after = model.num_timesteps
    R.check("learn(current+128, reset=False) advances",
            after > before,
            f"before={before}, after={after}, diff={after-before}")

    # 2d: Learn with reset=False and target <= current should NOT advance (returns immediately)
    before = model.num_timesteps
    small_target = before - 100
    model.learn(total_timesteps=small_target, reset_num_timesteps=False)
    after = model.num_timesteps
    R.check("learn(target < current, reset=False) does not advance",
            after == before,
            f"before={before}, after={after}")

    # 2e: CRITICAL TEST — simulate the exact resume scenario
    # Save model, reload, then learn with reset=False using the additive target
    print("    [info] Testing save → reload → learn cycle...")
    tmpdir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(tmpdir, "test_model")
        model.save(save_path)
        saved_steps = model.num_timesteps

        # Reload
        loaded = MaskablePPO.load(save_path, env=env, device="cpu")
        R.check("loaded model preserves num_timesteps",
                loaded.num_timesteps == saved_steps,
                f"saved={saved_steps}, loaded={loaded.num_timesteps}")

        # Learn with additive target
        additive_target = loaded.num_timesteps + 128
        loaded.learn(total_timesteps=additive_target, reset_num_timesteps=False)
        R.check("loaded model advances with additive target",
                loaded.num_timesteps > saved_steps,
                f"saved={saved_steps}, after_learn={loaded.num_timesteps}")
    finally:
        shutil.rmtree(tmpdir)


# ======================================================================
# TEST 3: Full training loop — checkpoint saving
# ======================================================================

def test_training_saves_checkpoints(R: TestResult):
    """Does the training loop actually save checkpoint .zip files?"""
    print("\n--- TEST 3: Checkpoint Saving ---")

    tmpdir = tempfile.mkdtemp()
    try:
        # Import the train function
        from train import train

        # Run a very short training (no resume, no UI)
        print("    [info] Running train(steps=200, save_interval=64, no_ui=True)...")
        train(
            total_timesteps=200,
            save_dir=tmpdir,
            save_interval=64,
            learning_rate=3e-4,
            reward_shaping=True,
            eval_interval=9999999,  # don't eval (too slow)
            eval_games=0,
            n_envs=1,
            resume=False,
            no_ui=True,
        )

        # Check that checkpoints were saved
        zips = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
        R.check("at least 1 checkpoint .zip saved",
                len(zips) >= 1,
                f"found {len(zips)} zips: {zips}")
        R.check("wallgo_final.zip exists",
                os.path.exists(os.path.join(tmpdir, "wallgo_final.zip")),
                f"files: {os.listdir(tmpdir)}")

        # Check stats.json
        stats_path = os.path.join(tmpdir, "stats.json")
        R.check("stats.json exists",
                os.path.exists(stats_path))

        # Extract step numbers from filenames
        steps_saved = []
        for f in zips:
            m = re.search(r'wallgo_(\d+)\.zip', f)
            if m:
                steps_saved.append(int(m.group(1)))
        steps_saved.sort()
        if steps_saved:
            R.check("first checkpoint step > 0",
                    steps_saved[0] > 0,
                    f"steps: {steps_saved}")
            print(f"    [info] Checkpoints saved at steps: {steps_saved}")
    finally:
        shutil.rmtree(tmpdir)


# ======================================================================
# TEST 4: Resume scenario — loop enters and saves more checkpoints
# ======================================================================

def test_resume_saves_more_checkpoints(R: TestResult):
    """After resume, does the loop enter and save NEW checkpoints?"""
    print("\n--- TEST 4: Resume Scenario ---")

    tmpdir = tempfile.mkdtemp()
    try:
        from train import train

        # Phase 1: initial training
        print("    [info] Phase 1: Initial training (200 steps)...")
        train(
            total_timesteps=200,
            save_dir=tmpdir,
            save_interval=64,
            learning_rate=3e-4,
            reward_shaping=True,
            eval_interval=9999999,
            eval_games=0,
            n_envs=1,
            resume=False,
            no_ui=True,
        )
        zips_phase1 = set(f for f in os.listdir(tmpdir) if f.endswith(".zip") and f != "wallgo_final.zip")
        print(f"    [info] Phase 1 checkpoints: {sorted(zips_phase1)}")

        # Phase 2: resume and train MORE
        print("    [info] Phase 2: Resume training (200 MORE steps)...")
        train(
            total_timesteps=200,   # 200 MORE steps (not absolute!)
            save_dir=tmpdir,
            save_interval=64,
            learning_rate=3e-4,
            reward_shaping=True,
            eval_interval=9999999,
            eval_games=0,
            n_envs=1,
            resume=True,
            no_ui=True,
        )
        zips_phase2 = set(f for f in os.listdir(tmpdir) if f.endswith(".zip") and f != "wallgo_final.zip")
        new_checkpoints = zips_phase2 - zips_phase1
        print(f"    [info] Phase 2 NEW checkpoints: {sorted(new_checkpoints)}")

        R.check("phase 2 created NEW checkpoints",
                len(new_checkpoints) > 0,
                f"phase1={len(zips_phase1)}, phase2={len(zips_phase2)}, new={len(new_checkpoints)}")

        # Check that new checkpoints have higher step numbers
        if new_checkpoints:
            max_phase1 = max(int(re.search(r'(\d+)', f).group()) for f in zips_phase1) if zips_phase1 else 0
            min_phase2_new = min(int(re.search(r'(\d+)', f).group()) for f in new_checkpoints)
            R.check("new checkpoint steps > old checkpoint steps",
                    min_phase2_new > max_phase1,
                    f"max_old={max_phase1}, min_new={min_phase2_new}")
    finally:
        shutil.rmtree(tmpdir)


# ======================================================================
# TEST 5: Eval triggering
# ======================================================================

def test_eval_triggers(R: TestResult):
    """Does the eval callback fire at the right interval?"""
    print("\n--- TEST 5: Eval Triggering ---")

    tmpdir = tempfile.mkdtemp()
    try:
        from train import train
        import io
        from contextlib import redirect_stdout

        # Capture stdout to check for eval output
        buf = io.StringIO()
        print("    [info] Running training with eval_interval=64...")
        # We need to capture the eval output
        import builtins
        original_print = builtins.print
        captured_lines = []
        def capturing_print(*args, **kwargs):
            line = " ".join(str(a) for a in args)
            captured_lines.append(line)
            original_print(*args, **kwargs)

        builtins.print = capturing_print
        try:
            train(
                total_timesteps=200,
                save_dir=tmpdir,
                save_interval=64,
                learning_rate=3e-4,
                reward_shaping=True,
                eval_interval=64,
                eval_games=5,     # just 5 games, fast
                n_envs=1,
                resume=False,
                no_ui=True,
            )
        finally:
            builtins.print = original_print

        # Check for eval output
        eval_lines = [l for l in captured_lines if "Eval vs Random" in l]
        R.check("eval output appeared at least once",
                len(eval_lines) >= 1,
                f"found {len(eval_lines)} eval lines")
        if eval_lines:
            print(f"    [info] Eval lines: {eval_lines[0][:80]}...")

        checkpoint_lines = [l for l in captured_lines if "Checkpoint saved" in l]
        R.check("checkpoint save message appeared",
                len(checkpoint_lines) >= 1,
                f"found {len(checkpoint_lines)} checkpoint lines")
    finally:
        shutil.rmtree(tmpdir)


# ======================================================================
# TEST 6: n_envs > 1 (multi-env)
# ======================================================================

def test_multi_env_training(R: TestResult):
    """Does training work with multiple parallel environments?"""
    print("\n--- TEST 6: Multi-Env Training ---")

    tmpdir = tempfile.mkdtemp()
    try:
        from train import train

        print("    [info] Running training with n_envs=2...")
        train(
            total_timesteps=200,
            save_dir=tmpdir,
            save_interval=64,
            learning_rate=3e-4,
            reward_shaping=True,
            eval_interval=9999999,
            eval_games=0,
            n_envs=2,
            resume=False,
            no_ui=True,
        )

        zips = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
        R.check("multi-env: checkpoints saved",
                len(zips) >= 1,
                f"found {len(zips)} zips")
    finally:
        shutil.rmtree(tmpdir)


# ======================================================================
# TEST 7: Chunk size is never negative or zero mid-loop
# ======================================================================

def test_chunk_arithmetic(R: TestResult):
    """Verify chunk arithmetic stays correct across various scenarios."""
    print("\n--- TEST 7: Chunk Arithmetic ---")

    scenarios = [
        ("fresh start", 0, 1000, 100),
        ("resume mid", 500, 1500, 100),     # after additive: target=1500
        ("last chunk partial", 950, 1000, 100),  # chunk should be 50
        ("exact boundary", 1000, 1000, 100),     # should NOT enter loop
    ]

    for name, steps_done, total_timesteps, save_interval in scenarios:
        if steps_done < total_timesteps:
            chunk = min(save_interval, total_timesteps - steps_done)
            R.check(f"chunk({name})={chunk} > 0",
                    chunk > 0,
                    f"steps_done={steps_done}, total={total_timesteps}")
            learn_target = steps_done + chunk
            R.check(f"learn_target({name})={learn_target} > steps_done",
                    learn_target > steps_done)
        else:
            R.check(f"loop skipped({name}): correct",
                    steps_done >= total_timesteps)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  WallGo Training Loop — End-to-End Test Suite")
    print("=" * 60)

    R = TestResult()

    t0 = time.time()

    # Unit tests (fast)
    test_step_target_computation(R)
    test_chunk_arithmetic(R)

    # Integration tests (need model)
    test_model_learn_advances_steps(R)

    # E2E tests (run actual training, slower)
    test_training_saves_checkpoints(R)
    test_resume_saves_more_checkpoints(R)
    test_eval_triggers(R)
    test_multi_env_training(R)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    ok = R.summary()
    sys.exit(0 if ok else 1)
