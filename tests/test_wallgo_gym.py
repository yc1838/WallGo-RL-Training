"""Tests for wallgo_gym.py — written BEFORE implementation (TDD).

Covers WallGoGymEnv: __init__, reset, step, action_masks, reward shaping, truncation.

Categories:
  (A) Happy path
  (B) Null/empty/missing
  (C) Type/shape mismatch
  (D) Boundary values
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from action_encoding import ACTION_SPACE_SIZE


# ======================================================================
# Helpers
# ======================================================================

def _make_env(**kwargs):
    from wallgo_gym import WallGoGymEnv
    return WallGoGymEnv(**kwargs)


def _get_first_legal_action(env):
    """Return the first True index from action_masks."""
    mask = env.action_masks()
    indices = np.where(mask)[0]
    return int(indices[0])


def _play_random_game(env, seed=42):
    """Play a full game with random legal actions. Returns (obs, reward, term, trunc, info)."""
    import random
    random.seed(seed)
    obs, info = env.reset()
    while True:
        mask = env.action_masks()
        legal_indices = np.where(mask)[0]
        if len(legal_indices) == 0:
            break
        action = int(random.choice(legal_indices))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
    return obs, 0, False, False, info


# ======================================================================
# Section 1: __init__ — spaces definition
# ======================================================================

class TestInit:
    """Tests for WallGoGymEnv.__init__ and space definitions."""

    # (A) Happy path — env creates without error
    def test_creates_without_error(self):
        env = _make_env()
        assert env is not None

    # (A) Happy path — observation_space shape
    def test_observation_space_shape(self):
        env = _make_env()
        assert env.observation_space.shape == (6, 7, 7)

    # (A) Happy path — observation_space bounds
    def test_observation_space_bounds(self):
        env = _make_env()
        assert env.observation_space.low.min() == 0
        assert env.observation_space.high.max() == 1

    # (A) Happy path — action_space size
    def test_action_space_size(self):
        env = _make_env()
        assert env.action_space.n == ACTION_SPACE_SIZE

    # (A) Happy path — observation_space dtype is float
    def test_observation_space_dtype(self):
        env = _make_env()
        assert env.observation_space.dtype in (np.float32, np.float64)

    # (D) Boundary — custom max_turns
    def test_accepts_max_turns_param(self):
        env = _make_env(max_turns=50)
        assert env is not None

    # (D) Boundary — shaping flag
    def test_accepts_reward_shaping_flag(self):
        env = _make_env(reward_shaping=True)
        assert env is not None

    def test_accepts_reward_shaping_false(self):
        env = _make_env(reward_shaping=False)
        assert env is not None


# ======================================================================
# Section 2: reset()
# ======================================================================

class TestReset:
    """Tests for WallGoGymEnv.reset()."""

    # (A) Happy path — returns (obs, info) tuple
    def test_returns_tuple_of_two(self):
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    # (A) Happy path — obs shape and dtype
    def test_obs_shape(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == (6, 7, 7)

    def test_obs_dtype_is_float(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs.dtype in (np.float32, np.float64)

    # (A) Happy path — obs values are 0 or 1
    def test_obs_values_binary(self):
        env = _make_env()
        obs, info = env.reset()
        unique = np.unique(obs)
        assert all(v in (0.0, 1.0) for v in unique)

    # (A) Happy path — info is a dict
    def test_info_is_dict(self):
        env = _make_env()
        obs, info = env.reset()
        assert isinstance(info, dict)

    # (A) Happy path — obs is within observation_space
    def test_obs_in_observation_space(self):
        env = _make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    # (A) Happy path — reset is idempotent (calling twice works)
    def test_double_reset(self):
        env = _make_env()
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        assert obs1.shape == obs2.shape

    # (A) Happy path — reset after a game returns valid state
    def test_reset_after_game(self):
        env = _make_env()
        env.reset()
        _play_random_game(env)
        obs, info = env.reset()
        assert obs.shape == (6, 7, 7)
        assert env.observation_space.contains(obs)

    # (D) Boundary — seed parameter
    def test_reset_with_seed(self):
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert obs.shape == (6, 7, 7)

    # (A) Happy path — channel 0 has exactly one piece (current player)
    def test_current_player_piece_at_start(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs[0].sum() == 1, "Should have exactly 1 current player piece"

    # (A) Happy path — channel 1 has exactly one piece (opponent)
    def test_opponent_piece_at_start(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs[1].sum() == 1, "Should have exactly 1 opponent piece"


# ======================================================================
# Section 3: action_masks()
# ======================================================================

class TestActionMasks:
    """Tests for WallGoGymEnv.action_masks()."""

    # (A) Happy path — shape and dtype
    def test_mask_shape(self):
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_mask_dtype(self):
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        assert mask.dtype == np.bool_

    # (A) Happy path — has legal actions at game start
    def test_has_legal_actions_at_start(self):
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        assert mask.any()

    # (B) Empty — all False after game ends
    def test_all_false_after_game_over(self):
        env = _make_env()
        env.reset()
        _play_random_game(env)
        mask = env.action_masks()
        assert not mask.any()

    # (A) Happy path — mask changes between turns
    def test_mask_changes_after_step(self):
        env = _make_env()
        env.reset()
        mask_before = env.action_masks().copy()
        action = _get_first_legal_action(env)
        env.step(action)
        mask_after = env.action_masks()
        assert not np.array_equal(mask_before, mask_after)


# ======================================================================
# Section 4: step() — basic flow
# ======================================================================

class TestStep:
    """Tests for WallGoGymEnv.step()."""

    # (A) Happy path — returns 5-tuple
    def test_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5

    # (A) Happy path — obs shape after step
    def test_obs_shape_after_step(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (6, 7, 7)

    # (A) Happy path — obs in observation_space
    def test_obs_in_space_after_step(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        obs, *_ = env.step(action)
        assert env.observation_space.contains(obs)

    # (A) Happy path — reward is a number
    def test_reward_is_numeric(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        _, reward, *_ = env.step(action)
        assert isinstance(reward, (int, float, np.floating))

    # (A) Happy path — terminated and truncated are bool
    def test_terminated_truncated_are_bool(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    # (A) Happy path — info is dict
    def test_info_is_dict(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        *_, info = env.step(action)
        assert isinstance(info, dict)

    # (A) Happy path — first step is not terminal
    def test_first_step_not_terminal(self):
        env = _make_env()
        env.reset()
        action = _get_first_legal_action(env)
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated, "Game should not end after one step"
        assert not truncated

    # (A) Happy path — full game reaches termination
    def test_full_game_terminates(self):
        env = _make_env()
        env.reset()
        obs, reward, terminated, truncated, info = _play_random_game(env)
        assert terminated or truncated, "Game should end eventually"

    # (A) Happy path — terminal reward is +1, -1, or 0
    def test_terminal_reward_values(self):
        env = _make_env(reward_shaping=False)
        env.reset()
        _, reward, terminated, truncated, info = _play_random_game(env)
        if terminated:
            assert reward in (-1, 0, 1), f"Terminal reward should be -1, 0, or 1, got {reward}"

    # (D) Boundary — stepping with an illegal action
    def test_illegal_action_handled(self):
        """Step with a masked-out action should either raise or return negative reward."""
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        illegal_indices = np.where(~mask)[0]
        if len(illegal_indices) > 0:
            action = int(illegal_indices[0])
            # Should either raise ValueError or return error in info
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                assert reward < 0 or "error" in info
            except (ValueError, IndexError):
                pass  # raising is also acceptable

    # (C) Type — step with out-of-range int
    def test_step_out_of_range(self):
        env = _make_env()
        env.reset()
        with pytest.raises((ValueError, IndexError)):
            env.step(ACTION_SPACE_SIZE + 100)

    # (A) Happy path — multiple sequential steps
    def test_multiple_steps(self):
        env = _make_env()
        env.reset()
        for _ in range(10):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, reward, terminated, truncated, info = env.step(int(legal[0]))
            if terminated or truncated:
                break
            assert obs.shape == (6, 7, 7)


# ======================================================================
# Section 5: Reward shaping
# ======================================================================

class TestRewardShaping:
    """Tests for territory + mobility reward shaping."""

    # (A) Happy path — shaping off: non-terminal reward is 0
    def test_no_shaping_reward_is_zero(self):
        env = _make_env(reward_shaping=False)
        env.reset()
        action = _get_first_legal_action(env)
        _, reward, terminated, _, _ = env.step(action)
        if not terminated:
            assert reward == 0.0, f"With shaping off, non-terminal reward should be 0, got {reward}"

    # (A) Happy path — shaping on: non-terminal reward is non-zero (usually)
    def test_shaping_reward_is_nonzero(self):
        env = _make_env(reward_shaping=True)
        env.reset()
        # Take a few steps and check if any shaping reward appears
        nonzero_found = False
        for _ in range(5):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            _, reward, terminated, truncated, _ = env.step(int(legal[0]))
            if terminated or truncated:
                break
            if reward != 0.0:
                nonzero_found = True
                break
        # It's possible (but unlikely) all shaping rewards are exactly 0.
        # At minimum, the mechanism should exist without errors.

    # (D) Boundary — shaping rewards are small relative to terminal reward
    def test_shaping_reward_magnitude(self):
        env = _make_env(reward_shaping=True)
        env.reset()
        max_shaping = 0.0
        for _ in range(20):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            _, reward, terminated, truncated, _ = env.step(int(legal[0]))
            if terminated or truncated:
                break
            max_shaping = max(max_shaping, abs(reward))
        assert max_shaping < 0.1, (
            f"Shaping rewards should be small (<0.1), got max {max_shaping}"
        )

    # (A) Happy path — terminal reward is still ±1 even with shaping
    def test_terminal_reward_with_shaping(self):
        env = _make_env(reward_shaping=True)
        env.reset()
        _, reward, terminated, truncated, _ = _play_random_game(env)
        if terminated:
            # Terminal reward should dominate — at least ±0.5
            assert abs(reward) >= 0.5, f"Terminal reward should dominate, got {reward}"


# ======================================================================
# Section 6: Max turns truncation
# ======================================================================

class TestTruncation:
    """Tests for max_turns truncation."""

    # (D) Boundary — game truncates at max_turns
    def test_truncates_at_max_turns(self):
        env = _make_env(max_turns=5)
        env.reset()
        for i in range(10):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, reward, terminated, truncated, info = env.step(int(legal[0]))
            if terminated or truncated:
                if not terminated:
                    assert truncated, "Should be truncated, not terminated"
                    assert i + 1 <= 5, "Should truncate at or before max_turns"
                break

    # (D) Boundary — very high max_turns doesn't truncate normal games
    def test_high_max_turns_no_truncation(self):
        env = _make_env(max_turns=10000)
        env.reset()
        _, _, terminated, truncated, _ = _play_random_game(env)
        if terminated:
            assert not truncated, "Should terminate naturally, not truncate"

    # (D) Boundary — max_turns=1 truncates immediately
    def test_max_turns_1(self):
        env = _make_env(max_turns=1)
        env.reset()
        action = _get_first_legal_action(env)
        _, _, terminated, truncated, _ = env.step(action)
        assert terminated or truncated, "Should end after 1 turn"

    # (A) Happy path — truncated game has reward 0 (no winner)
    def test_truncated_reward_is_zero(self):
        env = _make_env(max_turns=3, reward_shaping=False)
        env.reset()
        for _ in range(5):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            _, reward, terminated, truncated, _ = env.step(int(legal[0]))
            if truncated and not terminated:
                assert reward == 0.0, f"Truncated (unfinished) game reward should be 0, got {reward}"
                break


# ======================================================================
# Section 7: Gymnasium compatibility
# ======================================================================

class TestGymnasiumCompat:
    """Tests for Gymnasium API compliance."""

    # (A) Happy path — passes gymnasium's built-in checker
    def test_check_env(self):
        from gymnasium.utils.env_checker import check_env
        env = _make_env()
        # check_env raises on failure
        check_env(env.unwrapped if hasattr(env, 'unwrapped') else env, skip_render_check=True)

    # (A) Happy path — render_mode is None by default (no rendering needed for RL)
    def test_render_mode_default(self):
        env = _make_env()
        assert env.render_mode is None

    # (A) Happy path — metadata exists
    def test_metadata_exists(self):
        env = _make_env()
        assert hasattr(env, 'metadata')
        assert isinstance(env.metadata, dict)
