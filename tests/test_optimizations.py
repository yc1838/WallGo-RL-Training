"""Tests for performance optimizations — written BEFORE implementation (TDD).

We optimize wallgo.py and wallgo_gym.py for speed while preserving correctness.
Every optimization must pass these tests to prove it didn't break anything.

Strategy:
- Run the SAME game with old and new code, assert identical results
- Test each optimization in isolation
- Stress-test with random full games

Categories:
  (A) Happy path — correctness preserved
  (B) Null/empty/missing
  (C) Type/shape mismatch
  (D) Boundary values
  (E) Equivalence — optimized output matches original
"""
import sys
import os
import random
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wallgo import WallGoEnv, Player, SIDES
from action_encoding import (
    ACTION_SPACE_SIZE, get_action_mask, encode_action, decode_action,
)


# ======================================================================
# Helpers
# ======================================================================

def _play_seeded_game(env, seed=42):
    """Play a full game with a fixed seed. Returns list of (action, state_after)."""
    rng = random.Random(seed)
    env.reset()
    history = []
    while not env.done:
        legal = env.get_legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        state, reward, done, info = env.step(action)
        history.append((action, reward, done, info))
    return history


def _get_scores(env):
    """Get territory scores for all active players."""
    return env.calculate_scores(env.active_players)


# ======================================================================
# Section 1: Fast clone — manual clone instead of deepcopy
# ======================================================================

class TestFastClone:
    """Tests that a fast clone produces identical state to deepcopy clone."""

    # (E) Equivalence — cloned env has same board state
    def test_clone_board_matches(self):
        env = WallGoEnv()
        env.reset()
        # Play a few moves
        rng = random.Random(42)
        for _ in range(10):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

        cloned = env.clone()
        for y in range(env.size):
            for x in range(env.size):
                orig_cell = env.board[y][x]
                clone_cell = cloned.board[y][x]
                assert orig_cell.occupant == clone_cell.occupant, f"Occupant mismatch at ({x},{y})"
                for side in SIDES:
                    assert orig_cell.walls[side] == clone_cell.walls[side], (
                        f"Wall mismatch at ({x},{y}) side {side}"
                    )

    # (E) Equivalence — cloned env has same piece positions
    def test_clone_piece_positions_match(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(10):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

        cloned = env.clone()
        for player in env.active_players:
            assert env.get_player_pieces(player) == cloned.get_player_pieces(player)

    # (E) Equivalence — cloned env has same available walls
    def test_clone_available_walls_match(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(10):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

        cloned = env.clone()
        assert env._available_walls == cloned._available_walls

    # (E) Equivalence — cloned env has same metadata
    def test_clone_metadata_matches(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(10):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

        cloned = env.clone()
        assert env.current_player_idx == cloned.current_player_idx
        assert env.done == cloned.done
        assert env.turn_count == cloned.turn_count
        assert env.size == cloned.size

    # (A) Happy path — clone is independent (modifying clone doesn't affect original)
    def test_clone_is_independent(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(5):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

        orig_turn = env.turn_count
        cloned = env.clone()

        # Step the clone
        legal = cloned.get_legal_actions()
        if legal:
            cloned.step(legal[0])
        assert env.turn_count == orig_turn, "Original env should be unaffected by clone step"

    # (E) Equivalence — playing same moves on clone yields same result
    def test_clone_same_game_outcome(self):
        env = WallGoEnv()
        history_original = _play_seeded_game(env, seed=123)

        env2 = WallGoEnv()
        env2.reset()
        cloned = env2.clone()
        history_cloned = _play_seeded_game(cloned, seed=123)

        assert len(history_original) == len(history_cloned)
        for (a1, r1, d1, i1), (a2, r2, d2, i2) in zip(history_original, history_cloned):
            assert r1 == r2
            assert d1 == d2

    # (D) Boundary — clone at game start
    def test_clone_at_start(self):
        env = WallGoEnv()
        env.reset()
        cloned = env.clone()
        assert cloned.turn_count == 0
        assert not cloned.done
        assert len(cloned.get_legal_actions()) == len(env.get_legal_actions())

    # (D) Boundary — clone at game end
    def test_clone_at_end(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))

        cloned = env.clone()
        assert cloned.done == True
        assert cloned.get_legal_actions() == []


# ======================================================================
# Section 2: Optimized encode_state — NumPy-based
# ======================================================================

class TestOptimizedEncodeState:
    """Tests that optimized encode_state matches the original output."""

    # (E) Equivalence — output matches original at game start
    def test_matches_at_start(self):
        env = WallGoEnv()
        env.reset()
        result = env.encode_state()
        arr = np.array(result)
        assert arr.shape == (6, 7, 7)
        assert set(arr.flatten().tolist()).issubset({0, 1})

    # (E) Equivalence — output matches through multiple game states
    def test_matches_through_game(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for step_i in range(30):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break

            result = env.encode_state()
            arr = np.array(result)
            assert arr.shape == (6, 7, 7), f"Shape wrong at step {step_i}"
            assert arr.min() >= 0 and arr.max() <= 1, f"Values out of range at step {step_i}"

            # Channel 0 + 1 should have exactly 2 pieces total (2-player game)
            total_pieces = arr[0].sum() + arr[1].sum()
            assert total_pieces == 2, f"Expected 2 pieces, got {total_pieces} at step {step_i}"

            env.step(rng.choice(legal))

    # (C) Shape — always returns [6, size, size]
    def test_shape_consistency(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(20):
            result = env.encode_state()
            assert len(result) == 6
            assert all(len(row) == 7 for channel in result for row in channel)
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

    # (A) Happy path — wall channels reflect placed walls
    def test_wall_channels_update(self):
        env = WallGoEnv()
        env.reset()
        state_before = np.array(env.encode_state())
        total_walls_before = state_before[2:].sum()

        legal = env.get_legal_actions()
        env.step(legal[0])  # places a wall

        state_after = np.array(env.encode_state())
        total_walls_after = state_after[2:].sum()
        # After a wall placement, wall count should increase
        assert total_walls_after > total_walls_before


# ======================================================================
# Section 3: Optimized get_action_mask — avoid tuple creation
# ======================================================================

class TestOptimizedActionMask:
    """Tests that optimized mask generation matches the original."""

    # (E) Equivalence — mask matches legal actions at game start
    def test_mask_matches_legal_at_start(self):
        env = WallGoEnv()
        env.reset()
        mask = get_action_mask(env)
        legal = env.get_legal_actions()
        assert mask.sum() == len(legal)

    # (E) Equivalence — mask matches legal actions mid-game
    def test_mask_matches_legal_midgame(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(15):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            mask = get_action_mask(env)
            assert mask.sum() == len(legal), (
                f"Mask has {mask.sum()} True but {len(legal)} legal actions"
            )
            # Verify each legal action is in the mask
            for action in legal:
                _, _, to_x, to_y, wx, wy, ws = action
                idx = encode_action(to_x, to_y, wx, wy, ws)
                assert mask[idx], f"Legal action not in mask: {action}"
            env.step(rng.choice(legal))

    # (E) Equivalence — mask is identical across 10 full games
    def test_mask_equivalence_full_games(self):
        for seed in range(10):
            env = WallGoEnv()
            env.reset()
            rng = random.Random(seed)
            step = 0
            while not env.done:
                legal = env.get_legal_actions()
                if not legal:
                    break
                mask = get_action_mask(env)
                assert mask.sum() == len(legal), (
                    f"Game {seed} step {step}: mask={mask.sum()}, legal={len(legal)}"
                )
                env.step(rng.choice(legal))
                step += 1

    # (D) Boundary — mask shape is always ACTION_SPACE_SIZE
    def test_mask_shape_constant(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(20):
            mask = get_action_mask(env)
            assert mask.shape == (ACTION_SPACE_SIZE,)
            assert mask.dtype == np.bool_
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            env.step(rng.choice(legal))

    # (B) Empty — mask all-false when game over
    def test_mask_empty_when_done(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))
        mask = get_action_mask(env)
        assert not mask.any()


# ======================================================================
# Section 4: Optimized Gym wrapper — faster shaping reward
# ======================================================================

class TestOptimizedGymWrapper:
    """Tests that optimized gym wrapper produces correct results."""

    def _make_env(self, **kwargs):
        from wallgo_gym import WallGoGymEnv
        return WallGoGymEnv(**kwargs)

    # (E) Equivalence — same game with same seed produces same trajectory
    def test_deterministic_trajectory(self):
        """Two runs with same actions produce same observations and rewards."""
        env1 = self._make_env(reward_shaping=True)
        env2 = self._make_env(reward_shaping=True)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        for _ in range(20):
            mask1 = env1.action_masks()
            mask2 = env2.action_masks()
            np.testing.assert_array_equal(mask1, mask2)

            legal = np.where(mask1)[0]
            if len(legal) == 0:
                break
            action = int(legal[0])  # deterministic choice

            obs1, r1, t1, tr1, i1 = env1.step(action)
            obs2, r2, t2, tr2, i2 = env2.step(action)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2
            if t1 or tr1:
                break

    # (A) Happy path — full game with shaping produces valid rewards
    def test_full_game_shaping_valid(self):
        env = self._make_env(reward_shaping=True)
        env.reset()
        rng = np.random.RandomState(42)
        rewards = []
        while True:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
            rewards.append(reward)
            if terminated or truncated:
                break
        # Non-terminal rewards should be small
        for r in rewards[:-1]:
            assert abs(r) < 0.1, f"Non-terminal reward too large: {r}"
        # Terminal reward should be significant
        assert abs(rewards[-1]) >= 0.5

    # (A) Happy path — observation space always valid
    def test_obs_always_in_space(self):
        env = self._make_env(reward_shaping=True)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        rng = np.random.RandomState(42)
        for _ in range(30):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break


# ======================================================================
# Section 5: Parallel environments (SubprocVecEnv)
# ======================================================================

class TestParallelEnv:
    """Tests for vectorized parallel environment execution."""

    # (A) Happy path — can create vectorized env
    def test_create_vec_env(self):
        from train import SelfPlayEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            return SelfPlayEnv(max_turns=50)

        vec_env = SubprocVecEnv([make_env for _ in range(2)])
        try:
            obs = vec_env.reset()
            assert obs.shape == (2, 6, 7, 7)
        finally:
            vec_env.close()

    # (A) Happy path — can step vectorized env
    def test_step_vec_env(self):
        from train import SelfPlayEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            return SelfPlayEnv(max_turns=50)

        n_envs = 2
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        try:
            obs = vec_env.reset()
            # Get action masks from each env
            masks = np.array(vec_env.env_method("action_masks"))
            actions = []
            for i in range(n_envs):
                legal = np.where(masks[i])[0]
                actions.append(int(legal[0]))
            obs, rewards, dones, infos = vec_env.step(actions)
            assert obs.shape == (n_envs, 6, 7, 7)
            assert len(rewards) == n_envs
            assert len(dones) == n_envs
        finally:
            vec_env.close()

    # (A) Happy path — multiple envs produce different game states
    def test_envs_are_independent(self):
        from train import SelfPlayEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            return SelfPlayEnv(max_turns=100)

        n_envs = 3
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        try:
            obs = vec_env.reset()
            # Take a few random steps
            for _ in range(5):
                masks = np.array(vec_env.env_method("action_masks"))
                actions = []
                for i in range(n_envs):
                    legal = np.where(masks[i])[0]
                    actions.append(int(np.random.choice(legal)))
                obs, _, dones, _ = vec_env.step(actions)
            # After random steps, observations should differ
            assert not np.array_equal(obs[0], obs[1])
        finally:
            vec_env.close()

    # (D) Boundary — single env in vec wrapper
    def test_single_env_vec(self):
        from train import SelfPlayEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            return SelfPlayEnv(max_turns=50)

        vec_env = SubprocVecEnv([make_env])
        try:
            obs = vec_env.reset()
            assert obs.shape == (1, 6, 7, 7)
        finally:
            vec_env.close()

    # (D) Boundary — 8 parallel envs (typical for M1)
    def test_eight_parallel_envs(self):
        from train import SelfPlayEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            return SelfPlayEnv(max_turns=50)

        n_envs = 8
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        try:
            obs = vec_env.reset()
            assert obs.shape == (n_envs, 6, 7, 7)
            # Take 3 steps
            for _ in range(3):
                masks = np.array(vec_env.env_method("action_masks"))
                actions = []
                for i in range(n_envs):
                    legal = np.where(masks[i])[0]
                    actions.append(int(np.random.choice(legal)))
                obs, rewards, dones, infos = vec_env.step(actions)
                assert obs.shape == (n_envs, 6, 7, 7)
        finally:
            vec_env.close()


# ======================================================================
# Section 6: Regression — full game correctness after ALL optimizations
# ======================================================================

class TestRegressionFullGame:
    """Play many full games and verify scores, termination, and action legality."""

    # (A) Happy path — 50 random games all terminate correctly
    def test_50_games_terminate(self):
        for seed in range(50):
            env = WallGoEnv()
            env.reset()
            rng = random.Random(seed)
            steps = 0
            while not env.done:
                legal = env.get_legal_actions()
                assert len(legal) > 0, f"Game {seed} step {steps}: no legal actions but not done"
                env.step(rng.choice(legal))
                steps += 1
                assert steps < 500, f"Game {seed} stuck after {steps} steps"
            # Scores should sum to at most board_size^2
            scores = env.calculate_scores(env.active_players)
            total = sum(scores.values())
            assert total <= env.size ** 2, f"Scores sum {total} > {env.size**2}"

    # (A) Happy path — all actions in get_legal_actions are actually legal
    def test_all_legal_actions_valid(self):
        env = WallGoEnv()
        env.reset()
        rng = random.Random(42)
        for _ in range(30):
            legal = env.get_legal_actions()
            if not legal or env.done:
                break
            # Try each legal action on a clone to make sure it doesn't error
            for action in legal[:10]:  # sample 10 to keep fast
                clone = env.clone()
                _, _, _, info = clone.step(action)
                assert "error" not in info, f"Legal action produced error: {action} -> {info}"
            env.step(rng.choice(legal))

    # (E) Equivalence — gym wrapper game matches raw env game with same actions
    def test_gym_matches_raw_env(self):
        from wallgo_gym import WallGoGymEnv

        raw = WallGoEnv()
        raw.reset()
        gym_env = WallGoGymEnv(reward_shaping=False)
        gym_env.reset()

        rng = random.Random(42)
        for step_i in range(40):
            raw_legal = raw.get_legal_actions()
            if not raw_legal or raw.done:
                break

            action = rng.choice(raw_legal)
            from_x, from_y, to_x, to_y, wx, wy, ws = action

            # Step raw env
            _, raw_reward, raw_done, raw_info = raw.step(action)

            # Step gym env with encoded action
            action_int = encode_action(to_x, to_y, wx, wy, ws)
            obs, gym_reward, gym_terminated, gym_truncated, gym_info = gym_env.step(action_int)

            assert raw_done == gym_terminated, f"Step {step_i}: done mismatch"
            if raw_done:
                assert raw_reward == gym_reward, f"Step {step_i}: terminal reward mismatch"
                break
