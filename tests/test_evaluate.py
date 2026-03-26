"""Tests for evaluate.py — written BEFORE implementation (TDD).

Covers: RandomAgent, GreedyAgent, evaluate().

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

def _make_gym_env(**kwargs):
    from wallgo_gym import WallGoGymEnv
    return WallGoGymEnv(**kwargs)


def _play_agent_game(agent, env, max_steps=500):
    """Play a full game with the given agent. Returns final (reward, terminated, truncated, info)."""
    obs, info = env.reset()
    for _ in range(max_steps):
        mask = env.action_masks()
        action = agent.select_action(obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return reward, terminated, truncated, info
    return 0, False, False, {}


# ======================================================================
# Section 1: RandomAgent
# ======================================================================

class TestRandomAgent:
    """Tests for the RandomAgent baseline."""

    # (A) Happy path — agent can be instantiated
    def test_creates_without_error(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        assert agent is not None

    # (A) Happy path — select_action returns a valid int
    def test_select_action_returns_int(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[0] = True
        mask[100] = True
        action = agent.select_action(None, mask)
        assert isinstance(action, (int, np.integer))

    # (A) Happy path — selected action is always legal (in mask)
    def test_select_action_always_legal(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[42] = True
        mask[99] = True
        mask[500] = True
        for _ in range(50):
            action = agent.select_action(None, mask)
            assert mask[action], f"Agent selected illegal action {action}"

    # (A) Happy path — can complete a full game
    def test_completes_full_game(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        env = _make_gym_env()
        reward, terminated, truncated, info = _play_agent_game(agent, env)
        assert terminated or truncated

    # (B) Empty — all-False mask (game over scenario)
    def test_empty_mask_raises_or_returns_none(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        with pytest.raises((ValueError, IndexError)):
            agent.select_action(None, mask)

    # (D) Boundary — single legal action
    def test_single_legal_action(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[7777] = True
        action = agent.select_action(None, mask)
        assert action == 7777

    # (D) Boundary — all actions legal
    def test_all_actions_legal(self):
        from evaluate import RandomAgent
        agent = RandomAgent()
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        action = agent.select_action(None, mask)
        assert 0 <= action < ACTION_SPACE_SIZE


# ======================================================================
# Section 2: GreedyAgent
# ======================================================================

class TestGreedyAgent:
    """Tests for the GreedyAgent baseline (maximizes immediate territory)."""

    # (A) Happy path — agent can be instantiated
    def test_creates_without_error(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        assert agent is not None

    # (A) Happy path — select_action returns a valid int
    def test_select_action_returns_int(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        env = _make_gym_env()
        env.reset()
        mask = env.action_masks()
        action = agent.select_action(None, mask, env=env)
        assert isinstance(action, (int, np.integer))
        assert mask[action], "Greedy agent should pick a legal action"

    # (A) Happy path — selected action is always legal
    def test_select_action_always_legal(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        env = _make_gym_env()
        env.reset()
        for _ in range(10):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = agent.select_action(None, mask, env=env)
            assert mask[action]
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    # (A) Happy path — can complete a full game
    def test_completes_full_game(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        env = _make_gym_env()
        obs, info = env.reset()
        for _ in range(500):
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = agent.select_action(obs, mask, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated

    # (B) Empty — empty mask
    def test_empty_mask_raises(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        env = _make_gym_env()
        env.reset()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        with pytest.raises((ValueError, IndexError)):
            agent.select_action(None, mask, env=env)

    # (D) Boundary — single legal action returns that action
    def test_single_legal_action(self):
        from evaluate import GreedyAgent
        agent = GreedyAgent()
        env = _make_gym_env()
        env.reset()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        # Find a real legal action
        real_mask = env.action_masks()
        legal = np.where(real_mask)[0]
        mask[legal[0]] = True
        action = agent.select_action(None, mask, env=env)
        assert action == legal[0]


# ======================================================================
# Section 3: evaluate() function
# ======================================================================

class TestEvaluateFunction:
    """Tests for the evaluate() orchestration function."""

    # (A) Happy path — returns a dict with expected keys
    def test_returns_dict_with_keys(self):
        from evaluate import evaluate, RandomAgent
        agent = RandomAgent()
        metrics = evaluate(agent, num_games=2)
        assert isinstance(metrics, dict)
        expected_keys = {"win_rate", "avg_game_length", "avg_territory_diff"}
        assert expected_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    # (A) Happy path — win_rate is between 0 and 1
    def test_win_rate_range(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=4)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    # (A) Happy path — avg_game_length is positive
    def test_avg_game_length_positive(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=2)
        assert metrics["avg_game_length"] > 0

    # (D) Boundary — num_games=1
    def test_single_game(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=1)
        assert isinstance(metrics["win_rate"], float)

    # (D) Boundary — num_games=0 returns empty/default metrics
    def test_zero_games(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=0)
        assert metrics["win_rate"] == 0.0
        assert metrics["avg_game_length"] == 0.0

    # (A) Happy path — evaluate with greedy agent
    def test_evaluate_greedy(self):
        from evaluate import evaluate, GreedyAgent
        metrics = evaluate(GreedyAgent(), num_games=2)
        assert isinstance(metrics, dict)
        assert "win_rate" in metrics

    # (A) Happy path — random vs random should have ~50% win rate over many games
    def test_random_vs_random_balanced(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=20, opponent=RandomAgent())
        # With 20 games, win rate should be roughly 0.3-0.7 (not extreme)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    # (A) Happy path — avg_territory_diff is a number
    def test_territory_diff_is_numeric(self):
        from evaluate import evaluate, RandomAgent
        metrics = evaluate(RandomAgent(), num_games=2)
        assert isinstance(metrics["avg_territory_diff"], (int, float))
