"""Gymnasium-compatible wrapper for WallGo.

Wraps the WallGoEnv game engine into a standard Gymnasium environment with:
- Fixed discrete action space (9,604 actions) with legal action masking
- Shaped observation space [6, 7, 7] as float32
- Optional reward shaping (territory differential + mobility bonus)
- Max-turn truncation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    from wallgo_rs import WallGoEnv, Player
    _USE_RUST = True
except ImportError:
    from wallgo import WallGoEnv, Player
    _USE_RUST = False
from action_encoding import (
    ACTION_SPACE_SIZE, BOARD_SIZE, encode_action, decode_action, get_action_mask,
)


class WallGoGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_turns: int = 200,
        reward_shaping: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_turns = max_turns
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self._env = WallGoEnv(size=BOARD_SIZE)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._env.reset()
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action < 0 or action >= ACTION_SPACE_SIZE:
            raise ValueError(f"Action {action} out of range [0, {ACTION_SPACE_SIZE})")

        # Decode integer action to 7-tuple
        cur = self._env.current_player
        pieces = self._env.get_player_pieces(cur)
        px, py = pieces[0]
        action_tuple = decode_action(action, px, py)

        # Take the step
        state, reward, done, info = self._env.step(action_tuple)

        terminated = done
        truncated = False

        if not terminated and self._env.turn_count >= self.max_turns:
            truncated = True

        # Reward logic
        if terminated:
            # reward from env is +1/-1/0, possibly add shaping on top
            reward = float(reward)
            if self.reward_shaping:
                reward += self._shaping_reward()
        elif truncated:
            reward = 0.0
        else:
            if self.reward_shaping:
                reward = self._shaping_reward()
            else:
                reward = 0.0

        obs = self._encode_obs()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if _USE_RUST:
            return np.asarray(self._env.get_action_mask_np())
        return get_action_mask(self._env)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_obs(self) -> np.ndarray:
        if _USE_RUST:
            return np.asarray(self._env.encode_state_np())
        nested = self._env.encode_state()
        return np.array(nested, dtype=np.float32)

    def _shaping_reward(self) -> float:
        """Small intermediate reward: territory differential + mobility bonus."""
        cur = self._env.current_player
        total_cells = BOARD_SIZE * BOARD_SIZE

        # Territory differential
        scores = self._env.calculate_scores(self._env.active_players)
        my_score = scores.get(cur, 0)
        opp_score = max((s for p, s in scores.items() if p != cur), default=0)
        territory_reward = 0.01 * (my_score - opp_score) / total_cells

        # Mobility bonus
        pieces = self._env.get_player_pieces(cur)
        if pieces:
            moves = self._env.get_valid_moves(pieces[0][0], pieces[0][1])
            max_moves = 13  # theoretical max for 2-step BFS on 7x7
            mobility_reward = 0.005 * len(moves) / max_moves
        else:
            mobility_reward = 0.0

        return territory_reward + mobility_reward
