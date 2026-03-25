"""Evaluation baselines and metrics for WallGo RL training.

Provides:
- RandomAgent: picks a uniformly random legal action
- GreedyAgent: picks the action that maximizes immediate territory
- evaluate(): plays N games between two agents and returns metrics
"""

import argparse
import numpy as np
from typing import Optional, Dict

from wallgo import WallGoEnv, Player
from wallgo_gym import WallGoGymEnv
from action_encoding import (
    ACTION_SPACE_SIZE, decode_action, encode_action, get_action_mask,
)


# ======================================================================
# Baseline agents
# ======================================================================

class RandomAgent:
    """Picks a uniformly random legal action."""

    def select_action(self, obs: Optional[np.ndarray], mask: np.ndarray, **kwargs) -> int:
        legal = np.where(mask)[0]
        if len(legal) == 0:
            raise ValueError("No legal actions available")
        return int(np.random.choice(legal))


class GreedyAgent:
    """Picks the legal action that maximizes immediate territory gain.

    Requires env= keyword argument to access the game state for lookahead.
    """

    def select_action(self, obs: Optional[np.ndarray], mask: np.ndarray, *, env: WallGoGymEnv = None, **kwargs) -> int:
        legal = np.where(mask)[0]
        if len(legal) == 0:
            raise ValueError("No legal actions available")
        if env is None or len(legal) == 1:
            return int(legal[0])

        best_action = int(legal[0])
        best_score = -float('inf')
        inner = env._env
        cur = inner.current_player
        pieces = inner.get_player_pieces(cur)
        px, py = pieces[0]

        # Sample up to 50 actions to keep it fast
        candidates = legal if len(legal) <= 50 else np.random.choice(legal, 50, replace=False)

        for action_int in candidates:
            action_tuple = decode_action(int(action_int), px, py)
            clone = inner.clone()
            _, reward, done, info = clone.step(action_tuple)
            if done:
                score = reward * 100  # heavily weight winning
            else:
                uf = clone._build_union_find()
                scores = clone._scores_with_uf(uf, clone.active_players)
                my_score = scores.get(cur, 0)
                opp_score = max((s for p, s in scores.items() if p != cur), default=0)
                score = my_score - opp_score
            if score > best_score:
                best_score = score
                best_action = int(action_int)

        return best_action


# ======================================================================
# Evaluate function
# ======================================================================

def evaluate(
    agent,
    num_games: int = 10,
    opponent=None,
    max_turns: int = 200,
) -> Dict[str, float]:
    """Play num_games between agent (as player 1) and opponent (as player 2).

    The agent plays on even turns (0, 2, 4, ...) and the opponent on odd turns.
    Since WallGoGymEnv alternates players internally, both agents share the same
    env and each sees the board from the current player's perspective.

    Returns dict with: win_rate, avg_game_length, avg_territory_diff.
    """
    if opponent is None:
        opponent = RandomAgent()

    if num_games == 0:
        return {"win_rate": 0.0, "avg_game_length": 0.0, "avg_territory_diff": 0.0}

    wins = 0
    total_length = 0
    total_territory_diff = 0.0

    for _ in range(num_games):
        env = WallGoGymEnv(max_turns=max_turns, reward_shaping=False)
        obs, info = env.reset()
        turn = 0
        game_reward = 0.0

        while True:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break

            if turn % 2 == 0:
                action = agent.select_action(obs, mask, env=env)
            else:
                action = opponent.select_action(obs, mask, env=env)

            obs, reward, terminated, truncated, info = env.step(action)
            turn += 1

            if terminated or truncated:
                # reward is from perspective of the player who just moved
                # For agent (even turns): positive reward on even turn = agent win
                # But env alternates internally, so we track via info
                if terminated and "scores" in info:
                    scores = info["scores"]
                    players = list(scores.keys())
                    if len(players) >= 2:
                        p1_score = scores[players[0]]
                        p2_score = scores[players[1]]
                        total_territory_diff += p1_score - p2_score
                        if p1_score > p2_score:
                            wins += 1

                total_length += turn
                break

    return {
        "win_rate": wins / num_games,
        "avg_game_length": total_length / num_games,
        "avg_territory_diff": total_territory_diff / num_games,
    }


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WallGo agents")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--agent", choices=["random", "greedy"], default="random")
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    args = parser.parse_args()

    agents = {"random": RandomAgent, "greedy": GreedyAgent}
    agent = agents[args.agent]()
    opponent = agents[args.opponent]()

    metrics = evaluate(agent, num_games=args.num_games, opponent=opponent)
    print(f"Results ({args.num_games} games):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
