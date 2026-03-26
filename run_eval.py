import sys
import os
import time
import warnings
import numpy as np

from evaluate import evaluate, RandomAgent
from sb3_contrib import MaskablePPO
import torch
torch.distributions.Distribution.set_default_validate_args(False)

warnings.filterwarnings("ignore")

class RLAgent:
    def __init__(self, path, name):
        print(f"Loading {name} from {path}...")
        _dev = "mps" if torch.backends.mps.is_available() else "auto"
        self.model = MaskablePPO.load(path, device=_dev)
        print(f"Loaded {name} onto device: {self.model.device}")
        self.name = name

    def select_action(self, obs, mask, **kwargs):
        # We use deterministic=True for the purest performance test
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

def run_suite():
    earliest_path = "checkpoints/wallgo_10000.zip"
    latest_path = "checkpoints/wallgo_590000.zip"
    
    agent_early = RLAgent(earliest_path, "Early (10k)")
    agent_late = RLAgent(latest_path, "Late (590k)")
    agent_random = RandomAgent()

    num_games = 2000
    print(f"\n--- Starting Evaluation ({num_games} games each) ---\n")

    # # 1. Early vs Random
    # t0 = time.time()
    # print(f"1. Playing: Early (10k) vs Random...")
    # res1 = evaluate(agent_early, num_games=num_games, opponent=agent_random)
    # print(f"   => Win Rate (Early): {res1['win_rate']*100:.1f}%, Avg Len: {res1['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s")

    # 2. Late vs Random
    t0 = time.time()
    print(f"2. Playing: Late (340k) vs Random...")
    res2 = evaluate(agent_late, num_games=num_games, opponent=agent_random)
    print(f"   => Win Rate (Late): {res2['win_rate']*100:.1f}%, Avg Len: {res2['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s")

    # 3. Late vs Early
    t0 = time.time()
    print(f"3. Playing: Late (340k) vs Early (10k)...")
    res3 = evaluate(agent_late, num_games=num_games, opponent=agent_early)
    print(f"   => Win Rate (Late vs Early): {res3['win_rate']*100:.1f}%, Avg Len: {res3['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s")

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    run_suite()
