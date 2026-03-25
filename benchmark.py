"""Benchmark suite for WallGo RL training performance.

Measures:
1. Raw env simulation speed (games/sec, steps/sec)
2. Action mask generation speed
3. State encoding speed
4. Full training step throughput
"""

import time
import random
import numpy as np
from wallgo import WallGoEnv, Player
from wallgo_gym import WallGoGymEnv
from action_encoding import get_action_mask, encode_action


def benchmark_raw_games(num_games=200, seed=42):
    """Benchmark: how many raw WallGoEnv games per second."""
    rng = random.Random(seed)
    env = WallGoEnv()
    total_steps = 0
    t0 = time.perf_counter()
    for _ in range(num_games):
        env.reset()
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            env.step(rng.choice(legal))
            total_steps += 1
    elapsed = time.perf_counter() - t0
    return {
        "num_games": num_games,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "games_per_sec": num_games / elapsed,
        "steps_per_sec": total_steps / elapsed,
        "avg_steps_per_game": total_steps / num_games,
    }


def benchmark_action_mask(num_calls=2000, seed=42):
    """Benchmark: get_action_mask() calls per second."""
    rng = random.Random(seed)
    env = WallGoEnv()
    env.reset()
    # Play a few steps to get to a mid-game state
    for _ in range(10):
        legal = env.get_legal_actions()
        if not legal or env.done:
            break
        env.step(rng.choice(legal))

    t0 = time.perf_counter()
    for _ in range(num_calls):
        get_action_mask(env)
    elapsed = time.perf_counter() - t0
    return {
        "num_calls": num_calls,
        "elapsed_sec": elapsed,
        "calls_per_sec": num_calls / elapsed,
    }


def benchmark_encode_state(num_calls=5000, seed=42):
    """Benchmark: encode_state() calls per second."""
    rng = random.Random(seed)
    env = WallGoEnv()
    env.reset()
    for _ in range(10):
        legal = env.get_legal_actions()
        if not legal or env.done:
            break
        env.step(rng.choice(legal))

    t0 = time.perf_counter()
    for _ in range(num_calls):
        env.encode_state()
    elapsed = time.perf_counter() - t0
    return {
        "num_calls": num_calls,
        "elapsed_sec": elapsed,
        "calls_per_sec": num_calls / elapsed,
    }


def benchmark_get_legal_actions(num_calls=2000, seed=42):
    """Benchmark: get_legal_actions() calls per second."""
    rng = random.Random(seed)
    env = WallGoEnv()
    env.reset()
    for _ in range(10):
        legal = env.get_legal_actions()
        if not legal or env.done:
            break
        env.step(rng.choice(legal))

    t0 = time.perf_counter()
    for _ in range(num_calls):
        env.get_legal_actions()
    elapsed = time.perf_counter() - t0
    return {
        "num_calls": num_calls,
        "elapsed_sec": elapsed,
        "calls_per_sec": num_calls / elapsed,
    }


def benchmark_gym_games(num_games=200, seed=42):
    """Benchmark: full Gymnasium wrapper games per second."""
    rng = np.random.RandomState(seed)
    env = WallGoGymEnv(reward_shaping=True)
    total_steps = 0
    t0 = time.perf_counter()
    for _ in range(num_games):
        env.reset()
        while True:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - t0
    return {
        "num_games": num_games,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "games_per_sec": num_games / elapsed,
        "steps_per_sec": total_steps / elapsed,
    }


def benchmark_clone(num_clones=1000, seed=42):
    """Benchmark: env.clone() calls per second."""
    rng = random.Random(seed)
    env = WallGoEnv()
    env.reset()
    for _ in range(10):
        legal = env.get_legal_actions()
        if not legal or env.done:
            break
        env.step(rng.choice(legal))

    t0 = time.perf_counter()
    for _ in range(num_clones):
        env.clone()
    elapsed = time.perf_counter() - t0
    return {
        "num_clones": num_clones,
        "elapsed_sec": elapsed,
        "clones_per_sec": num_clones / elapsed,
    }


def run_all():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("WallGo RL Training — Performance Benchmark")
    print("=" * 60)

    print("\n1. Raw game simulation (WallGoEnv)")
    r = benchmark_raw_games()
    print(f"   {r['num_games']} games, {r['total_steps']} steps")
    print(f"   {r['games_per_sec']:.1f} games/sec, {r['steps_per_sec']:.0f} steps/sec")
    print(f"   Avg {r['avg_steps_per_game']:.1f} steps/game")

    print("\n2. Gym wrapper simulation (WallGoGymEnv + shaping)")
    r = benchmark_gym_games()
    print(f"   {r['num_games']} games, {r['total_steps']} steps")
    print(f"   {r['games_per_sec']:.1f} games/sec, {r['steps_per_sec']:.0f} steps/sec")

    print("\n3. get_legal_actions()")
    r = benchmark_get_legal_actions()
    print(f"   {r['calls_per_sec']:.0f} calls/sec")

    print("\n4. get_action_mask()")
    r = benchmark_action_mask()
    print(f"   {r['calls_per_sec']:.0f} calls/sec")

    print("\n5. encode_state()")
    r = benchmark_encode_state()
    print(f"   {r['calls_per_sec']:.0f} calls/sec")

    print("\n6. clone()")
    r = benchmark_clone()
    print(f"   {r['clones_per_sec']:.0f} clones/sec")

    print("\n" + "=" * 60)


def benchmark_parallel_games(num_games_per_env=50, n_envs=4, seed=42):
    """Benchmark: parallel game simulation via SubprocVecEnv."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from train import SelfPlayEnv

    def make_env():
        return SelfPlayEnv(max_turns=200)

    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    total_steps = 0
    games_done = 0
    target = num_games_per_env * n_envs

    t0 = time.perf_counter()
    obs = vec_env.reset()
    while games_done < target:
        masks = np.array(vec_env.env_method("action_masks"))
        actions = []
        for i in range(n_envs):
            legal = np.where(masks[i])[0]
            actions.append(int(np.random.choice(legal)) if len(legal) > 0 else 0)
        obs, rewards, dones, infos = vec_env.step(actions)
        total_steps += n_envs
        games_done += sum(dones)
    elapsed = time.perf_counter() - t0
    vec_env.close()

    return {
        "n_envs": n_envs,
        "games_done": games_done,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "games_per_sec": games_done / elapsed,
        "steps_per_sec": total_steps / elapsed,
    }


if __name__ == "__main__":
    run_all()

    print("\n7. Parallel simulation (SubprocVecEnv, 4 envs)")
    r = benchmark_parallel_games(num_games_per_env=50, n_envs=4)
    print(f"   {r['n_envs']} envs, {r['games_done']} games, {r['total_steps']} steps")
    print(f"   {r['games_per_sec']:.1f} games/sec, {r['steps_per_sec']:.0f} steps/sec")

    print("\n8. Parallel simulation (SubprocVecEnv, 8 envs)")
    r = benchmark_parallel_games(num_games_per_env=25, n_envs=8)
    print(f"   {r['n_envs']} envs, {r['games_done']} games, {r['total_steps']} steps")
    print(f"   {r['games_per_sec']:.1f} games/sec, {r['steps_per_sec']:.0f} steps/sec")

    print("\n" + "=" * 60)
