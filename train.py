"""Self-play training for WallGo using MaskablePPO.

Usage:
    python train.py --steps 100000 --save-dir checkpoints
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import random
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Group
from rich.text import Text
# Disable strict validation — newer PyTorch's Simplex check rejects valid
# masked distributions due to floating-point normalization in large action spaces.
torch.distributions.Distribution.set_default_validate_args(False)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from wallgo_gym import WallGoGymEnv
from action_encoding import ACTION_SPACE_SIZE
from evaluate import RandomAgent, evaluate

class ModelAgent:
    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic
    def select_action(self, obs, mask, **kwargs):
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)


# ======================================================================
# Self-play environment
# ======================================================================

class SelfPlayEnv(gym.Env):
    """Wraps WallGoGymEnv for self-play: the opponent plays with a random policy.

    During training the learning agent always sees the board from the current
    player's perspective. On the opponent's turns, a built-in opponent policy
    picks an action automatically, so the RL agent only acts on its own turns.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_turns: int = 200, reward_shaping: bool = True, opponent=None):
        super().__init__()
        self._inner = WallGoGymEnv(max_turns=max_turns, reward_shaping=reward_shaping)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        self._opponent = opponent or RandomAgent()
        self._agent_player_idx = 0  # agent is always player 0 (RED)
        self.render_mode = None

    def set_opponent_path(self, model_path: str):
        from sb3_contrib import MaskablePPO
        class RLOpponent:
            def __init__(self, path):
                self.model = MaskablePPO.load(path, device="cpu")
            def select_action(self, obs, mask):
                action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
                return int(action)
        self._opponent = RLOpponent(model_path)

    def reset(self, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed)
        self._agent_player_idx = 0
        return obs, info

    def step(self, action):
        # Agent takes its action
        obs, reward, terminated, truncated, info = self._inner.step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Opponent's turn: pick action and step
        mask = self._inner.action_masks()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return obs, reward, True, truncated, info

        opp_action = self._opponent.select_action(obs, mask)
        obs, opp_reward, terminated, truncated, info = self._inner.step(opp_action)

        # Negate opponent's reward (if opponent wins, agent loses)
        reward = -opp_reward if (terminated or truncated) else 0.0
        
        # Invert the result so the info log reflects the AGENT's outcome
        if terminated and "result" in info:
            if info["result"] == "win":
                info["result"] = "loss"
            elif info["result"] == "loss":
                info["result"] = "win"

        if self._inner.reward_shaping and not terminated and not truncated:
            reward = self._inner._shaping_reward()

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        mask = self._inner.action_masks()
        # sb3-contrib crashes if all masks are False (can't sample from empty
        # distribution). Return a dummy mask when game is over — the env will
        # reset before this action is actually used.
        if not mask.any():
            mask[0] = True
        return mask


def _mask_fn(env: SelfPlayEnv) -> np.ndarray:
    return env.action_masks()


# ======================================================================
# Model factory
# ======================================================================

class WallGoCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

def make_model(env, learning_rate: float = 1e-4, **kwargs) -> MaskablePPO:
    """Create a MaskablePPO model for the given environment."""
    policy_kwargs = dict(
        features_extractor_class=WallGoCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256]
    )
    wrapped = ActionMasker(env, _mask_fn)
    return MaskablePPO(
        "MlpPolicy",
        wrapped,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        **kwargs,
    )


# ======================================================================
# Terminal UI 
# ======================================================================

class CuteReporterCallback(BaseCallback):
    def __init__(self, live, progress, task_id, check_interval=10000, total_steps=100000):
        super().__init__()
        self.live = live
        self.progress = progress
        self.task_id = task_id
        self.check_interval = check_interval
        self.total_steps = total_steps
        
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.recent_wins = 0
        self.recent_games = 0
        self.last_recent_wins = 0
        self.last_recent_games = 0
        self.current_checkpoint = "—"
        self.current_opponent = "Random"
        
    def _on_step(self):
        viz_completed = min(self.num_timesteps, self.total_steps)
        self.progress.update(self.task_id, completed=viz_completed)
        
        for info in self.locals.get("infos", []):
            if "result" in info:
                self.recent_games += 1
                if info["result"] == "win":
                    self.wins += 1
                    self.recent_wins += 1
                elif info["result"] == "loss":
                    self.losses += 1
                elif info["result"] == "tie":
                    self.ties += 1
                    
        if self.n_calls % 100 == 0:
            total = self.wins + self.losses + self.ties
            win_rate_all = (self.wins / max(1, total)) * 100
            
            if self.n_calls % self.check_interval == 0:
                self.last_recent_wins = self.recent_wins
                self.last_recent_games = self.recent_games
                self.recent_wins = 0
                self.recent_games = 0
                
            recent_wr = (self.last_recent_wins / max(1, self.last_recent_games)) * 100 if self.last_recent_games else win_rate_all
            
            cyber_title = "[bold #ff00ff]///[/bold #ff00ff] [bold #00ffcc]NEURAL-LNK SYS // WALLGO-RL[/bold #00ffcc] [bold #ff00ff]///[/bold #ff00ff]"
            stats_text = f"""
[bold #00ffcc]▰▰▰ SYSTEM METRICS ▰▰▰[/bold #00ffcc]
[bold #ff00ff]▶ TOTAL GAMES PLAYED:[/bold #ff00ff] [bold white]{total:,}[/bold white]
[bold #ff00ff]▶ WIN / LOSS / TIE:[/bold #ff00ff] [bold #00FF00]{self.wins:,}[/bold #00FF00] / [bold #FF0055]{self.losses:,}[/bold #FF0055] / [bold #00CCFF]{self.ties:,}[/bold #00CCFF]

[bold #fcee0a]▰▰▰ ALGORITHMIC EFFICIENCY ▰▰▰[/bold #fcee0a]
[bold #fcee0a]▶ GLOBAL WIN RATE:[/bold #fcee0a] [bold white]{win_rate_all:.1f}%[/bold white]
[bold #fcee0a]▶ RECENT WIN RATE (last {self.check_interval:,} steps):[/bold #fcee0a] [bold white]{recent_wr:.1f}%[/bold white]

[bold #87CEEB]▰▰▰ CHECKPOINT STATUS ▰▰▰[/bold #87CEEB]
[bold #87CEEB]▶ LATEST SAVE:[/bold #87CEEB] [bold white]{self.current_checkpoint}[/bold white]
[bold #87CEEB]▶ CURRENT OPPONENT:[/bold #87CEEB] [bold white]{self.current_opponent}[/bold white]
"""
            group = Group(
                Text.from_markup(stats_text, justify="left"),
                self.progress
            )
            self.live.update(Panel(group, title=cyber_title, border_style="#00ffcc", padding=(1, 2)))
            
        return True

# ======================================================================
# Training loop
# ======================================================================

def train(
    total_timesteps: int = 100_000,
    save_dir: str = "checkpoints",
    save_interval: int = 10_000,
    learning_rate: float = 3e-4,
    reward_shaping: bool = True,
    eval_interval: int = 20_000,
    eval_games: int = 200,
    n_envs: int = 1,
    resume: bool = False,
    no_ui: bool = False,
):
    """Run the full self-play training loop."""
    os.makedirs(save_dir, exist_ok=True)
    import re
    
    past_models = []
    def get_step(p):
        m = re.search(r'wallgo_(\d+)', p)
        return int(m.group(1)) if m else -1
        
    for f in os.listdir(save_dir):
        if f.startswith("wallgo_") and f.endswith(".zip") and f != "wallgo_final.zip":
            past_models.append(os.path.join(save_dir, f))
    past_models = sorted(past_models, key=get_step)

    # Setup Environment
    if n_envs > 1:
        def _make():
            return ActionMasker(SelfPlayEnv(reward_shaping=reward_shaping), _mask_fn)
        env_to_use = SubprocVecEnv([_make for _ in range(n_envs)])
        vec_env = env_to_use
    else:
        env_to_use = ActionMasker(SelfPlayEnv(reward_shaping=reward_shaping), _mask_fn)
        single_env = env_to_use.env

    # Setup/Load Model
    steps_done = 0
    last_eval_step = 0
    if resume and past_models:
        latest = past_models[-1]
        steps_done = get_step(latest)
        last_eval_step = steps_done
        
        # Override saved hyperparameters to speed up the "thinking" phase
        _device = "mps" if torch.backends.mps.is_available() else "auto"
        custom_objects = {
            "learning_rate": learning_rate,
            "n_epochs": 4,          # Reduced from 10 to 4 for 2.5x faster gradient updates
        }
        model = MaskablePPO.load(latest, env=env_to_use, device=_device, custom_objects=custom_objects)
    else:
        policy_kwargs = dict(
            features_extractor_class=WallGoCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256]
        )
        model = MaskablePPO(
            "MlpPolicy",
            env_to_use,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,             # Reduced from 10 to 4 for faster updates
            gamma=0.99,
            verbose=0,
            device="mps" if torch.backends.mps.is_available() else "auto",
        )

    # Print startup diagnostics so you can verify device & engine
    print(f"[startup] PyTorch device: {model.device}")
    print(f"[startup] MPS available: {torch.backends.mps.is_available()}")
    try:
        import wallgo_rs
        print("[startup] Engine: Rust (wallgo_rs)")
    except ImportError:
        print("[startup] Engine: Python (wallgo.py)")

    if no_ui:
        # ---- Colab / headless mode: plain print, no rich ----
        import sys, time as _time
        _train_start = _time.time()

        while steps_done < total_timesteps:
            chunk = min(save_interval, total_timesteps - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            steps_done = model.num_timesteps

            elapsed = _time.time() - _train_start
            pct = steps_done / total_timesteps * 100
            print(f"[{elapsed:.0f}s] Steps: {steps_done:,}/{total_timesteps:,} ({pct:.1f}%)", flush=True)

            # Save checkpoint
            path = os.path.join(save_dir, f"wallgo_{steps_done}")
            model.save(path)
            past_models.append(path)
            print(f"  ✓ Checkpoint saved: wallgo_{steps_done}", flush=True)

            # Update opponent
            if random.random() < 0.2:
                chosen = past_models[-1]
            else:
                chosen = random.choice(past_models)
            print(f"  → Opponent: {os.path.basename(chosen)}", flush=True)
            if n_envs > 1:
                vec_env.env_method("set_opponent_path", chosen)
            else:
                single_env.set_opponent_path(chosen)

            # Periodic evaluation
            if steps_done - last_eval_step >= eval_interval or steps_done >= total_timesteps:
                last_eval_step = steps_done
                metrics_rnd = evaluate(ModelAgent(model), num_games=eval_games)
                print(f"  Eval vs Random: W/L/T=[{metrics_rnd['win_rate']:.2f}/{metrics_rnd['loss_rate']:.2f}/{metrics_rnd['tie_rate']:.2f}], "
                      f"len={metrics_rnd['avg_game_length']:.1f}, diff={metrics_rnd['avg_territory_diff']:.1f}", flush=True)

                if len(past_models) >= 2:
                    target_step = steps_done - 500_000
                    baseline_path = min(past_models[:-1], key=lambda p: abs(get_step(p) - target_step))
                    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "auto")
                    baseline_model = MaskablePPO.load(baseline_path, env=env_to_use, device=dev)
                    metrics_base = evaluate(ModelAgent(model), num_games=eval_games, opponent=ModelAgent(baseline_model, deterministic=False))
                    print(f"  Eval vs {os.path.basename(baseline_path)}: W/L/T=[{metrics_base['win_rate']:.2f}/{metrics_base['loss_rate']:.2f}/{metrics_base['tie_rate']:.2f}], "
                          f"len={metrics_base['avg_game_length']:.1f}, diff={metrics_base['avg_territory_diff']:.1f}", flush=True)

    else:
        # ---- Terminal mode: fancy rich UI ----
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        )
        task_id = progress.add_task("[orange3]Training Model...", total=total_timesteps, completed=steps_done)

        with Live(refresh_per_second=4, console=None) as live:
            ui_callback = CuteReporterCallback(live, progress, task_id, check_interval=10_000, total_steps=total_timesteps)

            while steps_done < total_timesteps:
                chunk = min(save_interval, total_timesteps - steps_done)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=ui_callback)
                steps_done = model.num_timesteps

                # Save checkpoint
                path = os.path.join(save_dir, f"wallgo_{steps_done}")
                model.save(path)
                past_models.append(path)
                total_games = ui_callback.wins + ui_callback.losses + ui_callback.ties
                ui_callback.current_checkpoint = f"wallgo_{steps_done} ({total_games:,} games)"
                live.console.print(f"[bold green]✓ Checkpoint saved: wallgo_{steps_done} (total games: {total_games:,})[/bold green]")

                # Update opponent
                if random.random() < 0.2:
                    chosen = past_models[-1]
                else:
                    chosen = random.choice(past_models)
                ui_callback.current_opponent = os.path.basename(chosen).replace(".zip", "")
                live.console.print(f"[bold #FF9900]➜ Self-Play: Opponent → {ui_callback.current_opponent}[/bold #FF9900]")
                if n_envs > 1:
                    vec_env.env_method("set_opponent_path", chosen)
                else:
                    single_env.set_opponent_path(chosen)

                # Periodic evaluation
                if steps_done - last_eval_step >= eval_interval or steps_done >= total_timesteps:
                    last_eval_step = steps_done
                    metrics_rnd = evaluate(ModelAgent(model), num_games=eval_games)
                    live.console.print(f"  [bold cyan]Eval vs Random:[/bold cyan] W/L/T=[{metrics_rnd['win_rate']:.2f}/{metrics_rnd['loss_rate']:.2f}/{metrics_rnd['tie_rate']:.2f}], "
                          f"len={metrics_rnd['avg_game_length']:.1f}, "
                          f"diff={metrics_rnd['avg_territory_diff']:.1f}")

                    if len(past_models) >= 2:
                        target_step = steps_done - 500_000
                        baseline_path = min(past_models[:-1], key=lambda p: abs(get_step(p) - target_step))
                        dev = "mps" if torch.backends.mps.is_available() else "auto"
                        baseline_model = MaskablePPO.load(baseline_path, env=env_to_use, device=dev)
                        metrics_base = evaluate(ModelAgent(model), num_games=eval_games, opponent=ModelAgent(baseline_model, deterministic=False))
                        live.console.print(f"  [bold magenta]Eval vs {os.path.basename(baseline_path)}:[/bold magenta] W/L/T=[{metrics_base['win_rate']:.2f}/{metrics_base['loss_rate']:.2f}/{metrics_base['tie_rate']:.2f}], "
                              f"len={metrics_base['avg_game_length']:.1f}, "
                              f"diff={metrics_base['avg_territory_diff']:.1f}")

    # Save final model
    final_path = os.path.join(save_dir, "wallgo_final")
    model.save(final_path)
    print(f"Training complete. Final model: {final_path}")
    return model


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WallGo RL agent")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Parallel envs (e.g. 4 or 8 for M1)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--no-ui", action="store_true", help="Plain text output (for Colab/notebooks)")
    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        learning_rate=args.lr,
        reward_shaping=not args.no_shaping,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        n_envs=args.n_envs,
        resume=args.resume,
        no_ui=args.no_ui,
    )
