"""
我们为 AI 和各种基准测试选手量身定做的“竞技场裁判系统”。

包含：
- RandomAgent: 一个只会掷骰子瞎走的绝对弱智（用来做 AI 进步的最底线标尺）。
- GreedyAgent: 一个贪婪短视的低端玩家（不仅瞎走，而且只看眼前哪一步占地盘最大，不计后果）。
- evaluate(): 最核心的铁面无私大裁判函数。负责把两个选手扔进八角笼对打 N 局，然后掏出它们胜率的小本本。
"""

# argparse 库用来让我们可以在黑框框（终端）里轻松打命令传参数（比如 --num-games 10）
import argparse
import os
import sys

# CRITICAL: Prevent TensorFlow from grabbing the GPU and causing CUDA conflicts on Colab.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except (ImportError, RuntimeError):
    pass

import numpy as np
import torch
import torch.nn as nn
# Disable strict validation — newer PyTorch's Simplex check rejects valid
# masked distributions due to floating-point normalization in large action spaces.
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# 敲代码时用的声明标注，没有任何实际执行作用，只是告诉程序员这里可以为空（Optional）或者是一个字典（Dict）
from typing import Optional, Dict

# 和之前一样，祭出能让速度翻 100 倍的 Rust 引擎。一旦你没装好，立马降级用 Python 原生引擎。
try:
    from wallgo_rs import WallGoEnv, Player
except ImportError:
    from wallgo import WallGoEnv, Player

# 导入刚才我们加过爆注的那个强化学习马甲环境
from wallgo_gym import WallGoGymEnv
# 把那些把数字翻译成人话的双向解析器也带上
from action_encoding import (
    ACTION_SPACE_SIZE, decode_action, encode_action, get_action_mask, BOARD_SIZE
)
from gymnasium import spaces

# ======================================================================
# 第一梯队：模型架构与代理（Neural Network & Agents）
# ======================================================================

class WallGoCNN(BaseFeaturesExtractor):
    """跟 train.py 里的架构必须一模一样，否则模型加载不出来。"""
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

class ModelAgent:
    """包装了神经网络模型的代理。"""
    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def select_action(self, obs, mask, **kwargs):
        # 核心：根据模型预测下一步。如果是评估模式，通常 deterministic=True。
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)


# ======================================================================
# 第一梯队：纯天然的弱智沙包选手（Baseline agents）
# ======================================================================

class RandomAgent:
    """一个只会在当前合理的路数里，随机掷骰子瞎走的蒙眼玩家。"""

    # 和我们在别的 AI 里看到的函数名一模一样。用来统一接口，让裁判能够无脑调用。
    def select_action(self, obs: Optional[np.ndarray], mask: np.ndarray, **kwargs) -> int:
        # np.where(mask) 会在一张长达 9604 个判断位的“生死状”里，瞬间挑出现有全场被打上了 True（合法）的考题号。
        legal = np.where(mask)[0]
        # 如果连天都亡它，一个合法的空格都没了，这傻子会当场暴毙报警
        if len(legal) == 0:
            raise ValueError("No legal actions available")
        # 否则它就随便从所有合法的活路里瞎抓一个号码，碾碎成普通数字（int）丢回去
        return int(np.random.choice(legal))


class GreedyAgent:
    """
    稍微高级一点点的势利眼玩家。它每一步都会短视地只看“在这一小步之内，哪一条路能让我瞬间占的领地最大”。
    完全不给自己留后路。但这通常比随机瞎走强那么一点点。
    """

    def select_action(self, obs: Optional[np.ndarray], mask: np.ndarray, *, env: WallGoGymEnv = None, **kwargs) -> int:
        # 同样，拿到眼前一万条路里所有没踩雷的活路列表
        legal = np.where(mask)[0]
        if len(legal) == 0:
            raise ValueError("No legal actions available")
        
        # 如果系统忘了把当前的游戏棋局（env）传给它看，或者它现在只有一条活路退无可退，那它连算都不算，直接无脑走那唯一的一招
        if env is None or len(legal) == 1:
            return int(legal[0])

        # 初始化：先把心目中最好的那一招暂定为列表的第一个，并把目前心目中的最高分设为“负无穷大”（等着被刷新）
        best_action = int(legal[0])
        best_score = -float('inf')
        
        # 撕开包装，强行摸到底层的血肉引擎
        inner = env._env
        # 查查现在自己是红方还是蓝方
        cur = inner.current_player
        # 找到自己的小命（唯一棋子）当前在哪
        pieces = inner.get_player_pieces(cur)
        px, py = pieces[0]

        # ======= 性能救星 =======
        # 因为我们不想让它在这个本来就是沙包的测试上耗费几个小时。
        # 如果它目前合法的线路多于 50 条，这势利眼就只随机抽 50 条去评估，不再穷举，免得卡死！
        candidates = legal if len(legal) <= 50 else np.random.choice(legal, 50, replace=False)

        # 在这 50 条（或者更少）的候补路线里，做穷举沙盘推演：
        for action_int in candidates:
            # 翻译这招到底是要去哪
            action_tuple = decode_action(int(action_int), px, py)
            # 在大脑里克隆一个和当前一模一样的平行宇宙黑客帝国（clone）
            clone = inner.clone()
            # 在平行宇宙里强行走出这步棋，看看会发生什么
            _, reward, done, info = clone.step(action_tuple)
            
            # 如果在这个平行宇宙里这步棋直接导致了游戏终结
            if done:
                # 如果赢了（reward为1），那得把得分猛灌到 100 这种超高分，让他像疯狗一样扑向这步必杀棋
                score = reward * 100  
            else:
                # 否则，它就去平行宇宙里唤醒“并查集老妖婆”（uf 计算地盘的工具）
                uf = clone._build_union_find()
                # 算出这一步走完后天下四方的积分
                scores = clone._scores_with_uf(uf, clone.active_players)
                my_score = scores.get(cur, 0)
                # 找出其他玩家里最高的分当做假想敌
                opp_score = max((s for p, s in scores.items() if p != cur), default=0)
                # 自己的得分减去敌人的得分，就是这步棋带来的短视“净赚分”
                score = my_score - opp_score
            
            # 如果这步棋赚到的分，超越了它心目中曾经的那个“最高分”
            if score > best_score:
                # 更新心目中的极品分数
                best_score = score
                # 死死记住那个能带来这个分数的招式名
                best_action = int(action_int)

        # 把这 50 种推演里短视最高效的一招直接扔上去
        return best_action


# ======================================================================
# 第二梯队：最核心的包青天裁判大厅（Evaluate 函数）
# ======================================================================

def evaluate(
    agent,                # 代表你扔进去受试的第一名选手
    num_games: int = 10,  # 规定必须打满多少局才能出关
    opponent=None,        # 代表它的假想敌
    max_turns: int = 200, # 回合数大限，超过就强行掐掉判定不死不休
) -> Dict[str, float]:    # 会吐出一份带着各种华丽指标平均胜率的成绩单字典
    """
    逼迫两大智能体去八角笼里死磕的宿命赛场。

    工作原理：
    测试选手（agent）永远固定执红方，占据天生的偶数回合（0, 2, 4...）。
    对手（opponent）永远执蓝方，在奇数回合还击。
    直到一方死亡或超时。
    直到一方死亡或超时。
    """
    if opponent is None:
        opponent = RandomAgent()

    if num_games == 0:
        return {"win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0, "avg_game_length": 0.0, "avg_territory_diff": 0.0}

    progressBar = num_games > 10 # 局数多的时候才显示进度

    # 记账本：不仅记总分，还要记谁在什么位置赢的。
    stats = {
        "red_starts": 0, "red_wins": 0, "blue_starts": 0, "blue_wins": 0,
        "total_len": 0, "total_diff": 0.0, "total_wins": 0, "total_losses": 0, "total_ties": 0
    }

    for i in range(num_games):
        if progressBar and (i + 1) % max(1, (num_games // 10)) == 0:
            print(f"  > Progress: {i+1}/{num_games} games...", flush=True)

        env = WallGoGymEnv(max_turns=max_turns, reward_shaping=False)
        obs, info = env.reset()
        turn = 0
        
        # 奇偶轮换：一半局 agent 先手 (RED)，一半局 agent 后手 (BLUE)
        agent_is_red = (i % 2 == 0)
        if agent_is_red: stats["red_starts"] += 1
        else: stats["blue_starts"] += 1

        while True:
            mask = env.action_masks()
            if not mask.any(): break

            # 轮到谁走？
            is_agent_turn = (turn % 2 == 0) if agent_is_red else (turn % 2 != 0)

            if is_agent_turn:
                action = agent.select_action(obs, mask, env=env)
            else:
                action = opponent.select_action(obs, mask, env=env)

            obs, reward, terminated, truncated, info = env.step(action)
            turn += 1

            if terminated or truncated:
                if "scores" in info:
                    scores = info["scores"]
                    red_s, blue_s = scores.get("RED", 0), scores.get("BLUE", 0)
                    
                    if agent_is_red:
                        me, opp = red_s, blue_s
                        if me > opp: stats["red_wins"] += 1
                    else:
                        me, opp = blue_s, red_s
                        if me > opp: stats["blue_wins"] += 1

                    stats["total_diff"] += (me - opp)
                    if me > opp: stats["total_wins"] += 1
                    elif me < opp: stats["total_losses"] += 1
                    else: stats["total_ties"] += 1

                stats["total_len"] += turn
                break
    
    # 汇总
    return {
        "win_rate": stats["total_wins"] / num_games,
        "loss_rate": stats["total_losses"] / num_games,
        "tie_rate": stats["total_ties"] / num_games,
        "red_win_rate": stats["red_wins"] / max(1, stats["red_starts"]),
        "blue_win_rate": stats["blue_wins"] / max(1, stats["blue_starts"]),
        "avg_game_length": stats["total_len"] / num_games,
        "avg_territory_diff": stats["total_diff"] / num_games,
    }

    # 当千锤百炼的 2000 把全部打完。
    # 我们把刚才记的五个小本本里的战果，统统除以总参战场数，得出非常神圣的百分率或平均值字典抛回人间！
    return {
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "tie_rate": ties / num_games,
        "avg_game_length": total_length / num_games,
        "avg_territory_diff": total_territory_diff / num_games,
    }


# ======================================================================
# CLI 命令行快捷通道区
# ======================================================================

# 和刚才那堆代码同理：如果你在外界直接生硬地打命令行 `python evaluate.py --agent greedy`。
# 那这个 `__main__` 条件就会苏醒。
def run_arena(steps_list=[50000, 1500000, 3500000], num_games=20, deterministic=True):
    """【竞技场模式】自动对比多个时间点的自己。"""
    import re
    cp_dir = "checkpoints"
    if not os.path.exists(cp_dir):
        print(f"Error: {cp_dir} 文件夹不存在！")
        return

    # 1. 找齐所有 checkpoint
    all_files = os.listdir(cp_dir)
    past_models = []
    for f in all_files:
        if f.startswith("wallgo_") and f.endswith(".zip"):
            m = re.search(r'wallgo_(\d+)', f)
            if m: past_models.append((int(m.group(1)), os.path.join(cp_dir, f)))
    past_models.sort()

    # 2. 加载最强的（也就是最新的）作为擂主
    final_path = os.path.join(cp_dir, "wallgo_final.zip")
    if not os.path.exists(final_path) and past_models:
        final_path = past_models[-1][1]
    
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Arena] 加载主模型 (擂主): {final_path} ...")
    main_model = MaskablePPO.load(final_path, device=_dev)
    main_agent = ModelAgent(main_model, deterministic=deterministic)

    print(f"{'Opponent':<20} | {'W/L/T Rate':<18} | {'RED WR':<8} | {'BLUE WR':<8} | {'Len/Diff':<12}")
    print("-" * 75)

    # 3. 逐个对战
    targets = sorted(steps_list)
    for target in targets:
        # 寻找最接近 target 步数的模型
        if not past_models: break
        closest = min(past_models, key=lambda x: abs(x[0] - target))
        opp_path = closest[1]
        opp_step = closest[0]

        print(f"正在加载对手 ({opp_step:,} steps)... ", end="", flush=True)
        opp_model = MaskablePPO.load(opp_path, device=_dev)
        opp_agent = ModelAgent(opp_model, deterministic=deterministic)
        
        res = evaluate(main_agent, num_games=num_games, opponent=opp_agent)
        rate_str = f"{res['win_rate']:.2f}/{res['loss_rate']:.2f}/{res['tie_rate']:.2f}"
        ld_str = f"{res['avg_game_length']:.1f}/{res['avg_territory_diff']:.1f}"
        print(f"\r{os.path.basename(opp_path):<20} | {rate_str:<18} | {res['red_win_rate']:.2f}   | {res['blue_win_rate']:.2f}   | {ld_str:<12}")

    # 4. 最后测一下随机沙包
    print(f"正在测试 RandomAgent... ", end="", flush=True)
    res_rnd = evaluate(main_agent, num_games=num_games, opponent=RandomAgent())
    rate_str = f"{res_rnd['win_rate']:.2f}/{res_rnd['loss_rate']:.2f}/{res_rnd['tie_rate']:.2f}"
    ld_str = f"{res_rnd['avg_game_length']:.1f}/{res_rnd['avg_territory_diff']:.1f}"
    print(f"\r{'RandomAgent':<20} | {rate_str:<18} | {res_rnd['red_win_rate']:.2f}   | {res_rnd['blue_win_rate']:.2f}   | {ld_str:<12}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WallGo agents")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--arena", action="store_true", help="Run multi-checkpoint comparison")
    parser.add_argument("--steps", type=str, default="50000,1500000,3500000", help="Steps for arena")
    parser.add_argument("--stochastic", action="store_true", help="Use sampling instead of greedy actions")
    
    parser.add_argument("--agent", choices=["random", "greedy"], default="random")
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    args = parser.parse_args()

    if args.arena:
        steps = [int(s) for s in args.steps.split(",")]
        run_arena(steps_list=steps, num_games=args.num_games, deterministic=not args.stochastic)
    else:
        agents = {"random": RandomAgent, "greedy": GreedyAgent}
        agent = agents[args.agent]()
        opponent = agents[args.opponent]()
        metrics = evaluate(agent, num_games=args.num_games, opponent=opponent)
        print(f"Results ({args.num_games} games, deterministic={not args.stochastic}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")
