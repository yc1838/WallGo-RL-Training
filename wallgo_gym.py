"""
Gymnasium-compatible wrapper for WallGo. (让它能去被强化学习框架“调教”的相容马甲)

这个文件是原版游戏引擎和那些高大全的 AI 训练框架之间的“中介公司”。
AI 只认识数字，也只习惯某种固定的格式。这里做的事就是：
- 给 AI 定制一个永远不会变的大小的选择题面板（9,604 个定死大小的选择支）。
- 把原本活生生的地图盘，切片成 AI 能看懂的“6重滤镜数字矩阵切片”。
- 给它偶尔发一块叫做（Reward Shaping 领地分差+机动性）的棒棒糖，让它训练时更听话。
- 如果磨洋工超过 200 回合死活打不完，直接一脚把它强制踢下线（截断机制）。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

# 用和刚才一样狡猾的 Try...Except 的双保险去套那个变态级别的 Rust 引擎。一旦加载失败就偷偷用 Python 引擎做备胎。
try:
    from wallgo_rs import WallGoEnv, Player
    _USE_RUST = True
except ImportError:
    from wallgo import WallGoEnv, Player
    _USE_RUST = False

# 把刚才你隔壁 action_encoding 里用来做黑魔法转码和遮罩裁判的几个大哥请过来打工。
from action_encoding import (
    ACTION_SPACE_SIZE, BOARD_SIZE, encode_action, decode_action, get_action_mask,
)


class WallGoGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_turns: int = 200,          # 如果它两百个回合都分不出胜负，强制中断。毕竟时间就是生命。
        reward_shaping: bool = False,  # 是否要给它微小的棒棒糖奖励让它走得更“人模人样”（开启辅助训练）
        render_mode: Optional[str] = None,
    ):
        # 乖乖继承健身房祖师爷（gym.Env）的基建
        super().__init__()
        self.max_turns = max_turns
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        # 在底层悄悄拉起那个真正承载了无数鲜血厮杀的原生游戏大厅
        self._env = WallGoEnv(size=BOARD_SIZE)

        # 【极其重要】给 AI 画出“视网膜”！
        # 告诉 AI：“从今天起，你看到的世界不再是一张图，而是具有 6 层透视效果的、长宽各为 7 的三维浮点数（float32）矩阵盒子。” （比如一层是红方格子，一层是所有左墙...）
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32,
        )
        # 告诉 AI：“你这一生只有这么多个纯整数按键（0 到 9603）可以按，别试图乱按！”
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # Gymnasium 强行规范的 API（外包公司员工准则）
    # ------------------------------------------------------------------

    # 当裁判大喊一声“时间到！重置战场！”的时候，它会立刻无脑跑下面这句：
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        # 把随机数种子给祖师爷留个档防身
        super().reset(seed=seed)
        # 用一盆冷水把底层的鲜血大厅彻底冲洗干净，恢复到棋子刚放好时的 1 比 1 婴儿局
        self._env.reset()
        # 强行把大厅现在的全新棋局拍一张“6层滤镜透视照片”（数字版本），交给 obs 变量
        obs = self._encode_obs()
        # 把照片和一份“今天没啥要交代”的空大纲字典（{}）一起交回给裁判！
        return obs, {}

    # 当 AI 思考了半天，终于咬牙切齿地扔出了一张“我要走某某步数号数字”的决定时：
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 首先保证它传进来的一定是个干净的纯整数，防止它拿一些带小数浮点的数字砸环境
        action = int(action)
        # AI 要是发疯给了一个负数，或者超过了 9603 的乱码，直接让大厅拉响红灯引发闪退！（实际上掩码会拦截它瞎下）
        if action < 0 or action >= ACTION_SPACE_SIZE:
            raise ValueError(f"Action {action} out of range [0, {ACTION_SPACE_SIZE})")

        # 我们去大厅查一下：当前到底是轮到哪一位红蓝大哥在操盘？
        cur = self._env.current_player
        # 从该大哥名下把那枚用来征战天下的唯一一个受死棋子坐标要出来。
        pieces = self._env.get_player_pieces(cur)
        px, py = pieces[0]
        # 用之前 action_encoding 里神级拆包函数，把这个冰冷编号连同当前棋子位置，暴力撕开还原成包含 7 个动作参数的大礼包。
        action_tuple = decode_action(action, px, py)

        # 拿着这份大礼包，毫无保留地砸给底层游戏去模拟下棋！
        # 这个底层如果是 Rust写的，它就能瞬间算完并且立刻结账并告诉你：“（新一轮局势图），奖励结果，有没有人死绝（done），各种分数小抄（info）”
        state, reward, done, info = self._env.step(action_tuple)

        # terminated 翻译过来叫“决出胜负了没”
        terminated = done
        # truncated 翻译过来叫“是不是打太久被裁判没收作案工具强行踢下去了”
        truncated = False

        # 如果你们还活着没砍死对面，但是底层的回合计数器已经悄然爬上了 200 回合的死亡大限。
        if not terminated and self._env.turn_count >= self.max_turns:
            # 立刻亮出红牌：游戏因为超时被硬生生腰斩截断了！
            truncated = True

        # AI 核心贪婪机制：发糖果了！
        if terminated:
            # 如果这盘游戏靠真本事分出胜负了，底层本来就会骄傲地发 +1（赢） / -1（输） / 0（平局）的大额钞票。
            reward = float(reward)
            # 如果设置了“我要发棒棒糖引诱它”，我们还在大头支票上给它多加那么一丁点可怜的辅助小奖金
            if self.reward_shaping:
                reward += self._shaping_reward()
        elif truncated:
            # 如果是被踢下去的，由于你没真赢，一毛钱都别想拿到，奖金归零！
            reward = 0.0
        else:
            # 如果战斗还在僵持，为了防止它在那边发呆不干活，
            if self.reward_shaping:
                # 不断用微不足道的棒棒糖分数告诉它：“朝着领地越多、走步越顺滑的地方去爬！”
                reward = self._shaping_reward()
            else:
                # 不发糖果模式：那你就只能默默忍受着毫无奖励的枯燥黑暗（0.0），直到有人死亡拿到那唯一的 1 块钱
                reward = 0.0

        # 最后给环境重新拍张照
        obs = self._encode_obs()
        # 作为外包保洁工，把新照片，当回合糖果，是否死亡，是否超时超时踢下线，辅助小抄全部扫给高高在上的 PPO 大爷！
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        # 当 PPO 找你要遮罩卡片（动作掩码）时，如果正在用高贵的 Rust 跑着，直接调用底层直接零秒生成的超级遮罩！
        if _USE_RUST:
            return np.asarray(self._env.get_action_mask_np())
        # 否则只能用 Python 老土办法去吭哧吭哧算一张出来
        return get_action_mask(self._env)

    # ------------------------------------------------------------------
    # 以下是不对外部公开的黑箱内幕（Internal）
    # ------------------------------------------------------------------

    def _encode_obs(self) -> np.ndarray:
        # 给世界拍照的核心函数。对于拥有 Rust 黑魔法的版本，这照相机甚至是纯 C 级别一键直出的 numpy 矩阵
        if _USE_RUST:
            return np.asarray(self._env.encode_state_np())
        # Python 就慢慢嵌套循环打印出一盆嵌套表，再转换
        nested = self._env.encode_state()
        return np.array(nested, dtype=np.float32)

    def _shaping_reward(self) -> float:
        """
        非常卑微的小奖励发机器：“给它哪怕 0.01 这种蚊子腿一样的分数，也是为了让它知道啥叫好坏！”
        主要包含两点：领地比别人大（抢地）+ 自己能走的生路比别人多（滑泥鳅）
        """
        # 是谁在领赏？
        cur = self._env.current_player
        total_cells = BOARD_SIZE * BOARD_SIZE

        # ====== 辅助考点一：抢占地盘大战 ======
        # 给底层的并查集老妖婆打电话，求它在一微秒内算出现在全场每一位玩家的势力绝对领土总面基！
        scores = self._env.calculate_scores(self._env.active_players)
        # 用一个最狠的手法，抽调出自己的领土值，没有就算 0
        my_score = scores.get(cur, 0)
        # 抽调出其他所有敌人中，领土最大的那个混蛋的分数
        opp_score = max((s for p, s in scores.items() if p != cur), default=0)
        # 相减得到差值。为了不喧宾夺主盖过真正赢一局（1.0分）的荣耀，我们把这个棒棒糖缩水放大了一百倍，并且还要按棋盘地皮比例平摊，只给 0.0几。
        territory_reward = 0.01 * (my_score - opp_score) / total_cells

        # ====== 辅助考点二：机动泥鳅法则 ======
        pieces = self._env.get_player_pieces(cur)
        if pieces:
            # 去求证我这个小可怜目前在这么残酷的墙壁封锁下还能有几条缝能跨过去？
            moves = self._env.get_valid_moves(pieces[0][0], pieces[0][1])
            # 在一个 7x7 且满地无墙的空地中央你最多能爬 13 个坑！作为满分标准。
            max_moves = 13  
            # 每多一条活路生机，我就多给你赏几厘钱（0.005 的比例）！
            mobility_reward = 0.005 * len(moves) / max_moves
        else:
            # 怎么可能连棋子都没了？难道见鬼了？防报错归零！
            mobility_reward = 0.0

        # 最后把这两个穷酸的棒棒糖加起来，凑出一份两三毛的小费扔给它。
        return territory_reward + mobility_reward
