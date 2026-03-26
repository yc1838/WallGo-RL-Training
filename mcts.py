"""
蒙特卡洛树搜索（MCTS）—— 让 AI 学会"往前想几步"。

普通的 PPO agent 看到棋盘就直接凭直觉选一步（像闪电一样快但不深思）。
MCTS 则是在每一步之前，在脑子里模拟几十次未来的走法，然后选最稳的那步。

工作原理（用下棋举例）：
假设你面前有 3 种走法。MCTS 会：
1. 选择（Select）：顺着搜索树往下走，每次挑"最值得探索的"分支
2. 扩展（Expand）：走到没探索过的叶子节点，展开它
3. 评估（Evaluate）：用神经网络估算"这个局面我大概能赢多少%"
4. 回传（Backprop）：把估算结果沿路传回去，更新每个节点的统计

重复 50 次后，选被访问次数最多的那步棋（最多次 = 最可靠的好棋）。

这个文件不改变训练流程——训练还是用 PPO。
MCTS 只在"下棋对战"时使用，让同一个模型变得更强。
"""

import math
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from action_encoding import (
    ACTION_SPACE_SIZE, BOARD_SIZE, decode_action, encode_action, get_action_mask,
)

# 同样的双保险：优先用 Rust 引擎，没有就用 Python
try:
    from wallgo_rs import WallGoEnv, Player
except ImportError:
    from wallgo import WallGoEnv, Player


# ======================================================================
# 神经网络接口：从 PPO 模型里提取"直觉"和"胜率预估"
# ======================================================================

def get_policy_and_value(model, obs: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    从训练好的 MaskablePPO 模型中，一次性提取两样东西：
    1. policy（策略）：一个长度为 9604 的概率数组，表示"每个动作有多好"
    2. value（价值）：一个 -1 到 +1 的数字，表示"当前局面我大概能赢多少"

    这两样东西就是 MCTS 的"大脑"：
    - policy 告诉搜索树"优先探索哪些分支"（先验概率）
    - value 告诉搜索树"这个局面值多少分"（不用真的下完一整盘棋）

    参数：
        model: 训练好的 MaskablePPO 模型
        obs: 形状为 (6, 7, 7) 的棋盘观察值

    返回：
        probs: 形状为 (9604,) 的概率数组
        value: 一个浮点数（-1 到 +1）
    """
    # 把 numpy 数组变成 PyTorch 张量，加一个 batch 维度（模型期望输入是批量的）
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)

    # torch.no_grad() 告诉 PyTorch "我只是在预测，不需要算梯度"，能省很多内存和时间
    with torch.no_grad():
        # 第一步：通过 CNN 提取特征（就是那个 6x7x7 → 256 维的过程）
        # SB3 的 extract_features 会根据 policy 或 value 网络选择对应的特征提取器
        # 传入 model.policy.features_extractor 确保用 policy 网络的特征
        features = model.policy.extract_features(obs_tensor, model.policy.features_extractor)

        # 第二步：通过 MLP 提取器（两层 256 的全连接网络），分叉出策略和价值两条路
        latent_pi, latent_vf = model.policy.mlp_extractor(features)

        # 第三步（策略路）：输出 9604 个 logits（原始分数，还没变成概率）
        logits = model.policy.action_net(latent_pi)
        # softmax 把原始分数变成概率（全部加起来 = 1.0）
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # 第三步（价值路）：输出一个数字，表示当前局面的"价值"
        value = model.policy.value_net(latent_vf).cpu().numpy()[0, 0]

    return probs, float(value)


# ======================================================================
# MCTS 搜索树的节点
# ======================================================================

class MCTSNode:
    """
    搜索树上的一个节点，代表棋局的一个状态。

    每个节点记录：
    - 它被访问了多少次（visit_count）
    - 累计得到了多少分（value_sum）
    - 神经网络认为走到这一步的先验概率有多大（prior）
    - 它的所有子节点（children），每个子节点对应一个可能的动作
    """

    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'value_sum',
                 'children', 'env_state', 'is_terminal']

    def __init__(self, parent: Optional['MCTSNode'] = None,
                 action: int = -1, prior: float = 0.0):
        self.parent = parent          # 父节点（根节点的父亲是 None）
        self.action = action          # 从父节点走到这里用的动作编号
        self.prior = prior            # 神经网络给的先验概率 P
        self.visit_count = 0          # 被访问次数 N
        self.value_sum = 0.0          # 累计价值 W
        self.children: Dict[int, 'MCTSNode'] = {}  # 子节点字典：{动作编号: 子节点}
        self.env_state: Optional[WallGoEnv] = None  # 这个节点对应的游戏状态
        self.is_terminal = False      # 这个节点是否是游戏结束的状态

    def q_value(self) -> float:
        """
        平均价值 Q = W / N。
        如果还没被访问过（N=0），返回 0。
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.41) -> float:
        """
        UCB（Upper Confidence Bound）分数 = Q + 探索奖励。

        公式：UCB = Q(s,a) + c_puct × P(s,a) × √(N_parent) / (1 + N(s,a))

        第一项 Q：这个节点的平均价值（"已知有多好"）
        第二项：探索奖励（"还没怎么试过，应该去看看"）
            - P 越大（网络觉得它好），探索奖励越高
            - N 越小（去的次数少），探索奖励越高
            - 父节点 N 越大（父节点很热门），也会推高探索奖励

        c_puct 控制探索程度：越大越爱冒险尝试新路线，越小越保守只走已知好的。
        """
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration

    def is_leaf(self) -> bool:
        """是否是叶子节点（还没展开过子节点）"""
        return len(self.children) == 0

    def best_child(self, c_puct: float = 1.41) -> 'MCTSNode':
        """在所有子节点中，选 UCB 分数最高的那个"""
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))


# ======================================================================
# MCTS 搜索引擎
# ======================================================================

def _get_obs_from_env(env: WallGoEnv) -> np.ndarray:
    """从原始游戏引擎获取 6x7x7 的观察值（和 wallgo_gym.py 里一样的格式）"""
    try:
        return np.asarray(env.encode_state_np())
    except AttributeError:
        return np.array(env.encode_state(), dtype=np.float32)


def _get_mask_from_env(env: WallGoEnv) -> np.ndarray:
    """从原始游戏引擎获取 9604 大小的动作掩码"""
    try:
        return np.asarray(env.get_action_mask_np())
    except AttributeError:
        return get_action_mask(env)


def run_mcts(
    root_env: WallGoEnv,
    model,
    num_simulations: int = 50,
    c_puct: float = 1.41,
) -> MCTSNode:
    """
    从给定的游戏状态开始，执行 MCTS 搜索。

    参数：
        root_env: 当前的游戏状态（会被 clone，不会修改原始状态）
        model: 训练好的 MaskablePPO 模型
        num_simulations: 模拟次数（越多越强但越慢）
        c_puct: 探索系数

    返回：
        root: 搜索完成后的根节点（包含所有搜索统计信息）
    """
    # 创建根节点
    root = MCTSNode()
    root.env_state = root_env.clone()

    # 先对根节点做一次神经网络评估，获取先验概率
    root_obs = _get_obs_from_env(root.env_state)
    root_mask = _get_mask_from_env(root.env_state)
    root_probs, _ = get_policy_and_value(model, root_obs)

    # 只保留合法动作的概率，重新归一化
    legal_actions = np.where(root_mask)[0]
    if len(legal_actions) == 0:
        return root

    # 把非法动作的概率设为 0
    masked_probs = root_probs * root_mask
    prob_sum = masked_probs.sum()
    if prob_sum > 0:
        masked_probs /= prob_sum
    else:
        # 如果网络给所有合法动作的概率都是 0（不太可能但防御一下）
        masked_probs[legal_actions] = 1.0 / len(legal_actions)

    # 为根节点的每个合法动作创建子节点
    for action in legal_actions:
        child = MCTSNode(parent=root, action=int(action), prior=float(masked_probs[action]))
        root.children[int(action)] = child

    # 开始模拟！
    for _ in range(num_simulations):
        node = root

        # ==================== 第一步：选择（Select） ====================
        # 从根节点往下走，每次选 UCB 最高的子节点，直到碰到叶子节点
        while not node.is_leaf() and not node.is_terminal:
            node = node.best_child(c_puct)

        # 如果到达终局节点，直接用终局结果回传
        if node.is_terminal:
            _backpropagate(node, -node.q_value() if node.visit_count > 0 else 0.0)
            continue

        # ==================== 第二步：扩展（Expand） ====================
        # 克隆父节点的游戏状态，执行这个节点的动作
        if node.env_state is None:
            parent_env = node.parent.env_state
            node.env_state = parent_env.clone()

            # 找到当前玩家的棋子位置
            cur_player = node.env_state.current_player
            pieces = node.env_state.get_player_pieces(cur_player)
            px, py = pieces[0]

            # 解码动作并执行
            action_tuple = decode_action(node.action, px, py)
            _, reward, done, info = node.env_state.step(action_tuple)

            if done:
                # 游戏结束了！记录终局价值
                node.is_terminal = True
                # reward 是从执行动作的玩家视角看的（+1 赢 / -1 输）
                # 我们直接把这个 reward 传给 _backpropagate，
                # 这样对应的 action 节点就会得到正确的分数（赢了就得 +1，输了就得 -1）
                _backpropagate(node, float(reward))
                continue

        # 用神经网络评估这个新的叶子节点
        obs = _get_obs_from_env(node.env_state)
        mask = _get_mask_from_env(node.env_state)
        probs, value = get_policy_and_value(model, obs)

        # 为这个节点的每个合法动作创建子节点
        legal = np.where(mask)[0]
        if len(legal) > 0:
            masked_p = probs * mask
            p_sum = masked_p.sum()
            if p_sum > 0:
                masked_p /= p_sum
            else:
                masked_p[legal] = 1.0 / len(legal)

            for action in legal:
                child = MCTSNode(parent=node, action=int(action), prior=float(masked_p[action]))
                node.children[int(action)] = child

        # ==================== 第三步：回传（Backpropagate） ====================
        # value 是从当前玩家视角的评估（"我觉得我能赢多少"）
        # 但树的上一层是对手的视角，所以要取反
        _backpropagate(node, -value)

    return root


def _backpropagate(node: MCTSNode, value: float):
    """
    把评估结果从叶子节点一路传回根节点。

    关键点：每上一层就取反一次 value。
    因为在两人对弈中，对我好的局面对对手是坏的，反之亦然。
    比如叶子节点估值 +0.8（对叶子节点的当前玩家有利），
    传到上一层（对手）就是 -0.8，再上一层（我）又是 +0.8。
    """
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value  # 每上一层取反！
        node = node.parent


# ======================================================================
# MCTS Agent：实现标准的 select_action 接口
# ======================================================================

class MCTSAgent:
    """
    使用 MCTS 搜索的 AI 选手。

    和普通的 ModelAgent 一样实现 select_action 接口，
    可以直接丢进 evaluate() 函数里和别人对战。

    区别是：ModelAgent 看到棋盘直接选动作（闪电快），
    MCTSAgent 会先在脑子里模拟几十次再选（慢但强）。

    用法：
        model = MaskablePPO.load("checkpoints/wallgo_final")
        agent = MCTSAgent(model, num_simulations=50)
        result = evaluate(agent, num_games=100)
    """

    def __init__(self, model, num_simulations: int = 50,
                 c_puct: float = 1.41, temperature: float = 0.0):
        """
        参数：
            model: 训练好的 MaskablePPO 模型
            num_simulations: 每步模拟次数（50 = 标准，200 = 最强）
            c_puct: 探索系数（1.41 是经典值，越大越爱冒险）
            temperature: 选择温度
                - 0.0 = 贪心（永远选被访问最多的，最稳定）
                - 1.0 = 按访问次数的概率随机选（增加多样性）
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.name = f"MCTS-{num_simulations}"

    def select_action(self, obs: Optional[np.ndarray], mask: np.ndarray,
                      *, env=None, **kwargs) -> int:
        """
        MCTS 的核心入口。evaluate() 函数会调用这个方法。

        参数：
            obs: 6x7x7 的棋盘观察值（MCTS 不直接用这个，而是从 env 克隆状态）
            mask: 9604 的合法动作掩码
            env: WallGoGymEnv 实例（MCTS 需要这个来克隆游戏状态）

        返回：
            选中的动作编号（0-9603）
        """
        # 如果没有传入 env，只能退化为直接用网络选（和普通 ModelAgent 一样）
        if env is None:
            probs, _ = get_policy_and_value(self.model, obs)
            masked = probs * mask
            if masked.sum() > 0:
                masked /= masked.sum()
            return int(np.argmax(masked))

        # 获取底层游戏引擎（WallGoEnv），evaluate.py 传的是 WallGoGymEnv
        raw_env = env._env if hasattr(env, '_env') else env

        # 执行 MCTS 搜索！
        root = run_mcts(raw_env, self.model, self.num_simulations, self.c_puct)

        if not root.children:
            # 搜索树是空的（不应该发生），用掩码随机选
            legal = np.where(mask)[0]
            return int(np.random.choice(legal))

        # 从搜索结果中选择动作
        if self.temperature == 0.0:
            # 贪心模式：选被访问次数最多的动作
            best = max(root.children.values(), key=lambda c: c.visit_count)
            return best.action
        else:
            # 温度模式：按访问次数的概率分布随机选
            actions = []
            visits = []
            for action, child in root.children.items():
                actions.append(action)
                visits.append(child.visit_count)

            visits = np.array(visits, dtype=np.float64)
            if self.temperature != 1.0:
                visits = visits ** (1.0 / self.temperature)

            visit_sum = visits.sum()
            if visit_sum > 0:
                probs = visits / visit_sum
            else:
                probs = np.ones(len(visits)) / len(visits)

            chosen = np.random.choice(actions, p=probs)
            return int(chosen)
