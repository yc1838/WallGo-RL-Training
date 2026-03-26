"""
我们为 AI 和各种基准测试选手量身定做的“竞技场裁判系统”。

包含：
- RandomAgent: 一个只会掷骰子瞎走的绝对弱智（用来做 AI 进步的最底线标尺）。
- GreedyAgent: 一个贪婪短视的低端玩家（不仅瞎走，而且只看眼前哪一步占地盘最大，不计后果）。
- evaluate(): 最核心的铁面无私大裁判函数。负责把两个选手扔进八角笼对打 N 局，然后掏出它们胜率的小本本。
"""

# argparse 库用来让我们可以在黑框框（终端）里轻松打命令传参数（比如 --num-games 10）
import argparse
# 镇馆之宝 numpy（简写np），用来做大规模矩阵数学运算
import numpy as np
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
    ACTION_SPACE_SIZE, decode_action, encode_action, get_action_mask,
)


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
    """
    # 如果没给他挑对手，系统就贴心地给它塞个什么都不懂的随机瞎走傻子让它练手
    if opponent is None:
        opponent = RandomAgent()

    # 如果说打的局数是 0 局。。。耍我呢？直接原地返回四个鸭蛋，退朝！
    if num_games == 0:
        return {"win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0, "avg_game_length": 0.0, "avg_territory_diff": 0.0}

    # 准备五个厚厚的小本本，开始画正字记战绩：
    wins = 0                 # 第一选手赢了几把
    losses = 0               # 输了几把
    ties = 0                 # 平了几把
    total_length = 0         # 所有对局回合数加起来（用于最后算平均每局多长）
    total_territory_diff = 0.0 # 所有局里，打完后净胜地主多大（看看是碾压还是险胜）

    # 开始进入暗无天日的“无数局”死循环
    for _ in range(num_games):
        # 召唤出一个全新的角斗场环境壳子，我们刻意把“给糖果”的辅助训练关掉（False），纯粹看真枪实弹的你死我活
        env = WallGoGymEnv(max_turns=max_turns, reward_shaping=False)
        # 上一盆冷水，把棋盘重置
        obs, info = env.reset()
        # 挂上倒计时牌（从 0 回合起算）
        turn = 0
        game_reward = 0.0

        # 进入这一盘游戏的至死方休内循环
        while True:
            # 裁判拿出红红绿绿的通行掩码打分卡
            mask = env.action_masks()
            # 瞄一眼里面还有没有活路可走
            legal = np.where(mask)[0]
            # 如果实在无路可走（比如被堵死了），强行冲破天际结束这场比赛
            if len(legal) == 0:
                break

            # 回合数是偶数时，轮到受试一号兵出列挑选动作
            if turn % 2 == 0:
                action = agent.select_action(obs, mask, env=env)
            # 奇数回合，由敌人二号兵选动作
            else:
                action = opponent.select_action(obs, mask, env=env)

            # 把最终被双房决定的那个宿命 action 号码球，塞回大厅去推进局势进度！
            # 瞬间吐出了下一局的天翻地覆的格局 obs，生死存亡 terminated，和裁决结果 info
            obs, reward, terminated, truncated, info = env.step(action)
            # 熬过一个回合，计分板上加一。
            turn += 1

            # 查房：要么因为有人真刀真枪拿下了对局（terminated），要么硬生生抗到了 200 回合被腰斩掐线（truncated）
            if terminated or truncated:
                # 判断是真枪真刀分出的胜负，且系统有乖巧地在底层给出各个红蓝的分数
                if terminated and "scores" in info:
                    scores = info["scores"]
                    # 测试选手永远是掌控大局的傲慢红方
                    agent_score = scores.get("RED", 0)
                    # 假想敌永远是蓝方
                    opp_score = scores.get("BLUE", 0)
                    # 计入总分差的大本本（用来最后评估那个人打赢这一局到底是靠虐菜还是只赢了一里地）
                    total_territory_diff += agent_score - opp_score
                    
                    # 极其朴素正面的判决书：
                    if agent_score > opp_score:
                        wins += 1    # 大赚特赚
                    elif agent_score < opp_score:
                        losses += 1  # 惨遭反杀
                    else:
                        ties += 1    # 苟出天际和平收尾

                # 无论如何，把这一把长达几十回合的寿命刻在计步本上
                total_length += turn
                # 一把打完，直接从当前的比赛里 Break 撕破结界出去，去开启下一个 for 重生对决循环
                break

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
if __name__ == "__main__":
    # 解析你这人类在黑屏幕里打的参数，什么 `--num-games` 呀之类的
    parser = argparse.ArgumentParser(description="Evaluate WallGo agents")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--agent", choices=["random", "greedy"], default="random")
    parser.add_argument("--opponent", choices=["random", "greedy"], default="random")
    args = parser.parse_args()

    # 弄一个快捷翻译簿：如果你输入 'random'，我就把 RandomAgent 类的本体当成大礼送上
    agents = {"random": RandomAgent, "greedy": GreedyAgent}
    # 实例化这俩货色
    agent = agents[args.agent]()
    opponent = agents[args.opponent]()

    # 调用上面那个大裁判，丢进去开打！
    metrics = evaluate(agent, num_games=args.num_games, opponent=opponent)
    
    # 比赛落幕，在你的黑框屏幕上打印这些金光闪闪的大字结果
    print(f"Results ({args.num_games} games):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
