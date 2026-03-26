# 导入系统自带的 sys 模块，用于处理系统级别的参数或退出机制
import sys
# 导入 os 模块，用于处理文件路径和操作系统相关的交互
import os
# 导入 time 模块，用于计算程序运行时间（比如一场游戏花了多少秒）
import time
# 导入 warnings 模块，用于消除或忽略一些烦人的警告信息
import warnings
# 导入 numpy 科学计算库，并简写为 np，它是 Python 里用来处理矩阵和数组的利器
import numpy as np

# 从我们自己写的 evaluate 文件中，导入评测函数 evaluate，和一直在瞎走的 RandomAgent
from evaluate import evaluate, RandomAgent
# 从 Stable Baselines 3 的高级算法库中，导入专门处理带有“不合法动作屏蔽”的近端策略优化算法（MaskablePPO）
from sb3_contrib import MaskablePPO
# 导入 PyTorch 深度学习框架
import torch

# PyTorch 默认会在算概率时做一些严格的底层校验，由于苹果电脑的 GPU（MPS）在某些偏门概率计算上由于兼容性问题还是个半残，
# 时不时会炸个雷报错。所以我们需要强制把这里的“默认严格校验”关掉，让苹果显卡哪怕遇到一丁点数据偏差也不要大惊小怪，防止程序无缘无故中途崩溃罢工。
torch.distributions.Distribution.set_default_validate_args(False)

# 忽略所有的警告信息，这样在使用过程中你黑框框终端界面里看起来更干净清爽，不会红绿黄字乱飞
warnings.filterwarnings("ignore")

# 定义一个我们自己写的类，它就像一个专门用来包装和装载训练好的怪物（AI 模型）的笼子
class RLAgent:
    # 类的初始化构建函数：当我们在别处写 "我要雇佣一个新智能体" 时，程序会自动最先执行这段准备代码
    # path: 你塞进去的那个装满神经大数据的 zip 压缩包路径
    # name: 我们随手贴在笼子上的一个人类可读的标签名（比如叫 "婴儿期"、"壮年期"）
    def __init__(self, path, name):
        # 在屏幕上打印出：“正在从某某路径装载某某AI”，这样做是为了让你知道它没死机卡住
        print(f"Loading {name} from {path}...")
        
        # 智能探测雷达：自动检查你这台电脑上到底有没有代表苹果最强 M 系算力的 MPS 引擎。
        # 如果有，就把 _dev 指定给 "mps" 开启赛博空间，否则只能自求多福选择 "auto" 把 AI 丢去 CPU 靠体力活死算
        _dev = "mps" if torch.backends.mps.is_available() else "auto"
        
        # 把大权交给 MaskablePPO（我们调用的官方强化学习算法主函数库）。
        # 让它用最暴力的手段（load）直接撕开你的 .zip 存档，把百万级别错综复杂的神经网络权重线生生地拉扯进去，并且严格规定必须落在刚才挑好的那块阵地（device=_dev）上。
        self.model = MaskablePPO.load(path, device=_dev)
        
        # 打印一行日志向你报喜，确认你的 AI 灵魂最终到底在这个虚拟肉身（硬件）的哪一块土壤上生了根（到底是 cpu 还是 mps）
        print(f"Loaded {name} onto device: {self.model.device}")
        # 把名字名牌也给它挂在这个笼子的自身属性面板上，为了以后方便指名道姓
        self.name = name

    # 笼子唯一的开口处：每次轮到这个 AI 出手打架时，系统就会强制调用它这个大招函数
    # obs: 游戏裁判推过来的“6x7x7 三重透视扫描阵列”，就是棋盘在它眼里的当前局势
    # mask: 游戏裁判提前盖了公章的“违禁动作名单”，长达 9604 个判断位，告诉它“别往雷区里踩”
    def select_action(self, obs, mask, **kwargs):
        # 核心改动：把 deterministic=True 改成了 False！
        # 这样它在算题时就不会永远 100% 死板地只挑概率最高的那固定唯一的一招。
        # 它会根据概率去“掷骰子”（比如 90% 选绝杀，5% 选另一招），从而让 2000 局打出 2000 种五花八门的神奇剧本！
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
        # 由于预测出来的经常可能是包裹了无数封装的 numpy 层级数学格式的对象，我们粗暴地用 int() 把它碾碎成一个纯正的 Python 下乡整数。打出去。
        return int(action)

# 定义一个类似于“测试主剧本”的宏大长篇函数，把你想要跑的擂台赛一次性写好全包揽
def run_suite():
    # 找到硬盘上最接近 50万、100万、200万、250万以及最终大结局的压缩包路径
    final_path = "checkpoints/wallgo_final.zip"
    cp_500k_path = "checkpoints/wallgo_450000.zip"
    cp_1m_path = "checkpoints/wallgo_880000.zip"
    cp_2m_path = "checkpoints/wallgo_2048000.zip"
    cp_2_5m_path = "checkpoints/wallgo_2506752.zip"
    
    # 把刚才找到的这些不同历史时期的脑子，全部分别关进我们打造好的装载笼子（RLAgent）里
    agent_final = RLAgent(final_path, "Final (3M)")
    agent_500k = RLAgent(cp_500k_path, "Early (500k)")
    agent_1m = RLAgent(cp_1m_path, "Mid (1M)")
    agent_2m = RLAgent(cp_2m_path, "Late (2M)")
    agent_2_5m = RLAgent(cp_2_5m_path, "Elite (2.5M)")
    
    # 牵一头只会瞎走的随机野兽出来，用来垫测底线
    agent_random = RandomAgent()

    # 一把辛酸泪，强制每场必须极其严苛地打满 2000 局，测出绝对硬核的胜率！
    num_games = 2000
    print(f"\n--- Starting Grand Final Evaluation ({num_games} games each) ---\n")

    # =========== 擂台第一战：让绝世帝王去杀猪（热身赛） ===========
    # 记录开打瞬间的秒表
    t0 = time.time()
    # 广播这场比赛的战旗
    print(f"1. Playing: Final (3M) vs Random...")
    # 把帝王和野兽丢进大裁判（evaluate）里互殴 num_games 局，拿到比分牌 res1
    res1 = evaluate(agent_final, num_games=num_games, opponent=agent_random)
    # 把冰冷的比分牌变成带百分号的、人类看得懂的华丽财报输出
    print(f"   => Win Rate (Final vs Random): {res1['win_rate']*100:.1f}%, Tie Rate: {res1['tie_rate']*100:.1f}%, Avg Len: {res1['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s\n")

    # =========== 擂台第二战：帝王降临虐杀50万步的自己 ===========
    t0 = time.time()
    print(f"2. Playing: Final (3M) vs Early (500k)...")
    res2 = evaluate(agent_final, num_games=num_games, opponent=agent_500k)
    print(f"   => Win Rate (Final vs 500k): {res2['win_rate']*100:.1f}%, Tie Rate: {res2['tie_rate']*100:.1f}%, Avg Len: {res2['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s\n")

    # =========== 擂台第三战：帝王降临迎战100万步的自己 ===========
    t0 = time.time()
    print(f"3. Playing: Final (3M) vs Mid (1M)...")
    res3 = evaluate(agent_final, num_games=num_games, opponent=agent_1m)
    print(f"   => Win Rate (Final vs 1M): {res3['win_rate']*100:.1f}%, Tie Rate: {res3['tie_rate']*100:.1f}%, Avg Len: {res3['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s\n")

    # =========== 擂台第四战：帝王降临切磋200万步的准高手 ===========
    t0 = time.time()
    print(f"4. Playing: Final (3M) vs Late (2M)...")
    res4 = evaluate(agent_final, num_games=num_games, opponent=agent_2m)
    print(f"   => Win Rate (Final vs 2M): {res4['win_rate']*100:.1f}%, Tie Rate: {res4['tie_rate']*100:.1f}%, Avg Len: {res4['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s\n")

    # =========== 擂台第五战：巅峰之战！对抗250万步的宿命死敌 ===========
    t0 = time.time()
    print(f"5. Playing: Final (3M) vs Elite (2.5M)...")
    res5 = evaluate(agent_final, num_games=num_games, opponent=agent_2_5m)
    print(f"   => Win Rate (Final vs 2.5M): {res5['win_rate']*100:.1f}%, Tie Rate: {res5['tie_rate']*100:.1f}%, Avg Len: {res5['avg_game_length']:.1f}, Time: {time.time()-t0:.1f}s\n")

    # 最后的谢幕礼仪
    print("--- Evaluation Complete ---")

# Python 独有的底层咒语结构。
# 意思是说：如果你这个文件不是被别人当做附庸模块导入过去的（不是 import），
# 而是“你，这个开发者本人从终端用敲入 `python run_eval.py` 这句强硬命令”直接当做最高主程序敲开来的。
# 那么系统就会认定你是个大爷，立刻乖乖执行里面藏着的 run_suite 主测试函数！否则它只会装死假装啥也没看见！
if __name__ == "__main__":
    run_suite()
