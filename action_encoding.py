"""
这其实是一个给 AI 从“人类现实世界”进入“虚拟数据矩阵”的【超级翻译器】模块。

强化学习的底层世界非常原始无脑：AI 大脑就是一个只会疯狂计算数学概率的公式团。
它根本听不懂你给它下的命令诸如：“请把它移到第三行四列，然后再在隔壁左边起一面墙去阻挡他”。
为了让它能在一秒钟成千上万次地下期，我们必须做一个庞大的词典：
把现实中在 7x7 棋盘上一共存在着的动作组合暴力穷举出来，每一种组合发放一个类似于门牌号的唯一编号。
AI 每次只管吼一声编号（比如：“给我来一出 8848 杀招！”），我们就靠这翻译官瞬间破译还原！

穷举算法很简单：
- 目的地格子：共有 7×7 = 49 个可能
- 要放新墙的中心格子：共有 7×7 = 49 个可能
- 墙壁插在那个网格的哪个边：共有 上/右/下/左 4 种可能
最终理论组合峰值：49 乘 49 乘 4 = 9604。
也就是说，“从 0 编到 9603”，足够涵盖它能想象出来的所有变化！
"""

# 万年不变的好大哥 numpy，提供超强的大规模数字数组操作
import numpy as np
# Typing 库里的 Tuple 用于代码审查提醒：喂，这里要返回好几个变量绑在一起的包裹
from typing import Tuple

# 这是个超级容错后路。
# 虽然我们主打让环境用天下武功唯快不破的 Rust（一个巨硬核极其变态快的神级语言）去替 AI 扫雷。
# 但如果你恰好换了一台没法跑出 Rust 引擎的新电脑，它还能迅速抓起身边的纯 Python 菜刀接着跑，宁可慢点也不让你崩溃死机。
try:
    from wallgo_rs import WallGoEnv, SIDES
except ImportError:
    from wallgo import WallGoEnv, SIDES

# 把我们老祖宗传下来的游戏规模定死了。棋盘的边长是 7
BOARD_SIZE = 7
# 网格绝对总数是 49 块地皮
NUM_CELLS = BOARD_SIZE * BOARD_SIZE  # 49
# 墙的可用插槽数量写死了，上下左右四个基底维
NUM_SIDES = 4

# 【核心暗码字典 1】用来极速把人类可读的可爱的英文字符串方向，变成极其冷血的数字标号。
SIDE_TO_INDEX = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}
# 【核心暗码字典 2 这段属于非常极其风骚的一行 Python 简写（字典推导式）！
# 它极其优雅地立刻顺手造出了个专门反着查询的反转互逆字典。你之后扔个 2 进去拿到的就是 'bottom'。
INDEX_TO_SIDE = {v: k for k, v in SIDE_TO_INDEX.items()}

# 9604！这是挂在这个文件最重要的天威图腾。AI 一辈子活动范畴的禁锢结界绝对大小！
ACTION_SPACE_SIZE = NUM_CELLS * NUM_CELLS * NUM_SIDES  # 9604


def encode_action(to_x: int, to_y: int, wall_x: int, wall_y: int, wall_side: str) -> int:
    """
    负责把一组复杂的动作要求（我要走哪去，墙要放哪横着竖着），压平、折叠、强压编码成 0 到 9603 之间的一个纯数字。
    它是现实进入矩阵世界的唯一入口。
    """
    # 稍微做个防御性检查，免得外面写外部逻辑的人醉驾传进来个像 'middle' 或者 'center' 之类的假方向。直接一脚把他踹飞报错！
    if wall_side not in SIDE_TO_INDEX:
        raise ValueError(f"Invalid wall side: {wall_side!r}")
    
    # 核心映射法：如何把一个二维在屏幕上的 x, y 坐标，一脚踹成一个线性的序列号？
    # 因为每一行有 7 个格子，所以我就简单粗暴地：直接 用你的行号 y 乘以 7，然后加上你的列号 x 即可！
    # （比如：第 2 行第 3 列（y=2, x=3），那么 2 * 7 + 3 = 17 号地皮！）
    to_cell = to_y * BOARD_SIZE + to_x
    # 同样粗暴的方法，搞出那堵用来阴人的隔离墙网格序列号
    wall_cell = wall_y * BOARD_SIZE + wall_x
    # 从上面的字典里迅速提取出此时方向映射成的标号。
    side_idx = SIDE_TO_INDEX[wall_side]
    
    # 融合时刻！你把这里看做是个三阶魔方数字密码：
    # （棋子坑位）是百位级别代表，（墙壁坑位）是十位级别，（四面墙向）是个位底座。
    # 它们以严格的 196 (49*4) 和 4 作为乘法跳板地基堆叠在一起！！只要数字互不重合污染，这把加密锁就造好了。
    return to_cell * (NUM_CELLS * NUM_SIDES) + wall_cell * NUM_SIDES + side_idx


def decode_action(action_int: int, piece_x: int, piece_y: int) -> Tuple[int, int, int, int, int, int, str]:
    """
    它就是那个刚在矩阵里厮杀完出来的解码器！
    AI 朝它大吼一句只有它听得懂的一组纯乱码数字（比如 4399）。
    这个函数负责拿起解剖刀：用数学除法极其凌厉地把它扒壳脱水，重新拼回“人能看懂的东西”给下面引擎执行操作去。
    之所以还要接收外界现在的 'piece_x' 跟 'y'，那是因为在这个体系里，9604 号空间根本无法包含“我到底以前站哪”，所以我干脆放弃压缩它，而是让你在解码时顺带把旧住址发我。
    """
    # 这个防御手段非常高阶：确保传入的参数一定是干干净净的一个纯血 Python 数学整数。
    if not isinstance(action_int, (int, np.integer)):
        raise TypeError(f"action_int must be int, got {type(action_int).__name__}")
    # 有时候虽然满足判断，但其实是个极易引发麻烦的 np.int64，为了最彻底洗净它，用 int() 活剥一层皮
    action_int = int(action_int)
    
    # 防止系统内存溢出越界。你叫嚣了一句要用“第一万号绝招”？抱歉咱们宇宙中不存在第一万种招法，当场拉闸阻挡并报警输出！
    if action_int < 0 or action_int >= ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_int {action_int} out of range [0, {ACTION_SPACE_SIZE})"
        )
    
    # 【逆向核心数学剥离术】：
    # 因为我们在上面把方向当做了基石的“个位”，占有度仅仅为“这四种”。
    # 所以我这时候拿出除法的【取余模块 `%` 】，对 4 疯狂开刀。
    # 多出来挤不进去除法的零头碎渣是什么？当然就是那个最细碎的方向编号啊！！直接一把收下。
    side_idx = action_int % NUM_SIDES
    # 取了余数之后剩下的部分怎么办？
    # 我用极其暴力的整数除法【`//`】将整个巨大天坑斩去了最低维的个位地基。拿到了商数（remainder）。这团大头肉里现在只包含纯洁的棋盘网格位了。
    remainder = action_int // NUM_SIDES
    
    # 咱们继续趁热打铁。上一级中途占了巨大比重的是墙坑号（最大 49）。我们如法炮制：对 49 取模收刮残余，墙的网格总序列号就在里面！
    wall_cell = remainder % NUM_CELLS
    # 原本商出来的整数继续被整除 49 斩杀掉之后，最后那块可怜的最核心大肥肉露出了原形——就只是一直端坐最高地基统治地位的：落子门牌号。
    to_cell = remainder // NUM_CELLS

    # 解剖到最后一步，把大块的线性网格号（比如 17 号地皮）重新还原到图纸（坐标 x 和 y）。
    # 这也是基础数学：比如用 17 被 7 整除，商 2 说明它是第二行。
    to_x = to_cell % BOARD_SIZE
    to_y = to_cell // BOARD_SIZE
    # 同样无脑复制一份对于墙体的解封。
    wall_x = wall_cell % BOARD_SIZE
    wall_y = wall_cell // BOARD_SIZE
    
    # 拿着刚才通过对 4 发泄拿到的尾端数字（0 到 3 之间的那点可怜的渣渣），跑去找反查大门要到了属于它真正的四方位英语代名词
    wall_side = INDEX_TO_SIDE[side_idx]

    # 一巴掌把 7 个拆出来的、被降解洗清过无数遍的无脑元素装进那个括号组成的特大长条返回管道里扔回去。
    return (piece_x, piece_y, to_x, to_y, wall_x, wall_y, wall_side)


def get_action_mask(env: WallGoEnv) -> np.ndarray:
    """
    这行代码掌握的是 AI 学习路上的最高级权威判卷官生杀大权：“动作屏蔽卡”。
    就算 AI 有着绝世聪明的脑子，想要在一张已经摆满了墙的下落地点，强行填一堵新墙卡死对方。
    这个卡片就起作用了：它会在那个违规的对应数组序号里画个冷酷的 False。
    底层的 PyTorch 等拿到这个打分卡以后，只要看见带有 False 的格子，就不可能在这一局中允许这个 AI 做出那个傻逼的判定而废掉这一整盘局了。只能考虑剩下的 True 路线。
    """
    # 第一步啥也不想直接上来直接创造一份 9604 个判断位的原始终极空白试炼卡。
    # 用了极其奢侈昂贵的 dtype=np.bool_ 去保证这里面的内存每一份每一秒都是纯正严苛且只占用一点点的布尔格式 True 或 False（这叫极度克制）。
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    
    # 如果说裁判官判案判到一半，接到了外太空的通透传音：“不用判卷了游戏早他妈完了！”
    # 直接放弃计算交个长满了 9604 个全军覆没大红叉的空卷子光速跑路。
    if env.done:
        return mask

    # 找出来到底谁还在战场上苟活等着我的神圣执裁，找出它是谁
    cur = env.current_player
    # 让系统报告一下，它的棋子如今在这浩大残酷世界中具体的苟活 x 和 y 坐标位置。
    pieces = env.get_player_pieces(cur)
    # 查阅环境底层的花名册：问问天地间现在到底还有哪些网格没有插上坚强的长城之墙，空余可用？
    wall_slots = env._available_walls  # 该函数由于提前去重非常优异，是一个像 [(横1,纵2,"左向")] 的列表。

    # 如果说这天地之大已经全是被砌的严严实实的密不透风的高墙，咱们的无辜小兵还扔不出任何一堵了。
    # 那就直接连步都不用让他迈出了，直接无脑又丢一个被全员判决了红叉的试卷光速逃离此局！
    if not wall_slots:
        return mask

    # ================= 注意：到了这里，代码开始不择手段了，一切为了让这玩意跑在极限加速的状态下！ =================
    # 在早期的代码里，为了拼凑墙壁和走路所有的合法序列，我们写了傻缺的 for for 循环嵌套了几百万次，拖垮了整个计算引擎。
    # 现在直接动用 numpy 这个大杀器，在一开始，通过仅仅只有一层的小遍历：
    # 直接就把所有“还能放新墙的网格坑”强悍精准地映射成了他们在 `9604` 分配表中独属于自己的一端残片小偏移值！
    wall_indices = np.array(
        [(wy * BOARD_SIZE + wx) * NUM_SIDES + SIDE_TO_INDEX[ws]
         for wx, wy, ws in wall_slots],
        dtype=np.int32,  # 精准分配为 32位 整数节省带宽
    )

    # 接下来这块真的称得上是最华丽的谢幕舞：
    # 既然此时棋子还活着，我们就只针对这唯一的棋子它当下被赋予的这枚孤胆位置。
    for px, py in pieces:
        # 去底层调用超硬核广度搜索（BFS），查清：在四周重重红砖阻挡之下，他此刻极限最多只能跨 2 步，还能钻进哪个夹缝活下去？
        # 把所有的生门路线全部给拿回来变成了 `valid_moves` 列表。
        valid_moves = env.get_valid_moves(px, py)
        # 最后，遍历这些生路。
        for mx, my in valid_moves:
            # 再一次用我们一开始那个三阶层级的密码：算出你要走的这块生门格在最巨大地表上到底代表哪一座大厦的高位基数。
            to_offset = (my * BOARD_SIZE + mx) * (NUM_CELLS * NUM_SIDES)
            # 全场最为狂妄也是最能立功的一行。完全不利用任何 Python 源生极端的 For 去填充。
            # 直接靠着刚刚建好的 `to_offset` 座高楼为基点，拉着那帮残片数组直接进行广播加法！！
            # 也就意味着这瞬间同时在成百上千个小格子里狠狠地用绿颜料给全员画出了个 True（也就是：可通行！！）！
            mask[to_offset + wall_indices] = True

    # 大事成功落定。把这篇涂满了红差和一点点微弱生机绿勾卡的巨大 9604 张生死状原图送交顶层判罚殿定夺。
    return mask
