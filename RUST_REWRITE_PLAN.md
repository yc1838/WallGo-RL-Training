# Rust Game Engine Rewrite Plan

## Why

Python 环境模拟是训练瓶颈之一。当前 gym wrapper 跑 ~6,200 steps/sec，Rust 版本预计能到 300K-600K steps/sec（50-100x）。结合 CNN forward pass 的时间，总训练速度预计提升 **3-5x**。

## 核心思路

**只重写 game engine（wallgo.py）**，训练代码（train.py、wallgo_gym.py）保持 Python。

Rust 通过 PyO3 编译成 Python 模块 `wallgo_rs`，暴露和 `wallgo.py` 一样的 API。切换只需改一行 import。

## 什么不变

- `train.py` — 不动
- `wallgo_gym.py` — 只改 `from wallgo import WallGoEnv` → `from wallgo_rs import WallGoEnv`
- `action_encoding.py` — 不动（纯 Python 数学运算，不是瓶颈）
- `evaluate.py` — 不动
- **所有现有 tests** — 不动，改完之后跑同样的 144 个测试来验证正确性
- **benchmark.py** — 不动，改完前后各跑一次来量化加速

## 文件结构

```
wallgo_rs/
├── Cargo.toml          # Rust 项目配置 + PyO3 依赖
├── src/
│   ├── lib.rs          # PyO3 模块入口，暴露 Python classes
│   ├── board.rs        # CellData, board 创建
│   ├── game.rs         # WallGoEnv 主逻辑 (reset, step, legal actions)
│   ├── union_find.rs   # UnionFind (connectivity + scoring)
│   └── types.rs        # Player enum, constants (SIDES, DIRECTIONS, etc.)
```

构建后生成 `wallgo_rs.so`（macOS 上是 `.dylib`），Python 直接 `import wallgo_rs`。

## 分步实施

### Step 0: Rust 项目脚手架
- 初始化 `wallgo_rs/` with `Cargo.toml`（PyO3 + maturin）
- 配置 `maturin` 构建工具（pip install maturin）
- 创建空的 `lib.rs`，验证 `maturin develop` 能编译出 Python 模块

### Step 1: Types + Constants
**Rust:** `types.rs`
- `Player` enum (RED, BLUE, GREEN, YELLOW) — 暴露给 Python，行为和 Python 的 `Player` enum 一致
- `SIDES`, `OPPOSITE`, `DIRECTIONS`, `SIDE_DELTA` — 常量

**验证:** import wallgo_rs; 检查 `Player.RED` 等存在

### Step 2: Board + CellData
**Rust:** `board.rs`
- `CellData` struct: x, y, occupant, walls (HashMap or fixed array)
- `create_initial_board(size) -> Vec<Vec<CellData>>`

**验证:** 能创建 7x7 board，cells 有正确的坐标

### Step 3: UnionFind
**Rust:** `union_find.rs`
- `UnionFind::new(n)`, `find(x)`, `union(a, b)`, `connected(a, b)`

**验证:** 和 Python 版本做 10 组随机 union/find 操作对比

### Step 4: WallGoEnv — 核心
**Rust:** `game.rs` + `lib.rs`
- `WallGoEnv` struct 暴露给 Python (#[pyclass])
- 实现所有 public methods:
  - `reset(players)` → state dict
  - `step(action)` → (state, reward, done, info)
  - `get_legal_actions()` → list of tuples
  - `get_valid_moves(x, y)` → list of tuples
  - `get_valid_wall_placements()` → list of tuples
  - `get_player_pieces(player)` → list of tuples
  - `encode_state()` → nested list (same [6,7,7] format)
  - `clone()` → new WallGoEnv
  - `calculate_scores(players)` → dict
  - `check_game_end_condition(players)` → bool
  - `place_wall(x, y, side, player)`
  - Properties: `done`, `turn_count`, `current_player`, `size`, `active_players`, `_available_walls`, `board`

**关键：** API 签名必须和 Python 版本完全一致，包括参数名和返回值格式。

### Step 5: 集成切换
- `wallgo_gym.py` 改 import: `from wallgo_rs import WallGoEnv, Player`
- `action_encoding.py` 改 import: `from wallgo_rs import WallGoEnv, SIDES`
- 保留 `wallgo.py` 不删除（作为参考和回退）

### Step 6: 验证 + Benchmark
- 跑全部 144 个现有测试 → 必须全过
- 跑 `benchmark.py` → 对比 Python vs Rust 数字
- 跑 `train.py --steps 1000` → 确认训练正常

## 现有测试的作用

| 测试文件 | 重写后还有用吗 | 说明 |
|---------|-------------|------|
| `tests/test_action_encoding.py` (36 tests) | **有用** | 直接验证 Rust env 的 `get_legal_actions()` 和 mask 是否正确 |
| `tests/test_wallgo_gym.py` (47 tests) | **有用** | gym wrapper 接 Rust env，验证 obs/reward/done 全链路 |
| `tests/test_evaluate.py` (21 tests) | **有用** | 验证 baseline agents 在 Rust env 上能跑完整局 |
| `tests/test_train.py` (12 tests) | **有用** | 验证 PPO 训练在 Rust env 上能跑 |
| `tests/test_optimizations.py` (28 tests) | **有用** | 50 局回归测试、clone 正确性、mask 等价性 |
| `benchmark.py` | **有用** | 改完前后各跑一次，量化加速倍数 |

**所有 144 个测试 + benchmark 全部保留，一个不删。** 这就是 TDD 的价值 — 重写底层引擎时，测试就是安全网。

## 预期结果

| Metric | Python (当前) | Rust (预期) | 加速 |
|--------|-------------|------------|------|
| Raw game sim | 195 games/sec | ~10,000 games/sec | ~50x |
| get_action_mask() | 31,271/sec | ~500,000/sec | ~15x |
| clone() | 81,224/sec | ~2,000,000/sec | ~25x |
| Gym wrapper (训练实际瓶颈) | 6,192 steps/sec | ~100,000 steps/sec | ~15x |
| **100K 步训练时间** | **~8 min** | **~2 min** | **~4x** |

注意：训练加速会低于纯 env 加速，因为 CNN forward/backward pass 时间不变（那是 PyTorch 的活）。

## 依赖

- `rustup` (Rust toolchain)
- `maturin` (pip install maturin) — Rust→Python 构建工具
- PyO3 (Cargo dependency) — Rust↔Python 绑定
