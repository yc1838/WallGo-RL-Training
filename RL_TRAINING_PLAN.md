# WallGo RL Training — Implementation Plan

## Context

We have a working WallGo game engine (`wallgo.py`) that lets two players take turns moving pieces and placing walls. We want to teach a computer to play this game well using **Reinforcement Learning (RL)** — the same family of techniques that powered AlphaGo and OpenAI Five.

Right now, the game engine is like a chess board with rules — it works, but there's no way to plug in a "brain" (neural network) that can learn from playing. This plan bridges that gap.

### Development approach: Tests first

Every component below follows **test-driven development** — we write the tests *before* writing the implementation. This forces us to think clearly about what each function should do (its inputs, outputs, and edge cases) before we write a single line of logic. Tests cover:

- **(A) Happy path** — does it work when everything is normal?
- **(B) Null/empty/missing** — what happens with no legal moves, empty boards, missing fields?
- **(C) Type/shape mismatch** — does it reject wrong input shapes or types?
- **(D) Boundary values** — does action index 0 work? Does 9,603 (the max)? What about -1 or 9,604?

---

## Step 0: Project Setup

**File to create:** `requirements.txt`

### What this means

Before writing any code, we need to install the tools (Python libraries) that our project depends on. This is like gathering ingredients before cooking.

### What to build

```
pytest
numpy
gymnasium
stable-baselines3[extra]
sb3-contrib
tensorboard
torch
```

---

## Step 1: Create a Fixed Action Space with Legal Action Masking

**Test file:** `tests/test_action_encoding.py` (written first)
**Implementation file:** `action_encoding.py`

### What this means

Right now, every turn the game says "here are your 900 possible moves, pick one." But next turn that list might have 850 moves, and the turn after that 920. The list is a different size and in a different order each time.

An RL agent is like a student taking a multiple-choice test. It needs the **same number of answer choices every time** (say, A through Z), and we just cross out the ones that aren't allowed this turn. This is called a **fixed action space** with **action masking**.

### Why we need this

Neural networks output a fixed-size vector (like a row of numbers). If we have 9,604 possible actions, the network always outputs 9,604 numbers — one "confidence score" per action. We then zero out the illegal ones and pick the highest-scoring legal action. Without this fixed mapping, the network can't learn which actions are good because the meaning of "action #5" changes every turn.

### What to build

- Define all possible actions as a flat list of integers (0 to N-1)
- Since each player has 1 piece in 2-player mode, we can drop `from_x, from_y` (it's always where your piece already is)
- Action = `(to_x, to_y, wall_x, wall_y, wall_side)` → encode as a single integer
- Encoding: `to_cell * (7 * 7 * 4) + wall_cell * 4 + side_index`
  - `to_cell` = `to_y * 7 + to_x` (0–48)
  - `wall_cell` = `wall_y * 7 + wall_x` (0–48)
  - `side_index` = index into `['top', 'right', 'bottom', 'left']` (0–3)
  - Total fixed action space size: **49 × 49 × 4 = 9,604**
- Build a `get_action_mask()` function that returns a boolean array of length 9,604 — `True` for legal actions, `False` for illegal ones
- Build `encode_action(action_tuple) → int` and `decode_action(int, current_piece_pos) → action_tuple` converters

### Tests to write first

| Category | Test |
|----------|------|
| (A) Happy path | `encode_action` → `decode_action` round-trip returns the original tuple |
| (A) Happy path | `get_action_mask` returns exactly the same legal actions as `get_legal_actions()` |
| (B) Empty | Mask has at least one `True` entry whenever the game isn't over |
| (C) Shape | Mask is always length 9,604, dtype bool |
| (D) Boundary | Action index 0 decodes correctly; index 9,603 decodes correctly |
| (D) Boundary | `decode_action(-1)` and `decode_action(9604)` raise errors |

---

## Step 2: Wrap the Environment for Gymnasium (with Reward Shaping)

**Test file:** `tests/test_wallgo_gym.py` (written first)
**Implementation file:** `wallgo_gym.py`

### What this means

RL libraries (the tools researchers use to train agents) all speak a common language called the **Gymnasium API**. It's like a universal plug — any game that fits this plug can be trained by any RL algorithm. Our game engine doesn't fit the plug yet.

We also add **reward shaping** here — small hints during the game so the agent doesn't have to wait until the very end to learn what's good. Imagine trying to learn chess but nobody tells you anything until the game is over — you wouldn't know which of your 40 moves were good or bad. Small bonuses for controlling territory and staying mobile fix this.

### Why we need this

Without the Gymnasium wrapper, we'd have to write all the training code from scratch. With it, we can use battle-tested libraries like Stable-Baselines3 that handle the hard math for us. Without reward shaping, the agent needs millions more games to learn because it only gets feedback at game end.

### What to build

- A `WallGoGymEnv` class that inherits from `gymnasium.Env`
- Define `observation_space`: a `Box(0, 1, shape=(6, 7, 7))` — tells the library our board state is a 6×7×7 grid of 0s and 1s
- Define `action_space`: a `Discrete(9604)` — tells the library there are 9,604 possible actions
- Implement `reset()` → returns `(observation, info)`
- Implement `step(action_int)` → decodes the integer action, calls the real `wallgo.py` step, returns `(observation, reward, terminated, truncated, info)`
- Implement `action_masks()` → returns the boolean mask from Step 1 (standard interface for masked action spaces)
- Add a `max_turns` truncation (e.g., 200 turns) so games don't run forever during early training
- **Reward shaping** (optional, controlled by a flag):
  - **Territory differential**: `0.01 * (my_territory - opponent_territory) / total_cells`
  - **Mobility bonus**: `0.005 * (my_moves / max_possible_moves)`
  - Terminal reward stays at +1/-1/0 — shaping rewards are just gentle nudges

### Tests to write first

| Category | Test |
|----------|------|
| (A) Happy path | `reset()` returns observation of shape `(6, 7, 7)` with values in `[0, 1]` |
| (A) Happy path | `step()` with a legal action returns valid `(obs, reward, term, trunc, info)` |
| (A) Happy path | `action_masks()` shape is `(9604,)` and dtype is bool |
| (A) Happy path | Game reaches `terminated=True` when players are fully separated |
| (B) Empty | `step()` after game is done returns `terminated=True` without crashing |
| (C) Shape | Observation always has shape `(6, 7, 7)` regardless of game state |
| (D) Boundary | Game truncates at `max_turns` with `truncated=True` |
| (D) Boundary | Shaping rewards are small (< 0.05 absolute) while terminal is ±1 |
| Special | Passes `gymnasium.utils.env_checker.check_env()` validation |
| Special | Shaping can be toggled off — non-terminal rewards are exactly 0 when disabled |

---

## Step 3: Build a Self-Play Training Loop

**Test file:** `tests/test_train.py` (written first)
**Implementation file:** `train.py`

### What this means

In a two-player game, the agent needs an opponent to practice against. The best opponent is **itself** — this is called **self-play**. The agent plays both sides: it makes a move as RED, then switches hats and responds as BLUE. As it gets better, its opponent automatically gets better too, which pushes it to keep improving.

This is exactly how AlphaGo was trained — it played millions of games against copies of itself.

### Why we need this

If we train against a fixed opponent (like a random player), the agent will learn to beat *that specific opponent* but may play terribly against anything else. Self-play forces the agent to learn general strategies because its opponent adapts with it.

### What to build

- Use **PPO (Proximal Policy Optimization)** as the RL algorithm — it's the most reliable general-purpose algorithm and works well with action masking
- Use the `MaskablePPO` from `sb3-contrib` (Stable-Baselines3 extension) which natively supports our action masks
- Training loop:
  1. Agent plays against a copy of itself for N games
  2. Collect all the moves and outcomes
  3. Update the agent's neural network to do more of what worked and less of what didn't
  4. Every K updates, save a checkpoint of the agent
  5. Repeat
- The neural network architecture: a small CNN (convolutional neural network) that takes the 6×7×7 board state and outputs both:
  - A **policy** (probability distribution over 9,604 actions) — "what move should I make?"
  - A **value** (single number between -1 and +1) — "how likely am I to win from here?"
- Opponent selection: keep a pool of past checkpoints, randomly pick opponents from recent checkpoints (not just the latest) to prevent "forgetting" how to beat older strategies

### Tests to write first

| Category | Test |
|----------|------|
| (A) Happy path | Smoke test: `train.py --steps=100` completes without error |
| (A) Happy path | Checkpoint file is saved to the expected directory after training |
| (A) Happy path | A saved checkpoint can be loaded and used for inference |
| (D) Boundary | Training with `--steps=0` exits cleanly |

### Key dependencies
- `gymnasium` — environment interface
- `stable-baselines3` + `sb3-contrib` — PPO with action masking
- `torch` (PyTorch) — neural network backend

---

## Step 4: Add Evaluation and Metrics

**Test file:** `tests/test_evaluate.py` (written first)
**Implementation file:** `evaluate.py`

### What this means

During training, we need to know if the agent is actually getting smarter or just spinning its wheels. We set up opponents of known strength and periodically test the agent against them, like a student taking practice exams.

### Why we need this

Without evaluation, we could train for days and not realize the agent isn't learning (maybe the learning rate is wrong, or the reward shaping is misleading it). Regular evaluation catches these problems early.

### What to build

- **Random baseline**: An agent that picks a legal action at random. Any trained agent should crush this quickly.
- **Greedy baseline**: An agent that picks the move maximizing immediate territory gain. This is stronger but still predictable.
- **Metrics to track**:
  - Win rate vs random (should approach ~100%)
  - Win rate vs greedy (should steadily climb)
  - Win rate vs past self (should stay around 55% — if it's 50% it's not improving, if it's 90% something is wrong)
  - Average game length (should stabilize as the agent learns purposeful play)
  - Average territory differential at game end
- **TensorBoard logging**: all metrics written to TensorBoard so we can watch training progress in real-time graphs
- Run evaluation every N training steps automatically

### Tests to write first

| Category | Test |
|----------|------|
| (A) Happy path | Random agent always picks a legal action and completes a full game |
| (A) Happy path | Greedy agent always picks a legal action and completes a full game |
| (A) Happy path | `evaluate()` returns a dict with all expected metric keys |
| (D) Boundary | Evaluation with `num_games=1` works; `num_games=0` returns empty metrics |

---

## Step 5: Performance Optimizations (If Needed)

**Modify:** `wallgo.py`, `wallgo_gym.py`

### What this means

RL training requires playing millions of games. If each game takes 1 second, a million games = 11.5 days. If we can make each game take 0.1 seconds, that's 1.15 days. Speed matters a lot.

### Why we need this

The agent learns by trial and error across a huge number of games. Faster games = faster learning = better results in the same wall-clock time.

### What to build (only if training is too slow)

- Replace `copy.deepcopy()` with a manual `clone()` that only copies what's needed
- Use NumPy arrays instead of nested Python lists for the board state and action masks
- Cache and incrementally update the action mask instead of recomputing from scratch
- Consider vectorized environments (multiple games running in parallel)

---

## Verification Plan

### Automated (run after every step)

```bash
pytest tests/ -v
```

### Manual checkpoints

| Step | How to verify |
|------|---------------|
| 0 | `pip install -r requirements.txt` succeeds |
| 1 | `pytest tests/test_action_encoding.py` — all pass |
| 2 | `pytest tests/test_wallgo_gym.py` — all pass, including `check_env()` |
| 3 | `python train.py --steps=1000` — completes, checkpoint file appears |
| 4 | `python evaluate.py --model=latest_checkpoint` — prints metrics, TensorBoard logs appear |
| 5 | Benchmark: time 1,000 random games before and after optimizations |

---

## File Summary

| File | Purpose |
|------|---------|
| `wallgo.py` | Existing game engine (mostly unchanged) |
| `requirements.txt` | Python dependencies |
| `action_encoding.py` | Fixed action ↔ integer mapping + masking |
| `tests/test_action_encoding.py` | Tests for action encoding (written first) |
| `wallgo_gym.py` | Gymnasium wrapper + reward shaping |
| `tests/test_wallgo_gym.py` | Tests for gym environment (written first) |
| `train.py` | Self-play PPO training loop |
| `tests/test_train.py` | Smoke tests for training (written first) |
| `evaluate.py` | Evaluation against baselines + metrics |
| `tests/test_evaluate.py` | Tests for evaluation logic (written first) |
