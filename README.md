# WallGo RL Training

Reinforcement learning environment for the board game **Wall Go**.

## Game Rules

### Overview

Wall Go is a territory game played on a **7x7 grid**. Players take turns moving a piece and placing walls to divide the board. When all players are completely separated by walls, the game ends and whoever controls the most territory wins.

### Setup

- **Board**: 7x7 grid of cells.
- **Players**: 2 players (RED and BLUE). RED goes first.
- **Starting positions**: RED starts at top-center (column 3, row 1). BLUE starts at bottom-center (column 3, row 5).

```
  0 1 2 3 4 5 6
0 . . . . . . .
1 . . . R . . .
2 . . . . . . .
3 . . . . . . .
4 . . . . . . .
5 . . . B . . .
6 . . . . . . .
```

### Turn Structure

Each turn consists of **two mandatory actions**, in order:

1. **Move** your piece (0, 1, or 2 steps)
2. **Place one wall** on any open cell edge

Both actions are required every turn. You cannot skip either.

### Movement

- A piece moves **orthogonally** (up, down, left, right) — no diagonal movement.
- A piece can move **0, 1, or 2 steps** per turn.
  - 0 steps (staying in place) is a valid move.
  - Each step must be to an **adjacent** cell with no wall blocking the path.
  - The path is computed via BFS, so you can change direction mid-move (e.g. one step right then one step up).
- A piece **cannot** move onto a cell occupied by another piece.
- A piece **cannot** pass through walls.

### Wall Placement

- After moving, the player **must** place exactly one wall on any cell edge that doesn't already have a wall.
- A wall blocks passage between two adjacent cells. For example, placing a wall on the **right** side of cell (2, 3) blocks movement between (2, 3) and (3, 3).
- Walls are **permanent** — once placed, they cannot be removed or moved.
- Walls are placed on a **single edge** of a single cell (top, right, bottom, or left). This is different from Quoridor where walls span two cells.
- There is **no limit** on the total number of walls a player can place over the course of the game.
- Board edges (e.g. the top edge of row 0) already act as natural walls — pieces cannot leave the board. Placing a wall on a board edge is legal but has no effect.

### Game End

The game ends when **no player can reach any other player** through unblocked paths. In other words, the walls have completely separated all players into disconnected regions of the board.

### Scoring

When the game ends, each player's **territory** is calculated:

- Territory = the number of cells reachable from the player's piece via flood fill (including the cell the piece stands on).
- Reachable means connected through cells with no walls blocking the path between them.
- Empty cells that no player can reach are not counted for anyone.

**The player with the largest territory wins.** If territories are equal, the game is a tie.

### Example

```
Turn 1 (RED):  Move piece from (3,1) to (3,2). Place wall on right side of (3,2).
Turn 2 (BLUE): Move piece from (3,5) to (3,4). Place wall on left side of (3,4).
...
Game ends when walls fully separate RED and BLUE.
Final score: RED controls 25 cells, BLUE controls 24 cells → RED wins.
```

## Code Structure

`wallgo.py` contains the full game environment:

- `WallGoEnv` — the main environment class
  - `reset()` — start a new game, returns initial state
  - `step(action)` — execute one turn (move + wall), returns `(state, reward, done, info)`
  - `get_legal_actions()` — list all valid actions for the current player
  - `encode_state()` — numeric `[6, 7, 7]` tensor representation of the board
  - `clone()` — deep copy for MCTS / lookahead
- `UnionFind` — disjoint set for efficient connectivity and territory checks
- `Player` — enum (RED, BLUE, GREEN, YELLOW)
- `CellData` — per-cell state (occupant + walls)

### Action Format

Each action is a 7-tuple:

```python
(from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side)
```

- `from_x, from_y` — current position of the piece to move
- `to_x, to_y` — destination (can equal from for 0-step move)
- `wall_x, wall_y, wall_side` — cell and edge to place the wall (`'top'`, `'right'`, `'bottom'`, or `'left'`)

### State Encoding

`encode_state()` returns a `[6, 7, 7]` tensor with channels:

| Channel | Meaning |
|---------|---------|
| 0 | Current player's pieces (1 where piece is) |
| 1 | Opponent pieces |
| 2 | Walls on top edge |
| 3 | Walls on right edge |
| 4 | Walls on bottom edge |
| 5 | Walls on left edge |
