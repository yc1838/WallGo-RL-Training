from enum import Enum
from collections import deque
from typing import List, Tuple, Set, Dict, Optional
import copy

SIDES = ('top', 'right', 'bottom', 'left')
OPPOSITE = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
# Map wall side to the (dx, dy) of the adjacent cell sharing that wall
SIDE_DELTA = {'top': (0, -1), 'bottom': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

class Player(Enum):
    RED = 'RED'
    BLUE = 'BLUE'
    GREEN = 'GREEN'
    YELLOW = 'YELLOW'

class CellData:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.occupant: Optional[Player] = None
        self.walls: Dict[str, Optional[Player]] = {
            'top': None, 'right': None, 'bottom': None, 'left': None
        }


# ======================================================================
# Union-Find (Disjoint Set) for fast connectivity checks
# ======================================================================

class UnionFind:
    """Weighted union-find with path compression. O(alpha(n)) per operation."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


class WallGoEnv:
    def __init__(self, size: int = 7):
        self.size = size
        self.board = self._create_initial_board()
        self.active_players: List[Player] = []
        self.current_player_idx: int = 0
        self.done: bool = False
        # Incremental tracking — avoids scanning the whole board each turn
        self._piece_positions: Dict[Player, List[Tuple[int, int]]] = {}
        self._available_walls: Set[Tuple[int, int, str]] = set()

    def reset(self, players: Optional[List[Player]] = None):
        """Starts a new game and places the initial pieces."""
        if players is None:
            players = [Player.RED, Player.BLUE]
        self.board = self._create_initial_board()
        self.active_players = list(players)
        self.current_player_idx = 0
        self.done = False

        # Initialize available wall set (all wall slots open at start)
        self._available_walls = set()
        for y in range(self.size):
            for x in range(self.size):
                for side in SIDES:
                    self._available_walls.add((x, y, side))

        # Initialize piece positions
        self._piece_positions = {p: [] for p in self.active_players}

        # Basic 2-player setup for 7x7
        if len(self.active_players) == 2:
            self._place_piece(self.active_players[0], 3, 1)
            self._place_piece(self.active_players[1], 3, 5)

        return self._get_state()

    def _place_piece(self, player: Player, x: int, y: int):
        """Place a piece and update the tracking dict."""
        self.board[y][x].occupant = player
        self._piece_positions[player].append((x, y))

    @property
    def current_player(self) -> Player:
        return self.active_players[self.current_player_idx]

    def clone(self) -> 'WallGoEnv':
        """Deep copy of the entire game state. Essential for MCTS / lookahead."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        """Returns both the raw board and a numeric encoding."""
        return {
            "board": self.board,
            "current_player": self.current_player,
            "numeric": self.encode_state(),
        }

    def encode_state(self) -> List[List[List[int]]]:
        """Numeric board encoding as plain nested lists (no numpy needed).

        Returns a [C, size, size] tensor-like structure with channels:
          0 - current player's pieces
          1 - opponent pieces (all opponents combined)
          2 - walls: top
          3 - walls: right
          4 - walls: bottom
          5 - walls: left
        Values are 0 or 1.
        """
        s = self.size
        cur = self.current_player
        # Create 6 channels of zeros
        state = [[[0] * s for _ in range(s)] for _ in range(6)]

        for y in range(s):
            for x in range(s):
                cell = self.board[y][x]
                if cell.occupant == cur:
                    state[0][y][x] = 1
                elif cell.occupant is not None:
                    state[1][y][x] = 1
                for idx, side in enumerate(SIDES):
                    if cell.walls[side] is not None:
                        state[2 + idx][y][x] = 1
        return state

    # ------------------------------------------------------------------
    # Legal action enumeration  (THE key addition for RL)
    # ------------------------------------------------------------------

    def get_player_pieces(self, player: Player) -> List[Tuple[int, int]]:
        """Return (x, y) positions of all pieces belonging to player. O(1) lookup."""
        return list(self._piece_positions[player])

    def get_valid_wall_placements(self) -> List[Tuple[int, int, str]]:
        """Return all (x, y, side) where a wall can still be placed. O(k) where k = open slots."""
        return list(self._available_walls)

    def get_legal_actions(self) -> List[Tuple[int, int, int, int, int, int, str]]:
        """Enumerate every legal (from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side).

        This is the FULL action list the current player can choose from.
        For RL, an agent picks an index into this list each turn.
        """
        if self.done:
            return []

        cur = self.current_player
        pieces = self.get_player_pieces(cur)
        wall_slots = self.get_valid_wall_placements()
        actions = []

        for (px, py) in pieces:
            for (mx, my) in self.get_valid_moves(px, py):
                # Temporarily execute the move so wall validity reflects
                # the board state AFTER moving (piece is now at mx, my).
                self.board[py][px].occupant = None
                self.board[my][mx].occupant = cur

                for (wx, wy, ws) in wall_slots:
                    # Wall slot must still be free (move doesn't affect walls,
                    # so the pre-computed list is still valid).
                    actions.append((px, py, mx, my, wx, wy, ws))

                # Undo temporary move
                self.board[my][mx].occupant = None
                self.board[py][px].occupant = cur

        return actions

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: Tuple[int, int, int, int, int, int, str]):
        """Executes one full turn and returns (state, reward, done, info).

        action: (from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side)
        """
        if self.done:
            return self._get_state(), 0, True, {"error": "Game already over"}

        from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side = action
        cur = self.current_player

        # 1. Validate piece ownership
        if not self.is_valid_coordinate(from_x, from_y) or self.board[from_y][from_x].occupant != cur:
            return self._get_state(), -1, False, {"error": "Invalid piece selected"}

        # 2. Validate and execute move
        valid_moves = self.get_valid_moves(from_x, from_y)
        if (to_x, to_y) not in valid_moves:
            return self._get_state(), -1, False, {"error": "Invalid move destination"}

        self.board[from_y][from_x].occupant = None
        self.board[to_y][to_x].occupant = cur
        # Update piece position tracking
        pieces = self._piece_positions[cur]
        for i, (px, py) in enumerate(pieces):
            if px == from_x and py == from_y:
                pieces[i] = (to_x, to_y)
                break

        # 3. Validate and execute wall placement
        if wall_side not in SIDES:
            # Undo move + piece tracking
            self._undo_move(cur, from_x, from_y, to_x, to_y)
            return self._get_state(), -1, False, {"error": f"Invalid wall side: {wall_side}"}

        if not self.is_valid_coordinate(wall_x, wall_y) or self.board[wall_y][wall_x].walls[wall_side] is not None:
            self._undo_move(cur, from_x, from_y, to_x, to_y)
            return self._get_state(), -1, False, {"error": "Invalid wall placement"}

        self.place_wall(wall_x, wall_y, wall_side, cur)

        # 4. Check end condition + score in one pass (single Union-Find build)
        uf = self._build_union_find()
        self.done = self._check_end_with_uf(uf, self.active_players)

        reward = 0
        info = {}
        if self.done:
            scores = self._scores_with_uf(uf, self.active_players)
            my_score = scores.get(cur, 0)
            opponent_score = max(s for p, s in scores.items() if p != cur)

            if my_score > opponent_score:
                reward = 1
                info["result"] = "win"
            elif my_score < opponent_score:
                reward = -1
                info["result"] = "loss"
            else:
                reward = 0
                info["result"] = "tie"
            info["scores"] = {p.value: s for p, s in scores.items()}
        else:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.active_players)

        return self._get_state(), reward, self.done, info

    def _create_initial_board(self) -> List[List[CellData]]:
        return [[CellData(x, y) for x in range(self.size)] for y in range(self.size)]

    def is_valid_coordinate(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _undo_move(self, player: Player, from_x: int, from_y: int, to_x: int, to_y: int):
        """Revert a piece move and its tracking data."""
        self.board[to_y][to_x].occupant = None
        self.board[from_y][from_x].occupant = player
        pieces = self._piece_positions[player]
        for i, (px, py) in enumerate(pieces):
            if px == to_x and py == to_y:
                pieces[i] = (from_x, from_y)
                break

    def place_wall(self, x: int, y: int, side: str, player: Player):
        """Places a wall and updates the adjacent cell + available wall tracking."""
        if not self.is_valid_coordinate(x, y) or side not in SIDES:
            return

        self.board[y][x].walls[side] = player
        self._available_walls.discard((x, y, side))

        # Update the neighbour cell's matching wall
        dx, dy = SIDE_DELTA[side]
        nx, ny = x + dx, y + dy
        if self.is_valid_coordinate(nx, ny):
            opp = OPPOSITE[side]
            self.board[ny][nx].walls[opp] = player
            self._available_walls.discard((nx, ny, opp))

    def is_blocked(self, from_cell: CellData, to_cell: CellData) -> bool:
        """Checks if movement is blocked by a wall between two adjacent cells."""
        if to_cell.x == from_cell.x and to_cell.y == from_cell.y - 1:
            return from_cell.walls['top'] is not None
        if to_cell.x == from_cell.x and to_cell.y == from_cell.y + 1:
            return from_cell.walls['bottom'] is not None
        if to_cell.x == from_cell.x - 1 and to_cell.y == from_cell.y:
            return from_cell.walls['left'] is not None
        if to_cell.x == from_cell.x + 1 and to_cell.y == from_cell.y:
            return from_cell.walls['right'] is not None
        return True

    def get_valid_moves(self, start_x: int, start_y: int) -> List[Tuple[int, int]]:
        """BFS to find valid moves (0, 1, or 2 steps)."""
        valid_moves = [(start_x, start_y)]  # 0 steps (stay) is valid
        queue = deque([((start_x, start_y), 0)])
        visited = {(start_x, start_y)}

        while queue:
            (cx, cy), dist = queue.popleft()

            if dist >= 2:
                continue

            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy

                if self.is_valid_coordinate(nx, ny):
                    current_cell = self.board[cy][cx]
                    target_cell = self.board[ny][nx]

                    if not self.is_blocked(current_cell, target_cell) and target_cell.occupant is None:
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            valid_moves.append((nx, ny))
                            queue.append(((nx, ny), dist + 1))

        return valid_moves

    def get_reachable_area(self, player: Player) -> Set[Tuple[int, int]]:
        """Flood fill to calculate territory size."""
        queue = deque()
        visited = set()

        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x].occupant == player:
                    queue.append((x, y))
                    visited.add((x, y))

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy

                if self.is_valid_coordinate(nx, ny):
                    current_cell = self.board[cy][cx]
                    next_cell = self.board[ny][nx]

                    if not self.is_blocked(current_cell, next_cell):
                        if (nx, ny) not in visited and next_cell.occupant is None:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

        return visited

    def _build_union_find(self) -> UnionFind:
        """Build a Union-Find over all cells, merging adjacent cells with no wall between them.

        Single O(n^2) pass replaces multiple BFS traversals.
        """
        s = self.size
        uf = UnionFind(s * s)
        for y in range(s):
            for x in range(s):
                cell = self.board[y][x]
                # Only need to check right and bottom to cover all edges once
                if x + 1 < s and cell.walls['right'] is None:
                    uf.union(y * s + x, y * s + (x + 1))
                if y + 1 < s and cell.walls['bottom'] is None:
                    uf.union(y * s + x, (y + 1) * s + x)
        return uf

    def _check_end_with_uf(self, uf: UnionFind, active_players: List[Player]) -> bool:
        """Check if all players are separated using a pre-built Union-Find."""
        s = self.size
        player_roots: Dict[Player, int] = {}
        for p in active_players:
            for (x, y) in self._piece_positions[p]:
                player_roots[p] = uf.find(y * s + x)
                break  # one piece per player is enough

        roots_seen: Dict[int, Player] = {}
        for p, root in player_roots.items():
            for other_root in roots_seen:
                if uf.connected(root, other_root):
                    return False
            roots_seen[root] = p
        return True

    def _scores_with_uf(self, uf: UnionFind, active_players: List[Player]) -> Dict[Player, int]:
        """Calculate territory scores using a pre-built Union-Find."""
        s = self.size

        component_size: Dict[int, int] = {}
        for y in range(s):
            for x in range(s):
                root = uf.find(y * s + x)
                component_size[root] = component_size.get(root, 0) + 1

        scores: Dict[Player, int] = {}
        for p in active_players:
            seen_roots: Set[int] = set()
            total = 0
            for (x, y) in self._piece_positions[p]:
                root = uf.find(y * s + x)
                if root not in seen_roots:
                    seen_roots.add(root)
                    total += component_size[root]
            scores[p] = total
        return scores

    def check_game_end_condition(self, active_players: List[Player]) -> bool:
        """Public API: check if all players are fully separated."""
        return self._check_end_with_uf(self._build_union_find(), active_players)

    def calculate_scores(self, active_players: List[Player]) -> Dict[Player, int]:
        """Public API: calculate territory scores."""
        return self._scores_with_uf(self._build_union_find(), active_players)