from enum import Enum
from collections import deque
from typing import List, Tuple, Set, Dict, Optional

class Player(Enum):
    RED = 'RED'
    BLUE = 'BLUE'
    GREEN = 'GREEN'
    YELLOW = 'YELLOW'

class GamePhase(Enum):
    PLACEMENT = 'PLACEMENT'
    ACTION_SELECT = 'ACTION_SELECT'
    ACTION_MOVE = 'ACTION_MOVE'
    ACTION_WALL = 'ACTION_WALL'
    GAME_OVER = 'GAME_OVER'

class CellData:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.occupant: Optional[Player] = None
        # Walls map a direction to the player who placed it
        self.walls: Dict[str, Optional[Player]] = {
            'top': None, 'right': None, 'bottom': None, 'left': None
        }

class WallGoEnv:
    def __init__(self, size: int = 7):
        self.size = size
        self.board = self._create_initial_board()
        self.active_players: List[Player] = []
        self.current_player_idx: int = 0
        self.done: bool = False

    def reset(self, players: List[Player] = [Player.RED, Player.BLUE]):
        """Starts a new game and places the initial pieces."""
        self.board = self._create_initial_board()
        self.active_players = players
        self.current_player_idx = 0
        self.done = False
        
        # Basic 2-player setup for 7x7
        if len(players) == 2:
            self.board[1][3].occupant = players[0] # RED Top Middle
            self.board[5][3].occupant = players[1] # BLUE Bottom Middle
            
        return self._get_state()

    def _get_state(self):
        """Returns the current board state for the AI to look at."""
        # In a real neural net, you'd convert this to a 3D matrix (tensors)
        return {
            "board": self.board,
            "current_player": self.active_players[self.current_player_idx]
        }

    def step(self, from_x: int, from_y: int, to_x: int, to_y: int, wall_x: int, wall_y: int, wall_side: str):
        """Executes one full turn (move piece + place wall) and returns (state, reward, done, info)."""
        if self.done:
            return self._get_state(), 0, True, {"error": "Game already over"}

        current_player = self.active_players[self.current_player_idx]

        # 1. Validate the piece belongs to the current player
        if not self.is_valid_coordinate(from_x, from_y) or self.board[from_y][from_x].occupant != current_player:
            # Invalid action: instantly penalize and end the episode
            self.done = True
            return self._get_state(), -1, True, {"error": "Invalid piece selected"}

        # 2. Validate and Execute Move
        valid_moves = self.get_valid_moves(from_x, from_y)
        if (to_x, to_y) not in valid_moves:
            self.done = True
            return self._get_state(), -1, True, {"error": "Invalid move destination"}

        self.board[from_y][from_x].occupant = None
        self.board[to_y][to_x].occupant = current_player

        # 3. Validate and Execute Wall Placement
        if not self.is_valid_coordinate(wall_x, wall_y) or self.board[wall_y][wall_x].walls.get(wall_side) is not None:
            self.done = True
            return self._get_state(), -1, True, {"error": "Invalid wall placement"}

        self.place_wall(wall_x, wall_y, wall_side, current_player)

        # 4. Check End Condition
        self.done = self.check_game_end_condition(self.active_players)
        
        reward = 0
        info = {}
        if self.done:
            # Game over! Calculate territory to see who won
            scores = self.calculate_scores(self.active_players)
            # Simple reward: +1 for win, -1 for loss, 0 for tie
            my_score = scores.get(current_player, 0)
            opponent_score = max([s for p, s in scores.items() if p != current_player])
            
            if my_score > opponent_score:
                reward = 1
                info["result"] = "win"
            elif my_score < opponent_score:
                reward = -1
                info["result"] = "loss"
            else:
                reward = 0
                info["result"] = "tie"
        else:
            # Switch turn if game isn't over
            self.current_player_idx = (self.current_player_idx + 1) % len(self.active_players)

        return self._get_state(), reward, self.done, info

    def _create_initial_board(self) -> List[List[CellData]]:
        return [[CellData(x, y) for x in range(self.size)] for y in range(self.size)]

    def is_valid_coordinate(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def place_wall(self, x: int, y: int, side: str, player: Player):
        """Places a wall and updates the adjacent cell's wall reference."""
        if not self.is_valid_coordinate(x, y):
            return

        self.board[y][x].walls[side] = player

        # Update adjacent cell
        if side == 'top' and y > 0:
            self.board[y - 1][x].walls['bottom'] = player
        elif side == 'bottom' and y < self.size - 1:
            self.board[y + 1][x].walls['top'] = player
        elif side == 'left' and x > 0:
            self.board[y][x - 1].walls['right'] = player
        elif side == 'right' and x < self.size - 1:
            self.board[y][x + 1].walls['left'] = player

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
        valid_moves = []
        queue = deque([((start_x, start_y), 0)])
        visited = {(start_x, start_y)}
        
        valid_moves.append((start_x, start_y)) # 0 steps is valid

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            (cx, cy), dist = queue.popleft()
            
            if dist >= 2:
                continue

            for dx, dy in directions:
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

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                if self.is_valid_coordinate(nx, ny):
                    current_cell = self.board[cy][cx]
                    next_cell = self.board[ny][nx]

                    if not self.is_blocked(current_cell, next_cell):
                        if (nx, ny) not in visited and next_cell.occupant is None:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        return visited

    def check_game_end_condition(self, active_players: List[Player]) -> bool:
        """BFS to check if any active player can reach another active player."""
        for p1 in active_players:
            queue = deque()
            visited = set()
            
            for y in range(self.size):
                for x in range(self.size):
                    if self.board[y][x].occupant == p1:
                        queue.append((x, y))
                        visited.add((x, y))

            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            while queue:
                cx, cy = queue.popleft()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    
                    if self.is_valid_coordinate(nx, ny):
                        current_cell = self.board[cy][cx]
                        next_cell = self.board[ny][nx]

                        if not self.is_blocked(current_cell, next_cell):
                            # Hit another player?
                            if next_cell.occupant and next_cell.occupant != p1 and next_cell.occupant in active_players:
                                return False # Connection found, game is not over
                            
                            if next_cell.occupant is None and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

        return True # No player can reach anyone else

    def calculate_scores(self, active_players: List[Player]) -> Dict[Player, int]:
        scores = {}
        for p in active_players:
            scores[p] = len(self.get_reachable_area(p))
        return scores