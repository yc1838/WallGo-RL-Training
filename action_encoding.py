"""Fixed action space encoding for WallGo RL training.

Maps between the variable-length action tuples from WallGoEnv and a fixed
integer action space of size 9,604 (49 destinations × 49 wall cells × 4 sides).

In 2-player mode each player has exactly 1 piece, so (from_x, from_y) is
implicit — we only encode (to_x, to_y, wall_x, wall_y, wall_side).
"""

import numpy as np
from typing import Tuple
from wallgo import WallGoEnv, SIDES

BOARD_SIZE = 7
NUM_CELLS = BOARD_SIZE * BOARD_SIZE  # 49
NUM_SIDES = 4

SIDE_TO_INDEX = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}
INDEX_TO_SIDE = {v: k for k, v in SIDE_TO_INDEX.items()}

ACTION_SPACE_SIZE = NUM_CELLS * NUM_CELLS * NUM_SIDES  # 9604


def encode_action(to_x: int, to_y: int, wall_x: int, wall_y: int, wall_side: str) -> int:
    """Encode a (to_x, to_y, wall_x, wall_y, wall_side) tuple into a single integer.

    Returns an int in [0, ACTION_SPACE_SIZE).
    """
    if wall_side not in SIDE_TO_INDEX:
        raise ValueError(f"Invalid wall side: {wall_side!r}")
    to_cell = to_y * BOARD_SIZE + to_x
    wall_cell = wall_y * BOARD_SIZE + wall_x
    side_idx = SIDE_TO_INDEX[wall_side]
    return to_cell * (NUM_CELLS * NUM_SIDES) + wall_cell * NUM_SIDES + side_idx


def decode_action(action_int: int, piece_x: int, piece_y: int) -> Tuple[int, int, int, int, int, int, str]:
    """Decode an integer action back into a full 7-tuple for WallGoEnv.step().

    Returns (from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side).
    """
    if not isinstance(action_int, (int, np.integer)):
        raise TypeError(f"action_int must be int, got {type(action_int).__name__}")
    action_int = int(action_int)
    if action_int < 0 or action_int >= ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_int {action_int} out of range [0, {ACTION_SPACE_SIZE})"
        )
    side_idx = action_int % NUM_SIDES
    remainder = action_int // NUM_SIDES
    wall_cell = remainder % NUM_CELLS
    to_cell = remainder // NUM_CELLS

    to_x = to_cell % BOARD_SIZE
    to_y = to_cell // BOARD_SIZE
    wall_x = wall_cell % BOARD_SIZE
    wall_y = wall_cell // BOARD_SIZE
    wall_side = INDEX_TO_SIDE[side_idx]

    return (piece_x, piece_y, to_x, to_y, wall_x, wall_y, wall_side)


def get_action_mask(env: WallGoEnv) -> np.ndarray:
    """Return a boolean mask of shape (ACTION_SPACE_SIZE,) for legal actions.

    True at index i means action i is legal for the current player.
    Returns all-False if the game is over.

    Optimized: pre-computes wall index array once, then combines with each
    move destination via vectorized offset, avoiding per-action tuple creation.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    if env.done:
        return mask

    cur = env.current_player
    pieces = env.get_player_pieces(cur)
    wall_slots = env._available_walls  # list of (wx, wy, ws) tuples

    if not wall_slots:
        return mask

    # Pre-compute wall indices as a numpy array (done once per call)
    wall_indices = np.array(
        [(wy * BOARD_SIZE + wx) * NUM_SIDES + SIDE_TO_INDEX[ws]
         for wx, wy, ws in wall_slots],
        dtype=np.int32,
    )

    for px, py in pieces:
        valid_moves = env.get_valid_moves(px, py)
        for mx, my in valid_moves:
            to_offset = (my * BOARD_SIZE + mx) * (NUM_CELLS * NUM_SIDES)
            mask[to_offset + wall_indices] = True

    return mask
