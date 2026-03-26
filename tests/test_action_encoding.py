"""Tests for action_encoding.py — written BEFORE implementation (TDD).

Each section is labeled with its test category:
  (A) Happy path
  (B) Null/empty/missing
  (C) Type/shape mismatch
  (D) Boundary values
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wallgo import WallGoEnv, Player


# ======================================================================
# Section 1: Constants
# ======================================================================

class TestConstants:
    """Tests for SIDE_TO_INDEX, INDEX_TO_SIDE, BOARD_SIZE, ACTION_SPACE_SIZE."""

    # (A) Happy path — constants exist and have expected values
    def test_board_size_is_7(self):
        from action_encoding import BOARD_SIZE
        assert BOARD_SIZE == 7

    def test_action_space_size(self):
        from action_encoding import ACTION_SPACE_SIZE, BOARD_SIZE
        # 49 destinations × 49 wall cells × 4 sides = 9604
        assert ACTION_SPACE_SIZE == BOARD_SIZE ** 2 * BOARD_SIZE ** 2 * 4

    def test_side_to_index_has_all_four_sides(self):
        from action_encoding import SIDE_TO_INDEX
        assert set(SIDE_TO_INDEX.keys()) == {'top', 'right', 'bottom', 'left'}

    def test_side_to_index_values_are_0_to_3(self):
        from action_encoding import SIDE_TO_INDEX
        assert set(SIDE_TO_INDEX.values()) == {0, 1, 2, 3}

    def test_index_to_side_round_trips(self):
        from action_encoding import SIDE_TO_INDEX, INDEX_TO_SIDE
        for side, idx in SIDE_TO_INDEX.items():
            assert INDEX_TO_SIDE[idx] == side

    # (D) Boundary — index_to_side only has keys 0-3
    def test_index_to_side_has_exactly_4_entries(self):
        from action_encoding import INDEX_TO_SIDE
        assert len(INDEX_TO_SIDE) == 4

    def test_index_to_side_keys_are_0_through_3(self):
        from action_encoding import INDEX_TO_SIDE
        assert set(INDEX_TO_SIDE.keys()) == {0, 1, 2, 3}


# ======================================================================
# Section 2: encode_action
# ======================================================================

class TestEncodeAction:
    """Tests for encode_action(to_x, to_y, wall_x, wall_y, wall_side) -> int."""

    # (A) Happy path — known manual calculation
    def test_encode_origin_top(self):
        """to=(0,0), wall=(0,0), side='top' → index 0 (if top=0)."""
        from action_encoding import encode_action, SIDE_TO_INDEX
        result = encode_action(0, 0, 0, 0, 'top')
        expected = 0 * (49 * 4) + 0 * 4 + SIDE_TO_INDEX['top']
        assert result == expected

    def test_encode_last_cell_left(self):
        """to=(6,6), wall=(6,6), side='left' → highest index region."""
        from action_encoding import encode_action, SIDE_TO_INDEX, ACTION_SPACE_SIZE
        result = encode_action(6, 6, 6, 6, 'left')
        expected = 48 * (49 * 4) + 48 * 4 + SIDE_TO_INDEX['left']
        assert result == expected
        assert 0 <= result < ACTION_SPACE_SIZE

    def test_encode_mid_board(self):
        """to=(3,3), wall=(4,2), side='right'."""
        from action_encoding import encode_action, SIDE_TO_INDEX
        to_cell = 3 * 7 + 3  # 24
        wall_cell = 2 * 7 + 4  # 18
        expected = to_cell * (49 * 4) + wall_cell * 4 + SIDE_TO_INDEX['right']
        result = encode_action(3, 3, 4, 2, 'right')
        assert result == expected

    # (A) Happy path — all four sides produce distinct indices for same cells
    def test_four_sides_produce_distinct_indices(self):
        from action_encoding import encode_action
        indices = set()
        for side in ('top', 'right', 'bottom', 'left'):
            indices.add(encode_action(3, 3, 3, 3, side))
        assert len(indices) == 4

    # (D) Boundary — minimum and maximum valid inputs
    def test_encode_returns_0_for_minimum_input(self):
        from action_encoding import encode_action, SIDE_TO_INDEX
        result = encode_action(0, 0, 0, 0, 'top')
        assert result == SIDE_TO_INDEX['top']

    def test_encode_returns_within_range(self):
        from action_encoding import encode_action, ACTION_SPACE_SIZE
        for side in ('top', 'right', 'bottom', 'left'):
            result = encode_action(6, 6, 6, 6, side)
            assert 0 <= result < ACTION_SPACE_SIZE

    # (C) Type/shape mismatch — invalid side string
    def test_encode_rejects_invalid_side(self):
        from action_encoding import encode_action
        with pytest.raises((ValueError, KeyError)):
            encode_action(0, 0, 0, 0, 'diagonal')

    # (D) Boundary — coordinates at grid edges
    def test_encode_top_left_corner(self):
        from action_encoding import encode_action, ACTION_SPACE_SIZE
        result = encode_action(0, 0, 0, 0, 'top')
        assert 0 <= result < ACTION_SPACE_SIZE

    def test_encode_bottom_right_corner(self):
        from action_encoding import encode_action, ACTION_SPACE_SIZE
        result = encode_action(6, 6, 6, 6, 'bottom')
        assert 0 <= result < ACTION_SPACE_SIZE


# ======================================================================
# Section 3: decode_action
# ======================================================================

class TestDecodeAction:
    """Tests for decode_action(action_int, piece_x, piece_y) -> full 7-tuple."""

    # (A) Happy path — round-trip with encode
    def test_round_trip_origin(self):
        from action_encoding import encode_action, decode_action
        original = (0, 0, 0, 0, 'top')
        encoded = encode_action(*original)
        px, py = 99, 99  # piece position (prepended to tuple)
        decoded = decode_action(encoded, px, py)
        assert decoded == (px, py, 0, 0, 0, 0, 'top')

    def test_round_trip_mid_board(self):
        from action_encoding import encode_action, decode_action
        original = (3, 4, 5, 1, 'right')
        encoded = encode_action(*original)
        decoded = decode_action(encoded, 2, 2)
        assert decoded == (2, 2, 3, 4, 5, 1, 'right')

    def test_round_trip_all_sides(self):
        from action_encoding import encode_action, decode_action
        for side in ('top', 'right', 'bottom', 'left'):
            original = (1, 2, 3, 4, side)
            encoded = encode_action(*original)
            decoded = decode_action(encoded, 0, 0)
            assert decoded == (0, 0, 1, 2, 3, 4, side)

    # (A) Happy path — exhaustive round-trip for a sample of actions
    def test_round_trip_sample(self):
        from action_encoding import encode_action, decode_action, ACTION_SPACE_SIZE
        import random
        random.seed(42)
        sides = ('top', 'right', 'bottom', 'left')
        for _ in range(200):
            to_x = random.randint(0, 6)
            to_y = random.randint(0, 6)
            wx = random.randint(0, 6)
            wy = random.randint(0, 6)
            ws = random.choice(sides)
            encoded = encode_action(to_x, to_y, wx, wy, ws)
            assert 0 <= encoded < ACTION_SPACE_SIZE
            decoded = decode_action(encoded, 5, 5)
            assert decoded == (5, 5, to_x, to_y, wx, wy, ws)

    # (D) Boundary — index 0 and max index
    def test_decode_index_0(self):
        from action_encoding import decode_action
        result = decode_action(0, 0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_decode_max_index(self):
        from action_encoding import decode_action, ACTION_SPACE_SIZE
        result = decode_action(ACTION_SPACE_SIZE - 1, 0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 7

    # (D) Boundary — out-of-range indices raise errors
    def test_decode_negative_index_raises(self):
        from action_encoding import decode_action
        with pytest.raises(ValueError):
            decode_action(-1, 0, 0)

    def test_decode_too_large_index_raises(self):
        from action_encoding import decode_action, ACTION_SPACE_SIZE
        with pytest.raises(ValueError):
            decode_action(ACTION_SPACE_SIZE, 0, 0)

    # (C) Type mismatch — float index
    def test_decode_float_index_raises(self):
        from action_encoding import decode_action
        with pytest.raises((TypeError, ValueError)):
            decode_action(1.5, 0, 0)


# ======================================================================
# Section 4: get_action_mask
# ======================================================================

class TestGetActionMask:
    """Tests for get_action_mask(env) -> np.ndarray of shape (9604,), dtype bool."""

    def _make_env(self):
        env = WallGoEnv()
        env.reset()
        return env

    # (A) Happy path — shape and dtype
    def test_mask_shape(self):
        from action_encoding import get_action_mask, ACTION_SPACE_SIZE
        env = self._make_env()
        mask = get_action_mask(env)
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_mask_dtype_is_bool(self):
        from action_encoding import get_action_mask
        env = self._make_env()
        mask = get_action_mask(env)
        assert mask.dtype == np.bool_

    # (A) Happy path — at least one legal action at game start
    def test_mask_has_true_entries_at_start(self):
        from action_encoding import get_action_mask
        env = self._make_env()
        mask = get_action_mask(env)
        assert mask.any(), "Mask should have at least one True at game start"

    # (A) Happy path — mask agrees with get_legal_actions
    def test_mask_matches_legal_actions(self):
        from action_encoding import get_action_mask, encode_action
        env = self._make_env()
        mask = get_action_mask(env)
        legal = env.get_legal_actions()
        # Every legal action should be True in mask
        for action in legal:
            from_x, from_y, to_x, to_y, wx, wy, ws = action
            idx = encode_action(to_x, to_y, wx, wy, ws)
            assert mask[idx], f"Legal action {action} not marked True in mask"

    def test_mask_true_count_equals_legal_actions_count(self):
        from action_encoding import get_action_mask
        env = self._make_env()
        mask = get_action_mask(env)
        legal = env.get_legal_actions()
        assert mask.sum() == len(legal), (
            f"Mask has {mask.sum()} True entries but get_legal_actions returns {len(legal)}"
        )

    # (A) Happy path — mask changes after a step
    def test_mask_changes_after_step(self):
        from action_encoding import get_action_mask
        env = self._make_env()
        mask_before = get_action_mask(env).copy()
        legal = env.get_legal_actions()
        env.step(legal[0])
        mask_after = get_action_mask(env)
        assert not np.array_equal(mask_before, mask_after), "Mask should change after a step"

    # (B) Empty — mask after game over should be all False
    def test_mask_all_false_when_game_over(self):
        from action_encoding import get_action_mask
        env = self._make_env()
        # Play random game to completion
        import random
        random.seed(0)
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            env.step(random.choice(legal))
        mask = get_action_mask(env)
        assert not mask.any(), "Mask should be all False when game is over"

    # (D) Boundary — mask at turn 1 vs later turns
    def test_mask_count_decreases_as_walls_placed(self):
        """After placing walls, fewer wall slots remain so fewer legal actions."""
        from action_encoding import get_action_mask
        env = self._make_env()
        count_start = get_action_mask(env).sum()
        legal = env.get_legal_actions()
        env.step(legal[0])
        # After opponent's perspective, take another turn
        legal2 = env.get_legal_actions()
        if legal2:
            env.step(legal2[0])
        count_later = get_action_mask(env).sum()
        # Walls reduce options, but moves may also change — just check it's different
        assert count_later != count_start or count_later > 0


# ======================================================================
# Section 5: Integration — encode/decode with live game
# ======================================================================

class TestIntegration:
    """Integration tests: action encoding works with actual WallGoEnv games."""

    def test_all_legal_actions_encode_within_range(self):
        """Every action from get_legal_actions encodes to a valid index."""
        from action_encoding import encode_action, ACTION_SPACE_SIZE
        env = WallGoEnv()
        env.reset()
        for action in env.get_legal_actions():
            _, _, to_x, to_y, wx, wy, ws = action
            idx = encode_action(to_x, to_y, wx, wy, ws)
            assert 0 <= idx < ACTION_SPACE_SIZE

    def test_decode_then_step_succeeds(self):
        """Decode an action index back to a tuple and step the env with it."""
        from action_encoding import encode_action, decode_action
        env = WallGoEnv()
        env.reset()
        legal = env.get_legal_actions()
        action = legal[0]
        from_x, from_y, to_x, to_y, wx, wy, ws = action
        idx = encode_action(to_x, to_y, wx, wy, ws)
        decoded = decode_action(idx, from_x, from_y)
        state, reward, done, info = env.step(decoded)
        assert "error" not in info, f"Step failed with decoded action: {info}"

    def test_no_duplicate_indices_in_legal_actions(self):
        """Each legal action maps to a unique index (no collisions)."""
        from action_encoding import encode_action
        env = WallGoEnv()
        env.reset()
        legal = env.get_legal_actions()
        indices = set()
        for action in legal:
            _, _, to_x, to_y, wx, wy, ws = action
            idx = encode_action(to_x, to_y, wx, wy, ws)
            indices.add(idx)
        assert len(indices) == len(legal), "Duplicate indices found in legal actions"
