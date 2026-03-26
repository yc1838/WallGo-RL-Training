mod board;
mod game;
mod types;
mod union_find;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use numpy::pyo3::Python as NpPython;
use numpy::{PyArray1, PyArray3};

use game::WallGoEnv as RustWallGoEnv;
use types::Player;

// ---------------------------------------------------------------------------
// PyWallGoEnv — #[pyclass] wrapper around the Rust WallGoEnv
// ---------------------------------------------------------------------------

#[pyclass(name = "WallGoEnv")]
struct PyWallGoEnv {
    inner: RustWallGoEnv,
}

#[pymethods]
impl PyWallGoEnv {
    #[new]
    #[pyo3(signature = (size=7, allow_border_walls=true))]
    fn new(size: usize, allow_border_walls: bool) -> Self {
        PyWallGoEnv {
            inner: RustWallGoEnv::new(size, allow_border_walls),
        }
    }

    // ---- Attributes exposed as properties ----

    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done
    }

    #[getter]
    fn turn_count(&self) -> usize {
        self.inner.turn_count
    }

    #[getter]
    fn current_player_idx(&self) -> usize {
        self.inner.current_player_idx
    }

    #[setter]
    fn set_current_player_idx(&mut self, idx: usize) {
        self.inner.current_player_idx = idx;
    }

    #[getter]
    fn allow_border_walls(&self) -> bool {
        self.inner.allow_border_walls
    }

    #[getter]
    fn active_players(&self) -> Vec<Player> {
        self.inner.active_players.clone()
    }

    #[getter]
    fn current_player(&self) -> Player {
        self.inner.current_player()
    }

    /// Expose _available_walls for action_encoding.py compatibility
    #[getter]
    fn _available_walls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let walls = &self.inner.available_walls;
        let result = PyList::empty(py);
        for &(x, y, s) in walls {
            let tup = PyTuple::new(py, &[
                x.into_pyobject(py)?.into_any(),
                y.into_pyobject(py)?.into_any(),
                s.into_pyobject(py)?.into_any(),
            ])?;
            result.append(tup)?;
        }
        Ok(result)
    }

    /// Expose _piece_positions for test compatibility
    #[getter]
    fn _piece_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (player, positions) in &self.inner.piece_positions {
            let pos_list = PyList::empty(py);
            for &(x, y) in positions {
                let tup = PyTuple::new(py, &[x, y])?;
                pos_list.append(tup)?;
            }
            result.set_item(*player, pos_list)?;
        }
        Ok(result)
    }

    // ---- Core methods ----

    #[pyo3(signature = (players=None))]
    fn reset<'py>(&mut self, py: Python<'py>, players: Option<Vec<Player>>) -> PyResult<Bound<'py, PyDict>> {
        self.inner.reset(players);
        self._get_state(py)
    }

    /// step(action) where action is a 7-tuple:
    /// (from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side)
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: &Bound<'py, PyTuple>,
    ) -> PyResult<(Bound<'py, PyDict>, i32, bool, Bound<'py, PyDict>)> {
        if action.len() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "action must be a 7-tuple (from_x, from_y, to_x, to_y, wall_x, wall_y, wall_side)",
            ));
        }
        let from_x: usize = action.get_item(0)?.extract()?;
        let from_y: usize = action.get_item(1)?.extract()?;
        let to_x: usize = action.get_item(2)?.extract()?;
        let to_y: usize = action.get_item(3)?.extract()?;
        let wall_x: usize = action.get_item(4)?.extract()?;
        let wall_y: usize = action.get_item(5)?.extract()?;
        let wall_side: String = action.get_item(6)?.extract()?;

        let (reward, done, info_map) =
            self.inner
                .step(from_x, from_y, to_x, to_y, wall_x, wall_y, &wall_side);

        let state = self._get_state(py)?;
        let info = PyDict::new(py);

        // Convert info_map to Python dict, handling "scores" specially
        // to match Python API: info["scores"] = {"RED": 25, "BLUE": 24}
        let mut score_entries: Vec<(String, String)> = Vec::new();
        for (k, v) in &info_map {
            if k.starts_with("score_") {
                let player_name = k.strip_prefix("score_").unwrap().to_string();
                score_entries.push((player_name, v.clone()));
            } else {
                info.set_item(k, v)?;
            }
        }
        if !score_entries.is_empty() {
            let scores_dict = PyDict::new(py);
            for (player_name, score_str) in &score_entries {
                let score: i64 = score_str.parse().unwrap_or(0);
                scores_dict.set_item(player_name, score)?;
            }
            info.set_item("scores", scores_dict)?;
        }

        Ok((state, reward, done, info))
    }

    fn clone(&self) -> PyWallGoEnv {
        PyWallGoEnv {
            inner: self.inner.clone(),
        }
    }

    fn encode_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let state = self.inner.encode_state();
        // Convert Vec<Vec<Vec<i32>>> to nested Python list
        let outer = PyList::empty(py);
        for channel in &state {
            let ch_list = PyList::empty(py);
            for row in channel {
                let row_list = PyList::new(py, row)?;
                ch_list.append(row_list)?;
            }
            outer.append(ch_list)?;
        }
        Ok(outer)
    }

    fn get_legal_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let actions = self.inner.get_legal_actions();
        let result = PyList::empty(py);
        for (fx, fy, tx, ty, wx, wy, ws) in actions {
            let tup = PyTuple::new(py, &[
                fx.into_pyobject(py)?.into_any(),
                fy.into_pyobject(py)?.into_any(),
                tx.into_pyobject(py)?.into_any(),
                ty.into_pyobject(py)?.into_any(),
                wx.into_pyobject(py)?.into_any(),
                wy.into_pyobject(py)?.into_any(),
                ws.into_pyobject(py)?.into_any(),
            ])?;
            result.append(tup)?;
        }
        Ok(result)
    }

    fn get_valid_moves<'py>(
        &self,
        py: Python<'py>,
        start_x: usize,
        start_y: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let moves = self.inner.get_valid_moves(start_x, start_y);
        let result = PyList::empty(py);
        for (x, y) in moves {
            let tup = PyTuple::new(py, &[x, y])?;
            result.append(tup)?;
        }
        Ok(result)
    }

    fn get_valid_wall_placements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let walls = self.inner.get_valid_wall_placements();
        let result = PyList::empty(py);
        for (x, y, s) in walls {
            let tup = PyTuple::new(py, &[
                x.into_pyobject(py)?.into_any(),
                y.into_pyobject(py)?.into_any(),
                s.into_pyobject(py)?.into_any(),
            ])?;
            result.append(tup)?;
        }
        Ok(result)
    }

    fn get_player_pieces<'py>(
        &self,
        py: Python<'py>,
        player: Player,
    ) -> PyResult<Bound<'py, PyList>> {
        let pieces = self.inner.get_player_pieces(player);
        let result = PyList::empty(py);
        for (x, y) in pieces {
            let tup = PyTuple::new(py, &[x, y])?;
            result.append(tup)?;
        }
        Ok(result)
    }

    fn get_reachable_area<'py>(
        &self,
        py: Python<'py>,
        player: Player,
    ) -> PyResult<Bound<'py, PySet>> {
        let area = self.inner.get_reachable_area(player);
        let result = PySet::empty(py)?;
        for (x, y) in area {
            let tup = PyTuple::new(py, &[x, y])?;
            result.add(tup)?;
        }
        Ok(result)
    }

    /// calculate_scores(active_players) — accepts players arg to match Python API
    #[pyo3(signature = (active_players=None))]
    fn calculate_scores<'py>(
        &self,
        py: Python<'py>,
        active_players: Option<Vec<Player>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = active_players; // ignored — uses self.inner.active_players
        let scores = self.inner.calculate_scores();
        let result = PyDict::new(py);
        for (player, score) in scores {
            result.set_item(player, score)?;
        }
        Ok(result)
    }

    fn check_game_end_condition(&self) -> bool {
        self.inner.check_game_end_condition()
    }

    fn place_wall(&mut self, x: usize, y: usize, side: &str, player: Player) {
        self.inner.do_place_wall(x, y, side, player);
    }

    fn is_valid_coordinate(&self, x: i32, y: i32) -> bool {
        self.inner.is_valid_coordinate(x, y)
    }

    /// Fast action mask: computes entirely in Rust, returns numpy bool array of shape (9604,).
    fn get_action_mask_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        const BOARD_SIZE: usize = 7;
        const NUM_CELLS: usize = BOARD_SIZE * BOARD_SIZE;
        const NUM_SIDES: usize = 4;
        const ACTION_SPACE_SIZE: usize = NUM_CELLS * NUM_CELLS * NUM_SIDES;

        let mut mask = vec![false; ACTION_SPACE_SIZE];
        if !self.inner.done {
            let cur = self.inner.current_player();
            let pieces = self.inner.get_player_pieces(cur);
            let wall_slots = &self.inner.available_walls;

            if !wall_slots.is_empty() {
                // Pre-compute wall indices
                let wall_indices: Vec<usize> = wall_slots
                    .iter()
                    .map(|&(wx, wy, ws)| {
                        let side_idx = match ws {
                            "top" => 0,
                            "right" => 1,
                            "bottom" => 2,
                            "left" => 3,
                            _ => 0,
                        };
                        (wy * BOARD_SIZE + wx) * NUM_SIDES + side_idx
                    })
                    .collect();

                for &(px, py) in &pieces {
                    let valid_moves = self.inner.get_valid_moves(px, py);
                    for &(mx, my) in &valid_moves {
                        let to_offset = (my * BOARD_SIZE + mx) * (NUM_CELLS * NUM_SIDES);
                        for &wi in &wall_indices {
                            mask[to_offset + wi] = true;
                        }
                    }
                }
            }
        }
        PyArray1::from_vec(py, mask)
    }

    /// Fast state encoding: returns numpy float32 array of shape (6, 7, 7).
    fn encode_state_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let state = self.inner.encode_state();
        // Convert i32 to f32
        let float_state: Vec<Vec<Vec<f32>>> = state
            .iter()
            .map(|ch| ch.iter().map(|row| row.iter().map(|&v| v as f32).collect()).collect())
            .collect();
        // from_vec3 takes &[Vec<Vec<T>>]
        let refs: Vec<Vec<f32>> = float_state.into_iter().flatten().collect();
        // Use from_vec to create 1D then reshape
        let flat: Vec<f32> = state.iter().flat_map(|ch| ch.iter().flat_map(|row| row.iter().map(|&v| v as f32))).collect();
        let arr1d = PyArray1::<f32>::from_vec(py, flat);
        // Reshape via Python
        let reshaped = arr1d.call_method1("reshape", ((6, self.inner.size, self.inner.size),))?;
        reshaped.downcast::<PyArray3<f32>>().map(|a| a.clone()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // ---- Private-ish methods used by wallgo_gym.py's _shaping_reward ----
    // Exposed so the gym wrapper works without modification.

    fn _build_union_find(&self) -> PyUnionFind {
        PyUnionFind {
            inner: self.inner.build_union_find(),
        }
    }

    fn _scores_with_uf<'py>(
        &self,
        py: Python<'py>,
        uf: &mut PyUnionFind,
        _players: Vec<Player>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let scores = self.inner.scores_with_uf(&mut uf.inner);
        let result = PyDict::new(py);
        for (player, score) in scores {
            result.set_item(player, score)?;
        }
        Ok(result)
    }
}

// Helper method (not exposed to Python)
impl PyWallGoEnv {
    fn _get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("current_player", self.inner.current_player())?;
        d.set_item("numeric", self.encode_state(py)?)?;
        d.set_item("turn", self.inner.turn_count)?;
        // Note: we don't include "board" — it's not used by RL code
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// PyUnionFind — thin wrapper so _scores_with_uf can accept it
// ---------------------------------------------------------------------------

#[pyclass(name = "UnionFind")]
struct PyUnionFind {
    inner: union_find::UnionFind,
}

#[pymethods]
impl PyUnionFind {
    fn find(&mut self, x: usize) -> usize {
        self.inner.find(x)
    }

    fn union_(&mut self, a: usize, b: usize) {
        self.inner.union(a, b);
    }

    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.inner.connected(a, b)
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn wallgo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWallGoEnv>()?;
    m.add_class::<PyUnionFind>()?;

    // Export SIDES as a tuple of strings (matches Python's SIDES constant)
    let py = m.py();
    let sides = PyTuple::new(py, &["top", "right", "bottom", "left"])?;
    m.add("SIDES", sides)?;

    // Export Player as a simple namespace with string constants
    // Python code uses Player.RED, Player.BLUE etc. which are enum members
    // whose .value is "RED", "BLUE" etc. For wallgo_rs, we use the PyPlayer wrapper.
    m.add_class::<PyPlayer>()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PyPlayer — mimics Python's Player enum for `from wallgo_rs import Player`
// ---------------------------------------------------------------------------

/// Player enum exposed to Python.
/// Instances compare equal to their string value ("RED", "BLUE", etc.)
/// and have a .value property returning the string.
#[pyclass(name = "Player")]
#[derive(Clone)]
struct PyPlayer {
    inner: Player,
}

#[pymethods]
impl PyPlayer {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        Player::from_value(value)
            .map(|p| PyPlayer { inner: p })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid player: {}", value))
            })
    }

    #[getter]
    fn value(&self) -> &str {
        self.inner.value()
    }

    fn __repr__(&self) -> String {
        format!("<Player.{}: '{}'>", self.inner.value(), self.inner.value())
    }

    fn __str__(&self) -> &str {
        self.inner.value()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Support comparison with both PyPlayer instances and strings
        if let Ok(other_player) = other.extract::<PyPlayer>() {
            Ok(self.inner == other_player.inner)
        } else if let Ok(s) = other.extract::<String>() {
            Ok(self.inner.value() == s)
        } else {
            Ok(false)
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.value().hash(&mut hasher);
        hasher.finish()
    }

    // Class-level constants: Player.RED, Player.BLUE, etc.
    #[classattr]
    fn RED() -> PyPlayer {
        PyPlayer { inner: Player::Red }
    }

    #[classattr]
    fn BLUE() -> PyPlayer {
        PyPlayer { inner: Player::Blue }
    }

    #[classattr]
    fn GREEN() -> PyPlayer {
        PyPlayer { inner: Player::Green }
    }

    #[classattr]
    fn YELLOW() -> PyPlayer {
        PyPlayer { inner: Player::Yellow }
    }
}

// Allow PyPlayer to be used as a function argument that converts to/from Player
impl From<PyPlayer> for Player {
    fn from(p: PyPlayer) -> Player {
        p.inner
    }
}
