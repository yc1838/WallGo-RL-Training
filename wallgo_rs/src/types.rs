use pyo3::prelude::*;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    Red,
    Blue,
    Green,
    Yellow,
}

impl Player {
    pub fn value(&self) -> &'static str {
        match self {
            Player::Red => "RED",
            Player::Blue => "BLUE",
            Player::Green => "GREEN",
            Player::Yellow => "YELLOW",
        }
    }

    pub fn from_value(s: &str) -> Option<Player> {
        match s {
            "RED" => Some(Player::Red),
            "BLUE" => Some(Player::Blue),
            "GREEN" => Some(Player::Green),
            "YELLOW" => Some(Player::Yellow),
            _ => None,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl<'py> IntoPyObject<'py> for Player {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.value().into_pyobject(py)?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Player {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Try extracting as a plain string first ("RED", "BLUE", etc.)
        if let Ok(s) = ob.extract::<String>() {
            return Player::from_value(&s).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid player: {}", s))
            });
        }
        // Fall back: try getting a .value attribute (for PyPlayer instances)
        if let Ok(val_attr) = ob.getattr("value") {
            if let Ok(s) = val_attr.extract::<String>() {
                return Player::from_value(&s).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid player: {}",
                        s
                    ))
                });
            }
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a string or Player instance",
        ))
    }
}

pub const SIDES: [&str; 4] = ["top", "right", "bottom", "left"];

pub const DIRECTIONS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

pub fn opposite(side: &str) -> &'static str {
    match side {
        "top" => "bottom",
        "bottom" => "top",
        "left" => "right",
        "right" => "left",
        _ => panic!("Invalid side: {}", side),
    }
}

pub fn side_delta(side: &str) -> (i32, i32) {
    match side {
        "top" => (0, -1),
        "bottom" => (0, 1),
        "left" => (-1, 0),
        "right" => (1, 0),
        _ => panic!("Invalid side: {}", side),
    }
}

pub fn side_index(side: &str) -> usize {
    match side {
        "top" => 0,
        "right" => 1,
        "bottom" => 2,
        "left" => 3,
        _ => panic!("Invalid side: {}", side),
    }
}

/// Map direction (dx,dy) to the wall side name for movement check
pub fn dir_to_wall_side(dx: i32, dy: i32) -> &'static str {
    match (dx, dy) {
        (0, -1) => "top",
        (0, 1) => "bottom",
        (-1, 0) => "left",
        (1, 0) => "right",
        _ => panic!("Invalid direction: ({}, {})", dx, dy),
    }
}
