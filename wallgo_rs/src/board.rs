use crate::types::Player;

/// Per-cell state: occupant + 4 wall flags.
/// Walls store Option<Player> to indicate who placed them (or None).
#[derive(Clone)]
pub struct CellData {
    pub x: usize,
    pub y: usize,
    pub occupant: Option<Player>,
    /// [top, right, bottom, left] — None means no wall
    pub walls: [Option<Player>; 4],
}

impl CellData {
    pub fn new(x: usize, y: usize) -> Self {
        CellData {
            x,
            y,
            occupant: None,
            walls: [None; 4],
        }
    }

    pub fn get_wall(&self, side: &str) -> Option<Player> {
        self.walls[side_idx(side)]
    }

    pub fn set_wall(&mut self, side: &str, player: Option<Player>) {
        self.walls[side_idx(side)] = player;
    }

    pub fn has_wall(&self, side: &str) -> bool {
        self.walls[side_idx(side)].is_some()
    }
}

fn side_idx(side: &str) -> usize {
    match side {
        "top" => 0,
        "right" => 1,
        "bottom" => 2,
        "left" => 3,
        _ => panic!("Invalid side: {}", side),
    }
}

pub fn create_initial_board(size: usize) -> Vec<Vec<CellData>> {
    (0..size)
        .map(|y| (0..size).map(|x| CellData::new(x, y)).collect())
        .collect()
}
