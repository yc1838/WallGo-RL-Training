use std::collections::{HashMap, VecDeque};
use crate::board::{create_initial_board, CellData};
use crate::types::*;
use crate::union_find::UnionFind;

/// The main game environment.
#[derive(Clone)]
pub struct WallGoEnv {
    pub size: usize,
    pub allow_border_walls: bool,
    pub board: Vec<Vec<CellData>>,
    pub active_players: Vec<Player>,
    pub current_player_idx: usize,
    pub done: bool,
    pub turn_count: usize,
    pub piece_positions: HashMap<Player, Vec<(usize, usize)>>,
    /// Sorted canonical wall representations: (x, y, side_str)
    pub available_walls: Vec<(usize, usize, &'static str)>,
}

impl WallGoEnv {
    pub fn new(size: usize, allow_border_walls: bool) -> Self {
        WallGoEnv {
            size,
            allow_border_walls,
            board: create_initial_board(size),
            active_players: Vec::new(),
            current_player_idx: 0,
            done: false,
            turn_count: 0,
            piece_positions: HashMap::new(),
            available_walls: Vec::new(),
        }
    }

    pub fn current_player(&self) -> Player {
        self.active_players[self.current_player_idx]
    }

    pub fn reset(&mut self, players: Option<Vec<Player>>) {
        let players = players.unwrap_or_else(|| vec![Player::Red, Player::Blue]);
        self.board = create_initial_board(self.size);
        self.active_players = players.clone();
        self.current_player_idx = 0;
        self.done = false;
        self.turn_count = 0;

        // Build deduplicated canonical wall set
        let mut walls_set = std::collections::BTreeSet::new();
        for y in 0..self.size {
            for x in 0..self.size {
                for &side in &SIDES {
                    let canon = self.canonical_wall(x, y, side);
                    if !self.allow_border_walls && self.is_border_wall(canon.0, canon.1, canon.2) {
                        continue;
                    }
                    walls_set.insert(canon);
                }
            }
        }
        self.available_walls = walls_set.into_iter().collect();

        // Initialize piece positions
        self.piece_positions = players.iter().map(|&p| (p, Vec::new())).collect();

        // 2-player setup on 7x7
        if self.active_players.len() == 2 {
            self.place_piece(self.active_players[0], 3, 1);
            self.place_piece(self.active_players[1], 3, 5);
        }
    }

    fn place_piece(&mut self, player: Player, x: usize, y: usize) {
        self.board[y][x].occupant = Some(player);
        self.piece_positions.entry(player).or_default().push((x, y));
    }

    pub fn is_valid_coordinate(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.size && (y as usize) < self.size
    }

    // ------------------------------------------------------------------
    // Wall canonicalization
    // ------------------------------------------------------------------

    fn canonical_wall(&self, x: usize, y: usize, side: &str) -> (usize, usize, &'static str) {
        if side == "left" || side == "top" {
            let (dx, dy) = side_delta(side);
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if self.is_valid_coordinate(nx, ny) {
                return (nx as usize, ny as usize, opposite(side));
            }
        }
        // Convert side to static str
        let static_side: &'static str = match side {
            "top" => "top",
            "right" => "right",
            "bottom" => "bottom",
            "left" => "left",
            _ => panic!("Invalid side"),
        };
        (x, y, static_side)
    }

    fn is_border_wall(&self, x: usize, y: usize, side: &str) -> bool {
        let s = self.size;
        (side == "top" && y == 0)
            || (side == "bottom" && y == s - 1)
            || (side == "left" && x == 0)
            || (side == "right" && x == s - 1)
    }

    // ------------------------------------------------------------------
    // State encoding
    // ------------------------------------------------------------------

    pub fn encode_state(&self) -> Vec<Vec<Vec<i32>>> {
        let s = self.size;
        let cur = self.current_player();
        let mut state = vec![vec![vec![0i32; s]; s]; 6];

        // Pieces via tracked positions
        for &p in &self.active_players {
            let ch = if p == cur { 0 } else { 1 };
            if let Some(positions) = self.piece_positions.get(&p) {
                for &(x, y) in positions {
                    state[ch][y][x] = 1;
                }
            }
        }

        // Walls
        for y in 0..s {
            for x in 0..s {
                let cell = &self.board[y][x];
                if cell.has_wall("top") { state[2][y][x] = 1; }
                if cell.has_wall("right") { state[3][y][x] = 1; }
                if cell.has_wall("bottom") { state[4][y][x] = 1; }
                if cell.has_wall("left") { state[5][y][x] = 1; }
            }
        }
        state
    }

    // ------------------------------------------------------------------
    // Legal actions
    // ------------------------------------------------------------------

    pub fn get_player_pieces(&self, player: Player) -> Vec<(usize, usize)> {
        self.piece_positions.get(&player).cloned().unwrap_or_default()
    }

    pub fn get_valid_wall_placements(&self) -> Vec<(usize, usize, &'static str)> {
        self.available_walls.clone()
    }

    pub fn get_valid_moves(&self, start_x: usize, start_y: usize) -> Vec<(usize, usize)> {
        let size = self.size;
        let mut valid_moves = vec![(start_x, start_y)]; // staying is valid
        let mut visited = std::collections::HashSet::new();
        visited.insert((start_x, start_y));

        let wall_for_dir: [(i32, i32, &str); 4] = [
            (0, -1, "top"),
            (0, 1, "bottom"),
            (-1, 0, "left"),
            (1, 0, "right"),
        ];

        // BFS level by level, max 2 levels
        let mut level: Vec<(usize, usize)> = vec![(start_x, start_y)];
        for _depth in 0..2 {
            let mut next_level = Vec::new();
            for &(cx, cy) in &level {
                let cell = &self.board[cy][cx];
                for &(dx, dy, wall_side) in &wall_for_dir {
                    if cell.has_wall(wall_side) {
                        continue;
                    }
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < size && (ny as usize) < size {
                        let nxu = nx as usize;
                        let nyu = ny as usize;
                        if !visited.contains(&(nxu, nyu))
                            && self.board[nyu][nxu].occupant.is_none()
                        {
                            visited.insert((nxu, nyu));
                            valid_moves.push((nxu, nyu));
                            next_level.push((nxu, nyu));
                        }
                    }
                }
            }
            level = next_level;
        }
        valid_moves
    }

    pub fn get_legal_actions(&self) -> Vec<(usize, usize, usize, usize, usize, usize, &'static str)> {
        if self.done {
            return Vec::new();
        }
        let cur = self.current_player();
        let pieces = self.get_player_pieces(cur);
        let wall_slots = &self.available_walls;
        let mut actions = Vec::new();

        for &(px, py) in &pieces {
            let moves = self.get_valid_moves(px, py);
            for &(mx, my) in &moves {
                for &(wx, wy, ws) in wall_slots {
                    actions.push((px, py, mx, my, wx, wy, ws));
                }
            }
        }
        actions
    }

    // ------------------------------------------------------------------
    // Step
    // ------------------------------------------------------------------

    pub fn step(
        &mut self,
        from_x: usize, from_y: usize,
        to_x: usize, to_y: usize,
        wall_x: usize, wall_y: usize,
        wall_side: &str,
    ) -> (i32, bool, HashMap<String, String>) {
        let mut info = HashMap::new();

        if self.done {
            info.insert("error".to_string(), "Game already over".to_string());
            return (0, true, info);
        }

        let cur = self.current_player();

        // 1. Validate piece ownership
        if from_x >= self.size || from_y >= self.size
            || self.board[from_y][from_x].occupant != Some(cur)
        {
            info.insert("error".to_string(), "Invalid piece selected".to_string());
            return (-1, false, info);
        }

        // 2. Validate and execute move
        let valid_moves = self.get_valid_moves(from_x, from_y);
        if !valid_moves.contains(&(to_x, to_y)) {
            info.insert("error".to_string(), "Invalid move destination".to_string());
            return (-1, false, info);
        }

        self.board[from_y][from_x].occupant = None;
        self.board[to_y][to_x].occupant = Some(cur);
        // Update piece position tracking
        if let Some(pieces) = self.piece_positions.get_mut(&cur) {
            for pos in pieces.iter_mut() {
                if *pos == (from_x, from_y) {
                    *pos = (to_x, to_y);
                    break;
                }
            }
        }

        // 3. Validate and execute wall placement
        if !SIDES.contains(&wall_side) {
            self.undo_move(cur, from_x, from_y, to_x, to_y);
            info.insert("error".to_string(), format!("Invalid wall side: {}", wall_side));
            return (-1, false, info);
        }

        let (cwx, cwy, cws) = self.canonical_wall(wall_x, wall_y, wall_side);

        if !self.allow_border_walls && self.is_border_wall(cwx, cwy, cws) {
            self.undo_move(cur, from_x, from_y, to_x, to_y);
            info.insert("error".to_string(), "Border walls not allowed".to_string());
            return (-1, false, info);
        }

        if cwx >= self.size || cwy >= self.size || self.board[cwy][cwx].has_wall(cws) {
            self.undo_move(cur, from_x, from_y, to_x, to_y);
            info.insert("error".to_string(), "Invalid wall placement".to_string());
            return (-1, false, info);
        }

        self.do_place_wall(cwx, cwy, cws, cur);
        self.turn_count += 1;

        // 4. Check end condition + score
        let mut uf = self.build_union_find();
        self.done = self.check_end_with_uf(&mut uf);

        let mut reward = 0i32;
        if self.done {
            let scores = self.scores_with_uf(&mut uf);
            let my_score = *scores.get(&cur).unwrap_or(&0);
            let opp_score = scores
                .iter()
                .filter(|(&p, _)| p != cur)
                .map(|(_, &s)| s)
                .max()
                .unwrap_or(0);

            if my_score > opp_score {
                reward = 1;
                info.insert("result".to_string(), "win".to_string());
            } else if my_score < opp_score {
                reward = -1;
                info.insert("result".to_string(), "loss".to_string());
            } else {
                reward = 0;
                info.insert("result".to_string(), "tie".to_string());
            }
            // Add scores to info
            for (&p, &s) in &scores {
                info.insert(format!("score_{}", p.value()), s.to_string());
            }
        } else {
            self.current_player_idx =
                (self.current_player_idx + 1) % self.active_players.len();
        }

        (reward, self.done, info)
    }

    fn undo_move(&mut self, player: Player, from_x: usize, from_y: usize, to_x: usize, to_y: usize) {
        self.board[to_y][to_x].occupant = None;
        self.board[from_y][from_x].occupant = Some(player);
        if let Some(pieces) = self.piece_positions.get_mut(&player) {
            for pos in pieces.iter_mut() {
                if *pos == (to_x, to_y) {
                    *pos = (from_x, from_y);
                    break;
                }
            }
        }
    }

    pub fn do_place_wall(&mut self, x: usize, y: usize, side: &str, player: Player) {
        self.board[y][x].set_wall(side, Some(player));

        // Update neighbour
        let (dx, dy) = side_delta(side);
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if self.is_valid_coordinate(nx, ny) {
            let opp = opposite(side);
            self.board[ny as usize][nx as usize].set_wall(opp, Some(player));
        }

        // Remove from available walls (binary search since list is sorted)
        let canon = self.canonical_wall(x, y, side);
        if let Ok(idx) = self.available_walls.binary_search(&canon) {
            self.available_walls.remove(idx);
        }
    }

    // ------------------------------------------------------------------
    // Union-Find based checks
    // ------------------------------------------------------------------

    pub fn build_union_find(&self) -> UnionFind {
        let s = self.size;
        let mut uf = UnionFind::new(s * s);
        for y in 0..s {
            for x in 0..s {
                let cell = &self.board[y][x];
                if x + 1 < s && !cell.has_wall("right") {
                    uf.union(y * s + x, y * s + (x + 1));
                }
                if y + 1 < s && !cell.has_wall("bottom") {
                    uf.union(y * s + x, (y + 1) * s + x);
                }
            }
        }
        uf
    }

    fn check_end_with_uf(&self, uf: &mut UnionFind) -> bool {
        let s = self.size;
        let mut player_roots: HashMap<Player, Vec<usize>> = HashMap::new();
        for &p in &self.active_players {
            let mut roots = Vec::new();
            if let Some(positions) = self.piece_positions.get(&p) {
                for &(x, y) in positions {
                    let r = uf.find(y * s + x);
                    if !roots.contains(&r) {
                        roots.push(r);
                    }
                }
            }
            player_roots.insert(p, roots);
        }

        for i in 0..self.active_players.len() {
            for j in (i + 1)..self.active_players.len() {
                let p1 = self.active_players[i];
                let p2 = self.active_players[j];
                let roots1 = &player_roots[&p1];
                let roots2 = &player_roots[&p2];
                for &r1 in roots1 {
                    for &r2 in roots2 {
                        if uf.connected(r1, r2) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn scores_with_uf(&self, uf: &mut UnionFind) -> HashMap<Player, usize> {
        let s = self.size;
        let mut component_size: HashMap<usize, usize> = HashMap::new();
        for y in 0..s {
            for x in 0..s {
                let root = uf.find(y * s + x);
                *component_size.entry(root).or_insert(0) += 1;
            }
        }

        let mut scores = HashMap::new();
        for &p in &self.active_players {
            let mut seen_roots = Vec::new();
            let mut total = 0usize;
            if let Some(positions) = self.piece_positions.get(&p) {
                for &(x, y) in positions {
                    let root = uf.find(y * s + x);
                    if !seen_roots.contains(&root) {
                        seen_roots.push(root);
                        total += component_size.get(&root).copied().unwrap_or(0);
                    }
                }
            }
            scores.insert(p, total);
        }
        scores
    }

    pub fn check_game_end_condition(&self) -> bool {
        let mut uf = self.build_union_find();
        self.check_end_with_uf(&mut uf)
    }

    pub fn calculate_scores(&self) -> HashMap<Player, usize> {
        let mut uf = self.build_union_find();
        self.scores_with_uf(&mut uf)
    }

    pub fn get_reachable_area(&self, player: Player) -> Vec<(usize, usize)> {
        let mut queue = VecDeque::new();
        let mut visited = std::collections::HashSet::new();

        if let Some(positions) = self.piece_positions.get(&player) {
            for &(x, y) in positions {
                queue.push_back((x, y));
                visited.insert((x, y));
            }
        }

        let wall_for_dir: [(i32, i32, &str); 4] = [
            (0, -1, "top"),
            (0, 1, "bottom"),
            (-1, 0, "left"),
            (1, 0, "right"),
        ];

        while let Some((cx, cy)) = queue.pop_front() {
            let cell = &self.board[cy][cx];
            for &(dx, dy, wall_side) in &wall_for_dir {
                if cell.has_wall(wall_side) {
                    continue;
                }
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx >= 0 && ny >= 0 {
                    let nxu = nx as usize;
                    let nyu = ny as usize;
                    if nxu < self.size && nyu < self.size
                        && !visited.contains(&(nxu, nyu))
                        && self.board[nyu][nxu].occupant.is_none()
                    {
                        visited.insert((nxu, nyu));
                        queue.push_back((nxu, nyu));
                    }
                }
            }
        }
        visited.into_iter().collect()
    }
}
