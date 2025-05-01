///TODO!
/// - Pause
/// - Type
/// - Menu?
/// - Little timeout after clearing a line so that we don't accidentally push down on the next piece
mod point;

use Color::*;
use bitflags::bitflags;
use color_eyre::Result;
use rand::random;
use ratatui::buffer::{Buffer, Cell};
use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::crossterm::{ExecutableCommand, event};
use ratatui::layout::Alignment::Center;
use ratatui::layout::{Constraint, Flex, Layout, Margin, Rect};
use ratatui::prelude::{Stylize, Widget};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use std::ops::{Index, IndexMut, Sub};
use std::time::Duration;

type Line = u16;
const WALL_LINE: Line = 0b_1110_0000_0000_0111;
const FULL_LINE: Line = 0b_1111_1111_1111_1111;

const WIDTH: usize = 10;
const HEIGHT: usize = 20;
type Grid = [Line; HEIGHT + 5];
const GRID: Grid = [
    WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE,
    WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE,
    WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, WALL_LINE, FULL_LINE, FULL_LINE, FULL_LINE,
    FULL_LINE,
];

#[repr(u8)]
#[derive(Default, Copy, Clone)]
enum Rotation2 {
    #[default]
    Sideways,
    Upright,
}

impl Rotation2 {
    pub fn prev(&self) -> Rotation2 {
        match self {
            Self::Sideways => Self::Upright,
            Self::Upright => Self::Sideways,
        }
    }
    pub fn next(&self) -> Rotation2 {
        match self {
            Self::Sideways => Self::Upright,
            Self::Upright => Self::Sideways,
        }
    }
}

#[repr(u8)]
#[derive(Default, Copy, Clone)]
enum Rotation4 {
    #[default]
    Down,
    Right,
    Up,
    Left,
}

impl Rotation4 {
    pub fn prev(&self) -> Self {
        match self {
            Self::Down => Self::Left,
            Self::Right => Self::Down,
            Self::Up => Self::Right,
            Self::Left => Self::Up,
        }
    }
    pub fn next(&self) -> Rotation4 {
        match self {
            Self::Down => Self::Right,
            Self::Right => Self::Up,
            Self::Up => Self::Left,
            Self::Left => Self::Down,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Default)]
enum Tetromino {
    #[default]
    O = 0,
    I(Rotation2),
    S(Rotation2),
    Z(Rotation2),
    L(Rotation4),
    J(Rotation4),
    T(Rotation4),
}

const PALETTES: [([Color; 3], &'static str); 8] = [
    ([Red, LightRed, LightYellow], BLOCK_CHAR),
    ([LightBlue, LightRed, White], BLOCK_CHAR),
    ([Magenta, White, LightBlue], BLOCK_CHAR),
    ([LightYellow, Yellow, White], BLOCK_CHAR),
    ([White, Green, LightGreen], BLOCK_CHAR),
    ([LightCyan, Cyan, LightYellow], BLOCK_CHAR),
    ([White, LightBlue, Blue], BLOCK_CHAR),
    ([Magenta, LightMagenta, White], BLOCK_CHAR),
];

const BLOCKS: TetrominoBlocks = TetrominoMap {
    content: [[1, 2], [1, 2], [2, 1], [0, 1], [2, 1], [0, 1], [1, 2]],
};
impl From<u8> for Tetromino {
    fn from(value: u8) -> Self {
        match value {
            0 => Tetromino::O,
            1 => Tetromino::I(Rotation2::Sideways),
            2 => Tetromino::S(Rotation2::Sideways),
            3 => Tetromino::Z(Rotation2::Sideways),
            4 => Tetromino::L(Rotation4::Right),
            5 => Tetromino::J(Rotation4::Right),
            6 => Tetromino::T(Rotation4::Down),
            _ => unreachable!(),
        }
    }
}
impl From<Tetromino> for usize {
    fn from(value: Tetromino) -> Self {
        match value {
            Tetromino::O => 0,
            Tetromino::I(_) => 1,
            Tetromino::S(_) => 2,
            Tetromino::Z(_) => 3,
            Tetromino::L(_) => 4,
            Tetromino::J(_) => 5,
            Tetromino::T(_) => 6,
        }
    }
}

type Shape = [Line; 4];
impl Tetromino {
    pub fn rotate_left(&mut self) {
        match self {
            Tetromino::O => {}
            Tetromino::I(r) => *r = r.prev(),
            Tetromino::S(r) => *r = r.prev(),
            Tetromino::Z(r) => *r = r.prev(),
            Tetromino::L(r) => *r = r.prev(),
            Tetromino::J(r) => *r = r.prev(),
            Tetromino::T(r) => *r = r.prev(),
        }
    }
    pub fn rotate_right(&mut self) {
        match self {
            Tetromino::O => {}
            Tetromino::I(r) => *r = r.next(),
            Tetromino::S(r) => *r = r.next(),
            Tetromino::Z(r) => *r = r.next(),
            Tetromino::L(r) => *r = r.next(),
            Tetromino::J(r) => *r = r.next(),
            Tetromino::T(r) => *r = r.next(),
        }
    }

    const fn get_shape(&self) -> Shape {
        match self {
            Tetromino::O => [
                0b_0000, //
                0b_0000, //
                0b_0110, // ##
                0b_0110, // ##
            ],

            Tetromino::I(r) => match r {
                Rotation2::Sideways => [
                    0b_0000, //
                    0b_0000, //
                    0b_1111, //####
                    0b_0000, //
                ],
                Rotation2::Upright => [
                    0b_0100, // #
                    0b_0100, // #
                    0b_0100, // #
                    0b_0100, // #
                ],
            },
            Tetromino::S(r) => match r {
                Rotation2::Sideways => [
                    0b_0000, //
                    0b_0000, //
                    0b_0110, // ##
                    0b_1100, //##
                ],
                Rotation2::Upright => [
                    0b_0000, //
                    0b_1000, //#
                    0b_1100, //##
                    0b_0100, // #
                ],
            },
            Tetromino::Z(r) => match r {
                Rotation2::Sideways => [
                    0b_0000, //
                    0b_0000, //
                    0b_1100, //##
                    0b_0110, // ##
                ],
                Rotation2::Upright => [
                    0b_0000, //
                    0b_0100, // #
                    0b_1100, //##
                    0b_1000, //#
                ],
            },
            Tetromino::L(r) => match r {
                Rotation4::Right => [
                    0b_0000, //
                    0b_0000, //
                    0b_1110, //###
                    0b_1000, //#
                ],
                Rotation4::Up => [
                    0b_0000, //
                    0b_0100, // #
                    0b_0100, // #
                    0b_0110, // ##
                ],
                Rotation4::Left => [
                    0b_0000, //
                    0b_0010, //  #
                    0b_1110, //###
                    0b_0000, //
                ],
                Rotation4::Down => [
                    0b_0000, //
                    0b_1100, //##
                    0b_0100, // #
                    0b_0100, // #
                ],
            },
            Tetromino::J(r) => match r {
                Rotation4::Right => [
                    0b_0000, //
                    0b_0000, //
                    0b_1110, //###
                    0b_0010, //  #
                ],
                Rotation4::Up => [
                    0b_0000, //
                    0b_0110, // ##
                    0b_0100, // #
                    0b_0100, // #
                ],
                Rotation4::Left => [
                    0b_0000, //
                    0b_1000, //#
                    0b_1110, //###
                    0b_0000, //
                ],
                Rotation4::Down => [
                    0b_0000, //
                    0b_0100, // #
                    0b_0100, // #
                    0b_1100, //##
                ],
            },
            Tetromino::T(r) => match r {
                Rotation4::Down => [
                    0b_0000, //
                    0b_0000, //
                    0b_1110, //###
                    0b_0100, // #
                ],
                Rotation4::Right => [
                    0b_0000, //
                    0b_0100, // #
                    0b_0110, // ##
                    0b_0100, // #
                ],
                Rotation4::Up => [
                    0b_0000, //
                    0b_0100, // #
                    0b_1110, //###
                    0b_0000, //
                ],
                Rotation4::Left => [
                    0b_0000, //
                    0b_0100, // #
                    0b_1100, //##
                    0b_0100, // #
                ],
            },
        }
    }
}

#[derive(Default)]
struct TetrominoMap<T> {
    content: [T; 7],
}

impl<T> TetrominoMap<T> {}
impl<T> Index<Tetromino> for TetrominoMap<T> {
    type Output = T;

    fn index(&self, tetromino: Tetromino) -> &Self::Output {
        let i: usize = tetromino.into();
        &self.content[i]
    }
}
impl<T> IndexMut<Tetromino> for TetrominoMap<T> {
    fn index_mut(&mut self, tetromino: Tetromino) -> &mut Self::Output {
        let i: usize = tetromino.into();
        &mut self.content[i]
    }
}

type TetrominoBoard = TetrominoMap<Grid>;
type TetrominoCount = TetrominoMap<usize>;
type TetrominoBlocks = TetrominoMap<[usize; 2]>;

bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq, Default)]
    pub struct Input: u8 {
        const Up    = 1<<0;
        const Down  = 1<<1;
        const Left  = 1<<2;
        const Right = 1<<3;
        const A     = 1<<4;
        const B     = 1<<5;
        const SELECT= 1<<6;
        const START = 1<<7;
    }
}

//(previous, current)
#[derive(Default)]
struct InputBuffer(Input, Input);
impl InputBuffer {
    fn is_pressed(&self, input: Input) -> bool {
        self.1.contains(input)
    }

    fn was_pressed(&self, input: Input) -> bool {
        self.0.contains(input)
    }

    fn is_just_pressed(&self, input: Input) -> bool {
        !self.was_pressed(input) && self.is_pressed(input)
    }

    fn update(&mut self, input: Input) {
        self.0 = self.1;
        self.1 = input;
    }
}

struct GameState {
    input: InputBuffer,
    collision_board: Grid,
    tetromino_board: TetrominoBoard,
    tetromino_count: TetrominoCount,
    pos: (u8, u8),

    current_tetromino: Tetromino,
    next_tetromino: Tetromino,

    score: u16,
    lines: u16,
    level: u16,
    continuous_fall_count: u16,

    fall_speed: u8,

    frames_without_falling: u8,
    top_score: u16,
}

impl GameState {
    fn lose(&mut self) {
        let top = self.top_score;
        *self = Self::default();
        self.top_score = top;
    }
}

fn get_axis<T: From<bool> + Sub<Output = T>>(neg: bool, pos: bool) -> T {
    let n: T = neg.into();
    let p: T = pos.into();
    p - n
}

impl Default for GameState {
    fn default() -> Self {
        let mut ret = Self {
            input: Default::default(),
            collision_board: GRID,
            tetromino_board: Default::default(),
            tetromino_count: Default::default(),
            pos: (5, 0),
            current_tetromino: Self::random_tetromino(),
            next_tetromino: Self::random_tetromino(),
            score: 0,
            lines: 0,
            continuous_fall_count: 0,
            level: 0,
            frames_without_falling: 0,
            top_score: 0,
            fall_speed: 0,
        };
        ret.tetromino_count[ret.current_tetromino] += 1;
        ret.update_level();
        ret
    }
}

fn get_fall_speed(level: u16) -> u8 {
    match level {
        0 => 48,
        1 => 43,
        2 => 38,
        3 => 33,
        4 => 28,
        5 => 23,
        6 => 18,
        7 => 13,
        8 => 8,
        9 => 6,
        (10..=12) => 5,
        (13..=15) => 4,
        (16..=18) => 3,
        (19..=28) => 2,
        (29..) => 1,
    }
}

const POINTS_PER_LINE: [u16; 5] = [0, 40, 100, 300, 1200];
impl GameState {
    fn update(&mut self, input: Input) {
        self.input.update(input);

        let (mut x, mut y) = self.pos;
        let mut should_fall = false;

        'side_movement: {
            let dx = get_axis(
                self.input.is_just_pressed(Input::Left),
                self.input.is_just_pressed(Input::Right),
            );

            if dx == 0 {
                should_fall |= self.input.is_just_pressed(Input::Down);
                break 'side_movement;
            }

            let x1 = x.saturating_add_signed(dx);
            if !self.check_collision(x1, y, &self.current_tetromino) {
                self.pos.0 = x1;
                x = x1;
            }
        }

        'rotation: {
            let rot = get_axis(
                self.input.is_just_pressed(Input::A),
                self.input.is_just_pressed(Input::B),
            );
            if rot == 0 {
                break 'rotation;
            }
            let mut tetromino = self.current_tetromino;
            match rot {
                -1 => tetromino.rotate_left(),
                1 => tetromino.rotate_right(),
                _ => unreachable!(),
            }
            if !self.check_collision(x, y, &tetromino) {
                self.current_tetromino = tetromino;
            }
        }

        'fall: {
            self.frames_without_falling += 1;
            if should_fall {
                self.continuous_fall_count += 1;
            }
            if self.frames_without_falling >= self.fall_speed {
                should_fall = true;
                self.continuous_fall_count = 0;
            }
            if !should_fall {
                break 'fall;
            }
            self.frames_without_falling = 0;
            if self.check_collision(x, y + 1, &self.current_tetromino) {
                self.commit();
                return;
            } else {
                y = y + 1;
            }
        }

        self.pos = (x, y);
    }

    fn check_collision(&self, x: u8, y: u8, tetromino: &Tetromino) -> bool {
        let yu = y as usize;
        let xu = x as u16;

        let lines_to_check: &[Line] = &self.collision_board[yu..(yu + 4)];
        let tetra_board: [Line; 4] = tetromino.get_shape().map(|it| it << xu);
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .any(|(a, b)| a & b != 0)
    }

    fn increment_score(&mut self, by: u16) {
        self.score += by;
        self.top_score = self.top_score.max(self.score);
    }
    fn commit(&mut self) {
        let (x, y) = self.pos;

        if y == 0 {
            self.lose();
            return;
        }

        self.increment_score(self.continuous_fall_count);

        self.continuous_fall_count = 0;

        let yu = y as usize;
        let xu = x as u16;
        let tetromino = self.current_tetromino;

        let lines_to_check: &mut [Line] = &mut self.collision_board[yu..HEIGHT + 1];

        let tetra_board: [Line; 4] = tetromino.get_shape().map(|it| it << xu);
        let mut clears = [false, false, false, false];
        let mut lines_cleared_count = 0;
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .enumerate()
            .for_each(|(i, (a, b))| {
                *a |= b;
                let clear = *a == FULL_LINE;
                clears[i] = clear;
                if clear {
                    lines_cleared_count += 1
                }
            });

        let tetromino_grid = &mut self.tetromino_board[tetromino];
        let lines_to_check: &mut [Line] = &mut tetromino_grid[yu..HEIGHT + 1];
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .for_each(|(a, b)| {
                *a |= b;
            });

        Self::use_next_tetromino(
            &mut self.current_tetromino,
            &mut self.next_tetromino,
            &mut self.tetromino_count,
        );
        self.frames_without_falling = 0;
        self.pos = (WIDTH as u8 / 2, 0);

        if lines_cleared_count > 0 {
            (yu..HEIGHT + 1).zip(clears).for_each(|(y, clear)| {
                if clear {
                    self.collision_board[0..=y].rotate_right(1);
                    self.collision_board[0] = WALL_LINE;
                    for grid in &mut self.tetromino_board.content {
                        grid[0..=y].rotate_right(1);
                        grid[0] = 0;
                    }
                }
            });
            let n = self.level;
            self.increment_score(POINTS_PER_LINE[lines_cleared_count] * (n + 1));
            self.lines += lines_cleared_count as u16;
            self.update_level();
        }
    }

    fn update_level(&mut self) {
        self.level = self.level.max(self.lines / 10);
        self.fall_speed = get_fall_speed(self.level);
    }

    fn random_tetromino() -> Tetromino {
        let next: u8 = random();
        Tetromino::from(next % 7)
    }

    fn use_next_tetromino(
        current: &mut Tetromino,
        next: &mut Tetromino,
        tetromino_count: &mut TetrominoCount,
    ) {
        *current = *next;
        tetromino_count[*current] += 1;
        *next = Self::random_tetromino();
    }
}

#[derive(Default)]
struct RatatuiApp {
    tetris: GameState,
    input: Input,
    running: bool,
}

struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Self {
        std::io::stdout()
            .execute(event::EnableMouseCapture)
            .unwrap();
        Self {}
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = std::io::stdout()
            .execute(event::DisableMouseCapture)
            .unwrap();
        ratatui::restore();
    }
}

struct TetrominoWidget<'a> {
    tetromino: &'a Tetromino,
    palette: usize,
}

struct CounterWidget {
    count: u16,
    pal: (Color, Color),
}
impl Widget for CounterWidget {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        buf.set_string(
            area.x,
            area.y,
            format!("{:0>3}", self.count),
            Style::from(self.pal),
        );
    }
}

impl Widget for TetrominoWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let shape = self.tetromino.get_shape();
        let (ox, oy) = (area.x, area.y);
        for (j, line) in shape.iter().enumerate() {
            for i in 0..4 {
                let c = get_bit(*line, i);
                if c {
                    let [ibg, ifg] = BLOCKS[*self.tetromino];
                    let (pal, c) = PALETTES[self.palette % PALETTES.len()];
                    let bg = pal[ibg];
                    let fg = pal[ifg];
                    let mut cell = Cell::new(c);
                    cell.bg = bg;
                    cell.fg = fg;

                    buf[(ox + i, oy + j as u16 - 1)] = cell;
                }
            }
        }
    }
}

fn get_bit(source: u16, bit: u16) -> bool {
    source & (1 << bit) != 0
}

//const BLOCK_CHAR: &'static str = "□";
const BLOCK_CHAR: &'static str = "#";

struct GameWidget<'a> {
    tetromino_board: &'a TetrominoBoard,
    pos: (u16, u16),
    tetromino: Tetromino,
    palette: usize,
}
impl Widget for GameWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let (pal, c) = PALETTES[self.palette % PALETTES.len()];

        let (ox, oy) = (area.x, area.y);
        for (index, board) in self.tetromino_board.content.iter().enumerate() {
            for (j, line) in board.iter().enumerate() {
                for i in 0..16 {
                    let bit = get_bit(*line, i);
                    if bit {
                        let [ibg, ifg] = BLOCKS.content[index];
                        let bg = pal[ibg];
                        let fg = pal[ifg];
                        let mut cell = Cell::new(c);
                        cell.bg = bg;
                        cell.fg = fg;

                        buf[(ox + i - 3, oy + j as u16 - 1)] = cell;
                    }
                }
            }
        }

        /*for (j, line) in self.grid.iter().enumerate() {
            for i in 0..16 {
                let c = get_bit(*line, i);
                if c {
                    let mut cell = Cell::new("#");
                    cell.fg = Color::Yellow;
                    buf[(ox + i-3, oy + j as u16-1)] = cell;
                }
            }
        }*/

        for (j, line) in self.tetromino.get_shape().iter().enumerate() {
            for i in 0..4 {
                let bit = get_bit(*line, i);
                if bit {
                    let [ibg, ifg] = BLOCKS[self.tetromino];
                    let bg = pal[ibg];
                    let fg = pal[ifg];

                    let mut cell = Cell::new(c);
                    cell.bg = bg;
                    cell.fg = fg;
                    buf[(self.pos.0 + ox + i - 3, self.pos.1 + oy + j as u16 - 1)] = cell;
                }
            }
        }
    }
}

struct InputWidget<'a> {
    input: &'a InputBuffer,
}

impl Widget for InputWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        // ^
        //< >   A
        // v  B
        //
        // SEL STA
        let up = (area.left() + 1, area.top() + 0);

        let left = (area.left() + 0, area.top() + 1);
        let right = (area.left() + 2, area.top() + 1);
        let a = (area.left() + 6, area.top() + 1);

        let down = (area.left() + 1, area.top() + 2);
        let b = (area.left() + 4, area.top() + 2);

        let select = (area.left() + 1, area.top() + 4);
        let start = (area.left() + 5, area.top() + 4);

        let get_color = |button: Input| -> (Color, Color) {
            match (
                self.input.was_pressed(button),
                self.input.is_pressed(button),
            ) {
                (false, false) => (Reset, Reset),
                (false, true) => (Yellow, Red),
                (true, true) => (Reset, Red),
                (true, false) => (Red, Reset),
            }
        };

        let get_cell = |s: &'static str, button: Input| -> Cell {
            let (fg, bg) = get_color(button);
            let mut cell = Cell::new(s);
            cell.set_fg(fg).set_bg(bg);
            cell
        };

        buf[up] = get_cell("^", Input::Up);
        buf[left] = get_cell("<", Input::Left);
        buf[right] = get_cell(">", Input::Right);
        buf[down] = get_cell("v", Input::Down);

        buf[a] = get_cell("A", Input::A);
        buf[b] = get_cell("B", Input::B);

        buf[select] = get_cell("SEL", Input::SELECT);
        buf[start] = get_cell("STA", Input::START);
    }
}

fn get_palette(i: u16) -> [Color; 3] {
    PALETTES[i as usize % PALETTES.len()].0
}

impl RatatuiApp {
    pub fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        self.running = true;
        while self.running {
            self.handle_input()?;
            self.update();
            self.draw(&mut terminal)?;
        }
        Ok(())
    }

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        let area = area.clamp(Rect::new(0, 0, 32, 30));

        let w = area.width;
        let h = area.height;
        if w < 32 || h < 30 {
            Self::draw_too_small(frame, area, w, h);
        } else {
            self.draw_tetris(frame, area);
        }
    }

    fn draw_too_small(frame: &mut Frame, area: Rect, w: u16, h: u16) {
        let too_small_block = Block::new()
            .borders(Borders::all())
            .border_type(BorderType::Rounded)
            .border_style(Style::new())
            .title_top("Too small!")
            .title_style(Style::from(Red))
            .title_alignment(Center);

        let inner = Self::center(
            too_small_block.inner(area),
            Constraint::Min(0),
            Constraint::Max(2),
        );
        frame.render_widget(too_small_block, area);

        let too_small_text = Paragraph::new(format!("need: 32x30, have: {w}x{h}"))
            .centered()
            .wrap(Wrap { trim: false })
            .fg(Red);
        frame.render_widget(too_small_text, inner);
    }

    fn draw_tetris(&self, frame: &mut Frame, area: Rect) {
        let [_, col1, col2] = get_palette(self.tetris.level);
        let style = Style::from((col1, Reset));

        let tetris = Block::new()
            .borders(Borders::all())
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(col1))
            .title_style(Style::from(col2))
            .title_alignment(Center);
        frame.render_widget(tetris.clone().title_top("TETRIS"), area);

        let inner = tetris.inner(area);

        let h_layout =
            Layout::horizontal([Constraint::Min(10), Constraint::Min(12), Constraint::Min(8)]);

        let [left, middle, right] = h_layout.areas(inner);
        {
            // left
            let v_layout = Layout::vertical([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Length(23),
            ]);

            let [_, game_type, stats] = v_layout.areas(left);

            frame.render_widget(tetris.clone().title_top("type"), game_type);

            {
                // stats
                let area = stats;
                let widget = tetris.clone().title_top("stats");
                let inner = widget.inner(area);
                frame.render_widget(widget, area);

                for i in 0..7 {
                    let x = inner.x;
                    let y = inner.y + i * 3;
                    let count = self.tetris.tetromino_count.content[i as usize];

                    let sub_inner = Rect {
                        x,
                        y,
                        width: 4,
                        height: 4,
                    };
                    let tetromino = &Tetromino::from(i as u8);
                    frame.render_widget(
                        TetrominoWidget {
                            tetromino,
                            palette: self.tetris.level as usize,
                        },
                        sub_inner,
                    );
                    let sub_inner = Rect {
                        x: x + 4,
                        y,
                        width: 4,
                        height: 4,
                    }
                    .inner(Margin::new(1, 1));
                    frame.render_widget(
                        CounterWidget {
                            count: count as u16,
                            pal: (col1, Reset),
                        },
                        sub_inner,
                    );
                }
            }
        }

        {
            // center
            let v_layout = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(3),
                Constraint::Length(22),
                Constraint::Length(1),
            ]);

            let [_, lines, game, _] = v_layout.areas(middle);
            {
                // lines
                let area = lines;
                let widget = tetris.clone().title_top("lines");
                let inner = widget.inner(area);
                frame.render_widget(widget, area);
                frame.render_widget(
                    Paragraph::new(format!("{:0>3}", self.tetris.lines)).style(style),
                    inner,
                );
            }

            {
                // game
                let area = game;
                let widget = tetris.clone().title_top("game");
                let inner = widget.inner(area);
                frame.render_widget(widget, area);

                let (x, y) = self.tetris.pos;

                let game_widget = GameWidget {
                    tetromino_board: &self.tetris.tetromino_board,
                    pos: (x as u16, y as u16),
                    tetromino: self.tetris.current_tetromino,
                    palette: self.tetris.level as usize,
                };

                frame.render_widget(game_widget, inner);
            }
        }

        {
            // right
            let v_layout = Layout::vertical([
                Constraint::Length(2), //
                Constraint::Length(3), //top
                Constraint::Length(1), //
                Constraint::Length(3), //score
                Constraint::Length(1), //
                Constraint::Length(6), //next
                Constraint::Length(1), //
                Constraint::Length(3), //level
                Constraint::Length(1), //
                Constraint::Length(5), //input
                Constraint::Length(2), //
            ]);
            let [_, top, _, score, _, next, _, level, _, input, _] = v_layout.areas(right);

            {
                // top
                let area = top;
                let widget = tetris.clone().title_top("Top");
                let inner = widget.inner(area);
                let top_score = self.tetris.top_score;
                frame.render_widget(
                    Paragraph::new(format!("{:0>6}", top_score)).style(style),
                    inner,
                );
                frame.render_widget(widget, area);
            }
            {
                // score
                let area = score;
                let widget = tetris.clone().title_top("Score");

                let inner = widget.inner(area);
                let score = self.tetris.score;
                frame.render_widget(Paragraph::new(format!("{:0>6}", score)).style(style), inner);
                frame.render_widget(widget, area);
            }
            {
                // next
                let area = next;
                let widget = tetris.clone().title_top("Next");
                let inner = widget.inner(area);
                let next = self.tetris.next_tetromino;
                next.get_shape();

                frame.render_widget(widget, area);
                frame.render_widget(
                    TetrominoWidget {
                        tetromino: &self.tetris.next_tetromino,
                        palette: self.tetris.level as usize,
                    },
                    inner.inner(Margin {
                        horizontal: 1,
                        vertical: 1,
                    }),
                );
            }
            {
                // level
                let area = level;
                let widget = tetris.clone().title_top("Level");
                let inner = widget.inner(area);
                let level = self.tetris.level;
                frame.render_widget(Paragraph::new(format!("{:0>3}", level)).style(style), inner);
                frame.render_widget(widget, area);
            }
            {
                // input
                let area = input;
                let input_widget = InputWidget {
                    input: &self.tetris.input,
                };
                frame.render_widget(input_widget, area);
            }
        }
    }

    fn center(area: Rect, horizontal: Constraint, vertical: Constraint) -> Rect {
        let [area] = Layout::horizontal([horizontal])
            .flex(Flex::Center)
            .areas(area);
        let [area] = Layout::vertical([vertical]).flex(Flex::Center).areas(area);
        area
    }

    fn handle_input(&mut self) -> Result<()> {
        self.input = Input::empty();
        if !event::poll(Duration::from_secs_f64(1.0 / 60.0))? {
            return Ok(());
        }
        match event::read()? {
            Event::FocusGained => {}
            Event::FocusLost => {}
            Event::Key(KeyEvent {
                code,
                kind: kind @ (KeyEventKind::Press | KeyEventKind::Release),
                ..
            }) => {
                let pressed = if let KeyEventKind::Press = kind {
                    true
                } else {
                    false
                };

                let mut press = |i: Input| self.input.set(i, pressed);

                match code {
                    /*KeyCode::Char('b') => {
                        press(Input::Left);
                        press(Input::Down);
                        self.tetris.frames_without_falling = 100;
                    }
                    KeyCode::Char('j') => {
                        press(Input::Right);
                        press(Input::Down);
                        self.tetris.frames_without_falling = 100;
                    }*/
                    KeyCode::Char('n') => {
                        self.tetris.level += 1;
                        self.tetris.update_level();
                    }
                    KeyCode::Char('r') => {
                        self.tetris.lose();
                    }
                    KeyCode::Left => press(Input::Left),
                    KeyCode::Right => press(Input::Right),
                    KeyCode::Up => press(Input::Up),
                    KeyCode::Down => press(Input::Down),
                    KeyCode::Enter => press(Input::START),
                    KeyCode::Backspace => press(Input::SELECT),
                    KeyCode::Esc | KeyCode::Char('q' | 'Q') => self.quit(),
                    KeyCode::Char(c) => match c.to_ascii_lowercase() {
                        'z' => press(Input::A),
                        'x' => press(Input::B),
                        _ => {}
                    },
                    _ => {}
                }
            }
            Event::Mouse(_) => {}
            Event::Paste(_) => {}
            Event::Resize(_, _) => {}
            Event::Key(_) => {}
        }

        Ok(())
    }
    fn update(&mut self) {
        self.tetris.update(self.input);
    }

    fn draw(&self, terminal: &mut DefaultTerminal) -> Result<()> {
        terminal.draw(|frame| self.render(frame))?;
        Ok(())
    }

    fn quit(&mut self) {
        self.running = false;
    }
}

fn main() -> Result<()> {
    let _ = TerminalGuard::new();
    color_eyre::install()?;

    RatatuiApp::default().run(ratatui::init())
}
