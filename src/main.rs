mod point;

use bitflags::bitflags;
use color_eyre::Result;
use rand::random;
use ratatui::buffer::{Buffer, Cell};
use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers,
};
use ratatui::crossterm::{ExecutableCommand, event};
use ratatui::layout::Alignment::Center;
use ratatui::layout::{Constraint, Flex, Layout, Margin, Rect};
use ratatui::prelude::{Alignment, Stylize, Text, Widget};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::block::Title;
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
enum Tetramino {
    #[default]
    O = 0,
    I(Rotation2),
    S(Rotation2),
    Z(Rotation2),
    L(Rotation4),
    J(Rotation4),
    T(Rotation4),
}

macro_rules! tetramino4 {
    ($($bytes:literal)*) => {{
        let mut result: u16 = 0;
        let mut offset = 16;
        $(
            offset -= 4;
            result |= ($bytes as u16) << offset;
        )*

        result
    }}
}
impl From<u8> for Tetramino {
    fn from(value: u8) -> Self {
        match value {
            0 => Tetramino::O,
            1 => Tetramino::I(Rotation2::Sideways),
            2 => Tetramino::S(Rotation2::Sideways),
            3 => Tetramino::Z(Rotation2::Sideways),
            4 => Tetramino::L(Rotation4::Right),
            5 => Tetramino::J(Rotation4::Right),
            6 => Tetramino::T(Rotation4::Down),
            _ => unreachable!(),
        }
    }
}
impl From<Tetramino> for usize {
    fn from(value: Tetramino) -> Self {
        match value {
            Tetramino::O => 0,
            Tetramino::I(_) => 1,
            Tetramino::S(_) => 2,
            Tetramino::Z(_) => 3,
            Tetramino::L(_) => 4,
            Tetramino::J(_) => 5,
            Tetramino::T(_) => 6,
        }
    }
}

type Shape = [Line; 4];
impl Tetramino {
    pub fn rotate_left(&mut self) {
        match self {
            Tetramino::O => {}
            Tetramino::I(r) => *r = r.prev(),
            Tetramino::S(r) => *r = r.prev(),
            Tetramino::Z(r) => *r = r.prev(),
            Tetramino::L(r) => *r = r.prev(),
            Tetramino::J(r) => *r = r.prev(),
            Tetramino::T(r) => *r = r.prev(),
        }
    }
    pub fn rotate_right(&mut self) {
        match self {
            Tetramino::O => {}
            Tetramino::I(r) => *r = r.next(),
            Tetramino::S(r) => *r = r.next(),
            Tetramino::Z(r) => *r = r.next(),
            Tetramino::L(r) => *r = r.next(),
            Tetramino::J(r) => *r = r.next(),
            Tetramino::T(r) => *r = r.next(),
        }
    }

    const fn get_shape(&self) -> Shape {
        match self {
            Tetramino::O => [
                0b_0000, //
                0b_0000, //
                0b_0110, // ##
                0b_0110, // ##
            ],

            Tetramino::I(r) => match r {
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
            Tetramino::S(r) => match r {
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
            Tetramino::Z(r) => match r {
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
            Tetramino::L(r) => match r {
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
            Tetramino::J(r) => match r {
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
            Tetramino::T(r) => match r {
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
struct TetraminoMap<T> {
    content: [T; 7],
}

impl<T> TetraminoMap<T> {}
impl<T> Index<Tetramino> for TetraminoMap<T> {
    type Output = T;

    fn index(&self, tetramino: Tetramino) -> &Self::Output {
        let i: usize = tetramino.into();
        &self.content[i]
    }
}
impl<T> IndexMut<Tetramino> for TetraminoMap<T> {
    fn index_mut(&mut self, tetramino: Tetramino) -> &mut Self::Output {
        let i: usize = tetramino.into();
        &mut self.content[i]
    }
}

type TetraminoBoard = TetraminoMap<Grid>;
type TetraminoCount = TetraminoMap<usize>;
type TetraminoBlocks = TetraminoMap<(Color, Color, &'static str)>;

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

    fn is_just_released(&self, input: Input) -> bool {
        self.was_pressed(input) && !self.is_pressed(input)
    }

    fn update(&mut self, input: Input) {
        self.0 = self.1;
        self.1 = input;
    }

    fn clear(&mut self) {
        *self = InputBuffer(Input::empty(), Input::empty());
    }
}

struct GameState {
    input: InputBuffer,
    collision_board: Grid,
    tetramino_board: TetraminoBoard,
    tetramino_count: TetraminoCount,
    pos: (u8, u8),

    current_tetramino: Tetramino,
    next_tetramino: Tetramino,

    score: u16,
    lines: u16,
    level: u16,

    frames_without_falling: u8,
    top_score: u32,
}

fn get_axis<T: From<bool> + Sub<Output = T>>(neg: bool, pos: bool) -> T {
    let n: T = neg.into();
    let p: T = pos.into();
    p - n
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            input: Default::default(),
            collision_board: GRID,
            tetramino_board: Default::default(),
            tetramino_count: Default::default(),
            pos: (5, 0),
            current_tetramino: Default::default(),
            next_tetramino: Default::default(),
            score: 0,
            lines: 0,
            level: 1,
            frames_without_falling: 0,
            top_score: 0,
        }
    }
}
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
            if !self.check_collision(x1, y, &self.current_tetramino) {
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
            let mut tetramino = self.current_tetramino;
            match rot {
                -1 => tetramino.rotate_left(),
                1 => tetramino.rotate_right(),
                _ => unreachable!(),
            }
            if !self.check_collision(x, y, &tetramino) {
                self.current_tetramino = tetramino;
            }
        }

        'fall: {
            self.frames_without_falling += 1;
            should_fall |= self.frames_without_falling >= 30;
            if !should_fall {
                break 'fall;
            }
            if self.check_collision(x, y + 1, &self.current_tetramino) {
                self.commit();
                return;
            } else {
                y = y + 1;
            }
            self.frames_without_falling = 0;
        }

        self.pos = (x, y);
    }

    fn check_collision(&self, x: u8, y: u8, tetramino: &Tetramino) -> bool {
        let yu = y as usize;
        let xu = x as u16;

        let lines_to_check: &[Line] = &self.collision_board[yu..(yu + 4)];
        let tetra_board: [Line; 4] = tetramino.get_shape().map(|it| it << xu);
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .any(|(a, b)| a & b != 0)
    }

    fn commit(&mut self) {
        let (x, y) = self.pos;
        let yu = y as usize;
        let xu = x as u16;
        let tetramino = self.current_tetramino;

        let lines_to_check: &mut [Line] = &mut self.collision_board[yu..yu + 4];

        let tetra_board: [Line; 4] = tetramino.get_shape().map(|it| it << xu);
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .for_each(|(a, b)| *a |= b);

        let tetramino_grid = &mut self.tetramino_board[tetramino];
        let lines_to_check: &mut [Line] = &mut tetramino_grid[yu..yu + 4];
        lines_to_check
            .into_iter()
            .zip(&tetra_board)
            .for_each(|(a, b)| *a |= b);

        self.current_tetramino = self.next_tetramino;
        self.next_tetramino = self.generate_next_tetramino();
        self.frames_without_falling = 0;
        self.pos = (WIDTH as u8 / 2, 0);
    }

    fn generate_next_tetramino(&self) -> Tetramino {
        let next: u8 = random();
        Tetramino::from(next % 7)
    }
}

#[derive(Default)]
struct Renderer {}
impl Renderer {}

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

struct GameWidget<'a> {
    tetramino_board: &'a TetraminoBoard,
    grid: &'a Grid,
    pos: (u16, u16),
    tetramino: Tetramino,
}

fn get_bit(source: u16, bit: u16) -> bool {
    source & (1 << bit) != 0
}

//const BLOCK_CHAR: &'static str = "□";
const BLOCK_CHAR: &'static str = "#";

const BLOCKS: TetraminoBlocks = TetraminoMap {
    content: [
        (Color::Red, Color::LightRed, BLOCK_CHAR),
        (Color::Green, Color::LightGreen, BLOCK_CHAR),
        (Color::Blue, Color::LightBlue, BLOCK_CHAR),
        (Color::Yellow, Color::LightYellow, BLOCK_CHAR),
        (Color::Blue, Color::LightBlue, BLOCK_CHAR),
        (Color::Red, Color::LightRed, BLOCK_CHAR),
        (Color::Green, Color::LightGreen, BLOCK_CHAR),
    ],
};

impl Widget for GameWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let (ox, oy) = (area.x, area.y);
        for (index, board) in self.tetramino_board.content.iter().enumerate() {
            for (j, line) in board.iter().enumerate() {
                for i in 0..16 {
                    let c = get_bit(*line, i);
                    if c {
                        let (bg, fg, c) = BLOCKS.content[index];
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

        for (j, line) in self.tetramino.get_shape().iter().enumerate() {
            for i in 0..4 {
                let c = get_bit(*line, i);
                if c {
                    let (bg, fg, c) = BLOCKS[self.tetramino];
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
                (false, false) => (Color::Reset, Color::Reset),
                (false, true) => (Color::Yellow, Color::Red),
                (true, true) => (Color::Reset, Color::Red),
                (true, false) => (Color::Red, Color::Reset),
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
            .title_style(Style::from(Color::Red))
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
            .fg(Color::Red);
        frame.render_widget(too_small_text, inner);
    }

    fn draw_tetris(&self, frame: &mut Frame, area: Rect) {
        let tetris = Block::new()
            .borders(Borders::all())
            .border_type(BorderType::Rounded)
            .border_style(Style::new())
            .title_style(Style::from(Color::Green))
            .title_alignment(Center);
        frame.render_widget(tetris.clone().title_top("TETRIS"), area);

        let inner = tetris.inner(area);

        let h_layout =
            Layout::horizontal([Constraint::Min(10), Constraint::Min(12), Constraint::Min(8)]);

        let [left, middle, right] = h_layout.areas(inner);
        'left: {
            let v_layout = Layout::vertical([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Length(21),
                Constraint::Length(1),
            ]);

            let [_, game_type, stats, _] = v_layout.areas(left);

            frame.render_widget(tetris.clone().title_top("type"), game_type);
            frame.render_widget(tetris.clone().title_top("stats"), stats);
        }

        'center: {
            let v_layout = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(3),
                Constraint::Length(22),
                Constraint::Length(1),
            ]);

            let [_, lines, game, _] = v_layout.areas(middle);
            'lines: {
                let area = lines;
                frame.render_widget(tetris.clone().title_top("lines"), area);
            }

            'game: {
                let area = game;
                let widget = tetris.clone().title_top("game");
                let inner = widget.inner(area);
                frame.render_widget(widget, area);

                let (x, y) = self.tetris.pos;

                let game_widget = GameWidget {
                    tetramino_board: &self.tetris.tetramino_board,
                    grid: &self.tetris.collision_board,
                    pos: (x as u16, y as u16),
                    tetramino: self.tetris.current_tetramino,
                };

                frame.render_widget(game_widget, inner);
            }
        }

        'right: {
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

            'top: {
                let area = top;
                let widget = tetris.clone().title_top("Top");
                let inner = widget.inner(area);
                let top_score = self.tetris.top_score;
                frame.render_widget(Paragraph::new(format!("{:0>6}", top_score)), inner);
                frame.render_widget(widget, area);
            }
            'score: {
                let area = score;
                let widget = tetris.clone().title_top("Score");

                let inner = widget.inner(area);
                let score = self.tetris.score;
                frame.render_widget(Paragraph::new(format!("{:0>6}", score)), inner);
                frame.render_widget(widget, area);
            }
            'next: {
                let area = next;
                let widget = tetris.clone().title_top("Next");
                let inner = widget.inner(area);
                let next = self.tetris.next_tetramino;
                let shape = next.get_shape();

                fn bool_to_tile(b: bool) -> u8 {
                    if b { b'#' } else { b' ' }
                }
                fn bit_to_tile(line: Line, mask: u16) -> u8 {
                    bool_to_tile(line & mask != 0)
                }
                fn line_to_str(line: Line) -> [u8; 4] {
                    [
                        bit_to_tile(line, 1 << 0),
                        bit_to_tile(line, 1 << 1),
                        bit_to_tile(line, 1 << 2),
                        bit_to_tile(line, 1 << 3),
                    ]
                }

                let line0_ = line_to_str(shape[2]);
                let line1_ = line_to_str(shape[3]);

                let line0 = unsafe { std::str::from_utf8_unchecked(&line0_) };
                let line1 = unsafe { std::str::from_utf8_unchecked(&line1_) };

                frame.render_widget(widget, area);
                frame.render_widget(
                    Paragraph::new(format!("{line0}\n{line1}")),
                    inner.inner(Margin {
                        horizontal: 1,
                        vertical: 1,
                    }),
                );
            }
            'level: {
                let area = level;
                let widget = tetris.clone().title_top("Level");
                let inner = widget.inner(area);
                let level = self.tetris.level;
                frame.render_widget(Paragraph::new(format!("{:0>3}", level)), inner);
                frame.render_widget(widget, area);
            }
            'input: {
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
