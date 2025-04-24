mod point;

use std::ops::{Index, IndexMut, Sub};
use bitflags::bitflags;

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
            Self::Left => Self::Right,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone)]
enum Tetramino {
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

bitflags! {
    #[derive(Copy, Clone, Eq, PartialEq)]
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

    fn push(&mut self, input: Input) {
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
}

fn get_axis<T: From<bool> + Sub<Output = T>>(neg: bool, pos: bool) -> T {
    let n: T = neg.into();
    let p: T = pos.into();
    p - n
}

impl GameState {
    fn update(&mut self) {
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

        let lines_to_check: &[Line] = &self.collision_board[yu..yu + 4];
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

        self.current_tetramino = self.next_tetramino;
        self.next_tetramino = self.generate_next_tetramino();
        self.pos = (WIDTH as u8 / 2, 0);
    }

    fn generate_next_tetramino(&self) -> Tetramino {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");
}
