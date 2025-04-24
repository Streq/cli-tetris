use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Point<T>(T, T);

//From
impl<T> From<Point<T>> for (T, T) {
    fn from(p: Point<T>) -> Self {
        (p.0, p.1)
    }
}
impl<T> From<(T, T)> for Point<T> {
    fn from((x, y): (T, T)) -> Self {
        Self(x, y)
    }
}
impl<T: Copy> From<&Point<T>> for (T, T) {
    fn from(p: &Point<T>) -> Self {
        (p.0, p.1)
    }
}
impl<T: Copy> From<&(T, T)> for Point<T> {
    fn from((x, y): &(T, T)) -> Self {
        Self(*x, *y)
    }
}

impl<T> From<Point<T>> for [T; 2] {
    fn from(p: Point<T>) -> Self {
        [p.0, p.1]
    }
}
impl<T> From<[T; 2]> for Point<T> {
    fn from([x, y]: [T; 2]) -> Self {
        Self(x, y)
    }
}
impl<T: Copy> From<&Point<T>> for [T; 2] {
    fn from(p: &Point<T>) -> Self {
        [p.0, p.1]
    }
}
impl<T: Copy> From<&[T; 2]> for Point<T> {
    fn from([x, y]: &[T; 2]) -> Self {
        Self(*x, *y)
    }
}

// map type functions
impl<T> Point<T> {
    pub fn map<U, F>(self, mut f: F) -> Point<U>
    where
        F: FnMut(T) -> U,
    {
        Point(f(self.0), f(self.1))
    }
}
impl<T> Point<T> {
    pub fn map_ref<U, F>(&self, mut f: F) -> Point<U>
    where
        F: FnMut(&T) -> U,
    {
        Point(f(&self.0), f(&self.1))
    }
}

impl<T> Point<T> {
    pub fn map_mut<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut T),
    {
        f(&mut self.0);
        f(&mut self.1);
        self
    }
}
impl<T> Point<T> {
    pub fn zip_map<U, V, F>(self, other: Point<U>, mut f: F) -> Point<V>
    where
        F: FnMut(T, U) -> V,
    {
        Point(f(self.0, other.0), f(self.1, other.1))
    }
}

impl<T: Copy> Point<T> {
    pub fn map_mut_copy<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(T) -> T,
    {
        self.0 = f(self.0);
        self.1 = f(self.1);
        self
    }
}

impl<T: Copy + Ord> Point<T> {
    pub fn clamp_ref(&self, min: &Self, max: &Self) -> Self {
        Self(self.0.max(min.0).min(max.0), self.1.max(min.1).min(max.1))
    }
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.max(min.0).min(max.0), self.1.max(min.1).min(max.1))
    }
    pub fn is_inside_inclusive(&self, min: Self, max: Self) -> bool {
        (min.0..=max.0).contains(&self.0) && (min.1..=max.1).contains(&self.1)
    }
    pub fn is_inside(&self, min: Self, max: Self) -> bool {
        (min.0..=max.0).contains(&self.0) && (min.1..=max.1).contains(&self.1)
    }
}

macro_rules! impl_point_ops {
    ($trait:ident, $method:ident) => {
        impl<T: $trait<Output = T>> $trait for Point<T> {
            type Output = Point<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                let Point(x1, y1) = self;
                let Point(x2, y2) = rhs;
                Point(x1.$method(x2), y1.$method(y2))
            }
        }
    };
}
macro_rules! impl_point_assign_ops {
    ($trait:ident, $method:ident) => {
        impl<T: $trait> $trait for Point<T> {
            fn $method(&mut self, rhs: Self) {
                self.0.$method(rhs.0);
                self.1.$method(rhs.1);
            }
        }
    };
}
macro_rules! impl_point_ref_op {
    ($trait:ident, $method:ident) => {
        impl<T: $trait<Output = T> + Copy> $trait<&Point<T>> for &Point<T> {
            type Output = Point<T>;

            fn $method(self, rhs: &Point<T>) -> Self::Output {
                Point(self.0.$method(rhs.0), self.1.$method(rhs.1))
            }
        }
    };
}
macro_rules! impl_point_neg {
    () => {
        impl<T: std::ops::Neg<Output = T>> std::ops::Neg for Point<T> {
            type Output = Point<T>;
            fn neg(self) -> Self::Output {
                Point(-self.0, -self.1)
            }
        }
    };
}

impl_point_ops!(Add, add);
impl_point_ops!(Sub, sub);
impl_point_ops!(Mul, mul);
impl_point_ops!(Div, div);
impl_point_ops!(Rem, rem);

impl_point_ref_op!(Add, add);
impl_point_ref_op!(Sub, sub);
impl_point_ref_op!(Mul, mul);
impl_point_ref_op!(Div, div);
impl_point_ref_op!(Rem, rem);

impl_point_assign_ops!(AddAssign, add_assign);
impl_point_assign_ops!(SubAssign, sub_assign);
impl_point_assign_ops!(MulAssign, mul_assign);
impl_point_assign_ops!(DivAssign, div_assign);
impl_point_assign_ops!(RemAssign, rem_assign);

impl_point_neg!();

#[cfg(test)]
mod tests {
    use crate::point::Point;

    #[test]
    fn test() {
        assert_eq!(&Point(1, 1), &Point(1, 1));
        assert_eq!(Point(2, 1), &Point(1, 1) + &Point(1, 0));
        assert_eq!(Point(0, 1), &Point(1, 1) - &Point(1, 0));
        assert_eq!(Point(9, 9), Point(3, 3) * Point(3, 3));
        assert_eq!(Point(-1, -1), -Point(1, 1));
    }

    #[test]
    fn test_map() {
        let mut p = Point(1, 1);
        p.map_mut_copy(|a| a + 1);

        assert_eq!(Point(2, 2), p);

        let mut p = Point(1, 1);
        p.map_mut_copy(|a| a + 1).map_mut_copy(|a| a + 2);
        assert_eq!(Point(4, 4), p);

        let &mut p = Point(1, 1)
            .map_mut_copy(|a| a + 1)
            .map_mut_copy(|a| a + 2)
            .map_mut(|a| *a -= 1);
        assert_eq!(Point(3, 3), p);
    }

    #[test]
    fn test_clamp() {
        let lt = Point(1, 1);
        let rb = Point(16, 8);

        assert_eq!(Point(3, 6), Point(3, 6).clamp(lt, rb));
        assert_eq!(Point(16, 6), Point(177, 6).clamp(lt, rb));
        assert_eq!(Point(1, 6), Point(0, 6).clamp(lt, rb));
        assert_eq!(Point(3, 8), Point(3, 177).clamp(lt, rb));
        assert_eq!(Point(3, 1), Point(3, -2).clamp(lt, rb));
    }

    #[test]
    fn test_is_inside() {
        let lt = Point(1, 1);
        let rb = Point(16, 8);

        assert!(Point(3, 6).is_inside(lt, rb));
        assert!(!Point(177, 6).is_inside(lt, rb));
        assert!(!Point(0, 6).is_inside(lt, rb));
        assert!(!Point(3, 177).is_inside(lt, rb));
        assert!(!Point(3, -2).is_inside(lt, rb));
    }
}
