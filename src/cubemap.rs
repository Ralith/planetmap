use std::cmp::Ordering;
use std::iter;
use std::ops::{Index, IndexMut};

use crate::chunk::Face;

/// A dense, fixed-resolution cube map
///
/// Useful for storing and manipulating moderate-resolution samplings of radial functions such as
/// spherical heightmaps.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CubeMap<T> {
    resolution: usize,
    data: Box<[T]>,
}

impl<T> CubeMap<T> {
    /// Construct a cube map with `resolution` by `resolution` faces, containing `value` at every
    /// position
    pub fn new(resolution: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            resolution,
            data: iter::repeat(value)
                .take(resolution * resolution * 6)
                .collect::<Box<[T]>>(),
        }
    }

    /// Copy a cube map from a contiguous slice of `resolution` by `resolution` data, in +X, -X, +Y,
    /// -Y, +Z, -Z order.
    pub fn from_slice(resolution: usize, data: &[T]) -> Option<Self>
    where
        T: Copy,
    {
        if data.len() != resolution * resolution * 6 {
            return None;
        }
        Some(Self {
            resolution,
            data: data.into(),
        })
    }
}

impl<T> AsRef<[T]> for CubeMap<T> {
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T> AsMut<[T]> for CubeMap<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T> Index<Face> for CubeMap<T> {
    type Output = [T];
    fn index(&self, face: Face) -> &[T] {
        let face_size = self.resolution * self.resolution;
        let offset = face_size * face as usize;
        &self.data[offset..offset + face_size]
    }
}

impl<T> IndexMut<Face> for CubeMap<T> {
    fn index_mut(&mut self, face: Face) -> &mut [T] {
        let face_size = self.resolution * self.resolution;
        let offset = face_size * face as usize;
        &mut self.data[offset..offset + face_size]
    }
}

impl<'a, T> Index<&'a na::Unit<na::Vector3<f32>>> for CubeMap<T> {
    type Output = T;
    fn index(&self, x: &'a na::Unit<na::Vector3<f32>>) -> &T {
        &self.data[index(self.resolution, x)]
    }
}

impl<'a, T> IndexMut<&'a na::Unit<na::Vector3<f32>>> for CubeMap<T> {
    fn index_mut(&mut self, x: &'a na::Unit<na::Vector3<f32>>) -> &mut T {
        &mut self.data[index(self.resolution, x)]
    }
}

fn index(resolution: usize, x: &na::Unit<na::Vector3<f32>>) -> usize {
    let (face, texcoords) = coords(x);
    let texel = discretize(resolution, &texcoords);
    face as usize * resolution * resolution + texel.1 * resolution + texel.0
}

fn discretize(resolution: usize, texcoords: &na::Vector2<f32>) -> (usize, usize) {
    let texcoords = texcoords * (resolution - 1) as f32 + na::Vector2::new(0.5, 0.5);
    (texcoords.x as usize, texcoords.y as usize)
}

fn coords(x: &na::Unit<na::Vector3<f32>>) -> (Face, na::Vector2<f32>) {
    let (&value, &axis) = x
        .iter()
        .zip(&[Face::PX, Face::PY, Face::PZ])
        .max_by(|(l, _), (r, _)| l.abs().partial_cmp(&r.abs()).unwrap_or(Ordering::Less))
        .unwrap();
    let face = if value < 0.0 { -axis } else { axis };
    let wrt_face = na::convert::<_, na::Rotation3<f32>>(face.basis()).inverse() * x;
    (face, na::Vector2::new(wrt_face.x, wrt_face.y) * (0.5 / value) + na::Vector2::new(0.5, 0.5))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn index_sanity() {
        const RES: usize = 2048;
        assert_eq!(index(RES, &na::Vector3::x_axis()), (RES / 2) * (RES + 1));
    }

    #[test]
    fn coord_sanity() {
        assert_eq!(coords(&na::Vector3::x_axis()), (Face::PX, na::Vector2::new(0.5, 0.5)));
        assert_eq!(coords(&na::Vector3::y_axis()), (Face::PY, na::Vector2::new(0.5, 0.5)));
        assert_eq!(coords(&na::Vector3::z_axis()), (Face::PZ, na::Vector2::new(0.5, 0.5)));
        assert_eq!(coords(&-na::Vector3::x_axis()), (Face::NX, na::Vector2::new(0.5, 0.5)));
        assert_eq!(coords(&-na::Vector3::y_axis()), (Face::NY, na::Vector2::new(0.5, 0.5)));
        assert_eq!(coords(&-na::Vector3::z_axis()), (Face::NZ, na::Vector2::new(0.5, 0.5)));
    }

    #[test]
    fn discretize_sanity() {
        assert_eq!(discretize(100, &na::Vector2::new(1.0, 1.0)), (99, 99));
        assert_eq!(discretize(100, &na::Vector2::new(0.0, 0.0)), (0, 0));
        assert_eq!(discretize(100, &na::Vector2::new(0.996, 0.996)), (99, 99));
        assert_eq!(discretize(100, &na::Vector2::new(0.004, 0.004)), (0, 0));
        assert_eq!(discretize(100, &na::Vector2::new(0.006, 0.006)), (1, 1));
        assert_eq!(discretize(100, &na::Vector2::new(0.994, 0.994)), (98, 98));
    }
}
