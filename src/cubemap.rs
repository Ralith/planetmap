use std::cmp::Ordering;
use std::ops::{Index, IndexMut};
use std::{iter, vec};

use crate::chunk::Face;

/// A dense, fixed-resolution cube map
///
/// Useful for storing and manipulating moderate-resolution samplings of radial functions such as
/// spherical heightmaps.
// TODO: Make this a DST so we can overlay it on Vulkan memory
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CubeMap<T> {
    resolution: usize,
    // Could save a usize by using a thin pointer, but all the unsafety and effort probably isn't worth it.
    data: Box<[T]>,
}

impl<T> CubeMap<T> {
    /// Construct a cube map with faces containing `resolution * resolution` slots, each initialized
    /// to `value`.
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
    ///
    /// Returns `None` if `data.len()` isn't correct for `resolution`, i.e. `resolution * resolution
    /// * 6`.
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

    /// Compute a cube map based on the direction of each slot
    pub fn from_fn(resolution: usize, mut f: impl FnMut(na::Unit<na::Vector3<f32>>) -> T) -> Self {
        Self {
            resolution,
            data: (0..resolution * resolution * 6)
                .map(|index| f(get_dir(resolution, index).unwrap()))
                .collect(),
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.into_iter()
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
    let wrt_face = face.basis().inverse() * x;
    (
        face,
        na::Vector2::new(wrt_face.x, wrt_face.y) * (0.5 / value) + na::Vector2::new(0.5, 0.5),
    )
}

impl<T> IntoIterator for CubeMap<T> {
    type Item = (na::Unit<na::Vector3<f32>>, T);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            resolution: self.resolution,
            data: self.data.into_vec().into_iter(),
            index: 0,
        }
    }
}

pub struct IntoIter<T> {
    resolution: usize,
    data: vec::IntoIter<T>,
    index: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (na::Unit<na::Vector3<f32>>, T);
    fn next(&mut self) -> Option<Self::Item> {
        let dir = get_dir(self.resolution, self.index)?;
        let value = self.data.next().unwrap();
        self.index += 1;
        Some((dir, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.resolution * self.resolution;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        let total = self.resolution * self.resolution;
        (total - self.index) as usize
    }
}

impl<'a, T> IntoIterator for &'a CubeMap<T> {
    type Item = (na::Unit<na::Vector3<f32>>, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            resolution: self.resolution,
            data: &self.data,
            index: 0,
        }
    }
}

pub struct Iter<'a, T> {
    resolution: usize,
    data: &'a [T],
    index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (na::Unit<na::Vector3<f32>>, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        let dir = get_dir(self.resolution, self.index)?;
        let value = &self.data[self.index];
        self.index += 1;
        Some((dir, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.resolution * self.resolution;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        let total = self.resolution * self.resolution;
        (total - self.index) as usize
    }
}

impl<'a, T> IntoIterator for &'a mut CubeMap<T> {
    type Item = (na::Unit<na::Vector3<f32>>, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            resolution: self.resolution,
            data: &mut self.data,
            index: 0,
        }
    }
}

pub struct IterMut<'a, T> {
    resolution: usize,
    data: &'a mut [T],
    index: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (na::Unit<na::Vector3<f32>>, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        let dir = get_dir(self.resolution, self.index)?;
        let value = &mut self.data[self.index] as *mut T;
        self.index += 1;
        // We promise calling next twice won't yield the same reference.
        Some((dir, unsafe { &mut *value }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.resolution * self.resolution;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize {
        let total = self.resolution * self.resolution;
        (total - self.index) as usize
    }
}

fn get_dir(resolution: usize, index: usize) -> Option<na::Unit<na::Vector3<f32>>> {
    let face_size = resolution * resolution;
    if index >= face_size * 6 {
        return None;
    }
    let face = [Face::PX, Face::NX, Face::PY, Face::NY, Face::PZ, Face::NZ][index / face_size];
    let rem = index % face_size;
    let texcoord = if resolution == 1 {
        na::Vector2::new(0.5, 0.5)
    } else {
        let y = rem / resolution;
        let x = rem % resolution;
        na::Vector2::new(x as f32, y as f32) / ((resolution - 1) as f32)
    };
    let on_z = texcoord * 2.0 - na::Vector2::new(1.0, 1.0);
    Some(face.direction(&on_z))
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
        assert_eq!(
            coords(&na::Vector3::x_axis()),
            (Face::PX, na::Vector2::new(0.5, 0.5))
        );
        assert_eq!(
            coords(&na::Vector3::y_axis()),
            (Face::PY, na::Vector2::new(0.5, 0.5))
        );
        assert_eq!(
            coords(&na::Vector3::z_axis()),
            (Face::PZ, na::Vector2::new(0.5, 0.5))
        );
        assert_eq!(
            coords(&-na::Vector3::x_axis()),
            (Face::NX, na::Vector2::new(0.5, 0.5))
        );
        assert_eq!(
            coords(&-na::Vector3::y_axis()),
            (Face::NY, na::Vector2::new(0.5, 0.5))
        );
        assert_eq!(
            coords(&-na::Vector3::z_axis()),
            (Face::NZ, na::Vector2::new(0.5, 0.5))
        );
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

    #[test]
    fn iter() {
        for res in 0..8 {
            let map = CubeMap::new(res, 0);
            assert_eq!(map.iter().count(), res * res * 6);
        }
    }

    #[test]
    fn addressing_roundtrip() {
        const RES: usize = 2049; // must be odd for there to be a point exactly on the axes
        for dir in &[
            na::Vector3::x_axis(),
            na::Vector3::y_axis(),
            na::Vector3::z_axis(),
            -na::Vector3::x_axis(),
            -na::Vector3::y_axis(),
            -na::Vector3::z_axis(),
            na::Unit::new_normalize(na::Vector3::new(1.0, 1.0, 1.0)),
        ] {
            let index = index(RES, &dir);
            let out = get_dir(RES, index).unwrap();
            assert_eq!(dir, &out);
        }
    }
}
