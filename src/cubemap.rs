use std::cmp::Ordering;
#[cfg(feature = "simd")]
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Neg};
use std::{fmt, iter, vec};

use na::RealField;
#[cfg(feature = "simd")]
use simdeez::Simd;

/// A dense, fixed-resolution cube map
///
/// Useful for storing and manipulating moderate-resolution samplings of radial functions such as
/// spherical heightmaps.
///
/// For addressing purposes, texels along the edges and at the corners of a face do *not* overlap
/// with their neighbors. Note that `Coords::samples` nonetheless *does* produce samples that will
/// overlap along the edges of neighboring `Coords`.
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

impl<T> Index<Coords> for CubeMap<T> {
    type Output = T;
    fn index(&self, coord: Coords) -> &T {
        let face_size = self.resolution * self.resolution;
        let offset = face_size * coord.face as usize;
        &self.data[offset + self.resolution * coord.y as usize + coord.x as usize]
    }
}

impl<T> IndexMut<Coords> for CubeMap<T> {
    fn index_mut(&mut self, coord: Coords) -> &mut T {
        let face_size = self.resolution * self.resolution;
        let offset = face_size * coord.face as usize;
        &mut self.data[offset + self.resolution * coord.y as usize + coord.x as usize]
    }
}

impl<'a, T> Index<&'a na::Vector3<f32>> for CubeMap<T> {
    type Output = T;
    fn index(&self, x: &'a na::Vector3<f32>) -> &T {
        &self.data[index(self.resolution, x)]
    }
}

impl<'a, T> IndexMut<&'a na::Vector3<f32>> for CubeMap<T> {
    fn index_mut(&mut self, x: &'a na::Vector3<f32>) -> &mut T {
        &mut self.data[index(self.resolution, x)]
    }
}

fn index(resolution: usize, x: &na::Vector3<f32>) -> usize {
    let (face, texcoords) = Face::coords(x);
    let texel = discretize(resolution, texcoords);
    face as usize * resolution * resolution + texel.1 * resolution + texel.0
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
    let y = (rem / resolution) as u32;
    let x = (rem % resolution) as u32;
    Some(Coords { x, y, face }.center(resolution as u32))
}

/// Face of a cube map, identified by direction
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Face {
    /// The face in the +X direction
    PX,
    /// The face in the -X direction
    NX,
    /// The face in the +Y direction
    PY,
    /// The face in the -Y direction
    NY,
    /// The face in the +Z direction
    PZ,
    /// The face in the -Z direction
    NZ,
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Face::*;
        let s = match *self {
            PX => "+X",
            NX => "-X",
            PY => "+Y",
            NY => "-Y",
            PZ => "+Z",
            NZ => "-Z",
        };
        f.write_str(s)
    }
}

impl Neg for Face {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Face::*;
        match self {
            PX => NX,
            PY => NY,
            PZ => NZ,
            NX => PX,
            NY => PY,
            NZ => PZ,
        }
    }
}

impl Face {
    /// Find the face that intersects a vector originating at the center of a cube
    pub fn from_vector<N: RealField + PartialOrd>(x: &na::Vector3<N>) -> Self {
        let (&value, &axis) = x
            .iter()
            .zip(&[Face::PX, Face::PY, Face::PZ])
            .max_by(|(l, _), (r, _)| l.abs().partial_cmp(&r.abs()).unwrap_or(Ordering::Less))
            .unwrap();
        if value.is_sign_negative() {
            -axis
        } else {
            axis
        }
    }

    /// Compute which `Face` a vector intersects, and where the intersection lies
    pub fn coords<N: RealField>(x: &na::Vector3<N>) -> (Face, na::Point2<N>) {
        let face = Self::from_vector(x);
        let wrt_face = face.basis().inverse_transform_vector(x);
        (
            face,
            na::Point2::from(wrt_face.xy() * (na::convert::<_, N>(0.5) / wrt_face.z))
                + na::convert::<_, na::Vector2<N>>(na::Vector2::new(0.5, 0.5)),
        )
    }

    /// Transform from face space (facing +Z) to sphere space (facing the named axis).
    pub fn basis<N: RealField>(self) -> na::Rotation3<N> {
        use self::Face::*;
        let (x, y, z) = match self {
            PX => (na::Vector3::z(), -na::Vector3::y(), na::Vector3::x()),
            NX => (-na::Vector3::z(), -na::Vector3::y(), -na::Vector3::x()),
            PY => (na::Vector3::x(), -na::Vector3::z(), na::Vector3::y()),
            NY => (na::Vector3::x(), na::Vector3::z(), -na::Vector3::y()),
            PZ => (na::Vector3::x(), na::Vector3::y(), na::Vector3::z()),
            NZ => (-na::Vector3::x(), na::Vector3::y(), -na::Vector3::z()),
        };
        na::Rotation3::from_matrix_unchecked(na::Matrix3::from_columns(&[x, y, z]))
    }

    /// Iterator over all `Face`s
    pub fn iter() -> impl Iterator<Item = Face> {
        const VALUES: &[Face] = &[Face::PX, Face::NX, Face::PY, Face::NY, Face::PZ, Face::NZ];
        VALUES.iter().cloned()
    }

    /// Neighboring faces wrt. local axes [-x, -y, +x, +y].
    ///
    /// Returns the neighboring face, the edge of that face, and whether the axis shared with that face is parallel or antiparallel.
    ///
    /// Index by `sign << 1 | axis`.
    pub(crate) fn neighbors(self) -> &'static [(Face, Edge, bool); 4] {
        use self::Face::*;
        match self {
            PX => &[
                (NZ, Edge::NX, false),
                (PY, Edge::PX, false),
                (PZ, Edge::PX, false),
                (NY, Edge::PX, true),
            ],
            NX => &[
                (PZ, Edge::NX, false),
                (PY, Edge::NX, true),
                (NZ, Edge::PX, false),
                (NY, Edge::NX, false),
            ],
            PY => &[
                (NX, Edge::NY, true),
                (PZ, Edge::PY, true),
                (PX, Edge::NY, false),
                (NZ, Edge::PY, false),
            ],
            NY => &[
                (NX, Edge::PY, false),
                (NZ, Edge::NY, false),
                (PX, Edge::PY, true),
                (PZ, Edge::NY, true),
            ],
            PZ => &[
                (NX, Edge::NX, false),
                (NY, Edge::PY, true),
                (PX, Edge::PX, false),
                (PY, Edge::NY, true),
            ],
            NZ => &[
                (PX, Edge::NX, false),
                (NY, Edge::NY, false),
                (NX, Edge::PX, false),
                (PY, Edge::PY, false),
            ],
        }
    }

    /// Compute the direction identified by a [0..1]^2 vector on this face
    pub fn direction<N: RealField>(self, coords: &na::Point2<N>) -> na::Unit<na::Vector3<N>> {
        let dir_z = na::Unit::new_normalize(na::Vector3::new(coords.x, coords.y, N::one()));
        self.basis() * dir_z
    }
}

/// Coordinates in a discretized cubemap
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Coords {
    pub x: u32,
    pub y: u32,
    pub face: Face,
}

impl Coords {
    pub fn from_vector(resolution: u32, vector: &na::Vector3<f32>) -> Self {
        let (face, unit_coords) = Face::coords(vector);
        let (x, y) = discretize(resolution as usize, unit_coords);
        Self {
            x: x as u32,
            y: y as u32,
            face,
        }
    }

    pub fn neighbors(&self, resolution: u32) -> [Self; 4] {
        let Coords { x, y, face } = *self;
        let max = resolution - 1;
        let neighbor_chunk = |face: Face, edge: Edge| {
            let (neighboring_face, neighbor_edge, parallel_axis) = face.neighbors()[edge as usize];
            let other = match edge {
                Edge::NX | Edge::PX => y,
                Edge::NY | Edge::PY => x,
            };
            let other = if parallel_axis { other } else { max - other };
            let (x, y) = match neighbor_edge {
                Edge::NX => (0, other),
                Edge::NY => (other, 0),
                Edge::PX => (max, other),
                Edge::PY => (other, max),
            };
            Coords {
                x,
                y,
                face: neighboring_face,
            }
        };
        [
            if x == 0 {
                neighbor_chunk(face, Edge::NX)
            } else {
                Coords { x: x - 1, y, face }
            },
            if y == 0 {
                neighbor_chunk(face, Edge::NY)
            } else {
                Coords { x, y: y - 1, face }
            },
            if x == max {
                neighbor_chunk(face, Edge::PX)
            } else {
                Coords { x: x + 1, y, face }
            },
            if y == max {
                neighbor_chunk(face, Edge::PY)
            } else {
                Coords { x, y: y + 1, face }
            },
        ]
    }

    /// Select all `Coords` intersecting a cone opening towards `direction` with half-angle `theta`
    pub fn neighborhood(
        resolution: u32,
        direction: na::Vector3<f32>,
        theta: f32,
    ) -> impl Iterator<Item = Self> {
        /// Map [-1, 1] to [0, 1]
        fn remap(x: f32) -> f32 {
            (na::clamp(x, -1.0, 1.0) + 1.0) / 2.0
        }
        Face::iter()
            .filter(move |f| {
                (f.basis() * na::Vector3::z())
                    .dot(&direction)
                    .is_sign_positive()
            })
            .map(move |face| {
                let local = face.basis().inverse_transform_vector(&direction);
                let local = local.xy() / local.z;
                // atan(x / 1) = angle of `local` around Y axis through cube origin ("midpoint x")
                let theta_m_x = local.x.atan();
                // tan(θ_mx - θ) * 1 = coordinate of the intersection of the X lower bound with the cube
                let x_lower = (theta_m_x - theta).tan();
                // tan(θ_mx + θ) * 1 = coordinate of the intersection of the X upper bound with the cube
                let x_upper = (theta_m_x + theta).tan();
                // once more, perpendicular!
                let theta_m_y = local.y.atan();
                let y_lower = (theta_m_y - theta).tan();
                let y_upper = (theta_m_y + theta).tan();
                (face, (x_lower, y_lower), (x_upper, y_upper))
            })
            .filter(|(_, lower, upper)| {
                lower.0 <= 1.0 && lower.1 <= 1.0 && upper.0 >= -1.0 && upper.1 >= -1.0
            })
            .flat_map(move |(face, lower, upper)| {
                let (x_lower, y_lower) = discretize(
                    resolution as usize,
                    na::Point2::new(remap(lower.0), remap(lower.1)),
                );
                let (x_upper, y_upper) = discretize(
                    resolution as usize,
                    na::Point2::new(remap(upper.0), remap(upper.1)),
                );
                (y_lower..=y_upper).flat_map(move |y| {
                    (x_lower..=x_upper).map(move |x| Self {
                        x: x as u32,
                        y: y as u32,
                        face,
                    })
                })
            })
    }

    /// The approximate direction represented by these coordinates
    pub fn center<N: RealField>(&self, resolution: u32) -> na::Unit<na::Vector3<N>> {
        let texcoord = if resolution == 1 {
            na::Point2::new(0.5, 0.5)
        } else {
            na::Point2::new(self.x as f64, self.y as f64) / ((resolution - 1) as f64)
        };
        let on_z = texcoord * 2.0 - na::Vector2::new(1.0, 1.0);
        self.face.direction(&na::convert::<_, na::Point2<N>>(on_z))
    }

    /// Compute the direction identified by a point in the [0..1]^2 area covered by these
    /// coordinates
    pub fn direction<N: RealField>(
        &self,
        resolution: u32,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        let edge_length = Self::edge_length::<N>(resolution);
        let origin_on_face = na::Point2::from(
            na::convert::<_, na::Vector2<N>>(na::Vector2::new(self.x as f64, self.y as f64))
                * edge_length,
        ) - na::convert::<_, na::Vector2<N>>(na::Vector2::new(1.0, 1.0));
        let pos_on_face = origin_on_face + coords.coords * edge_length;
        self.face.direction(&pos_on_face)
    }

    /// Length of an edge of the bounding square of the cubemap-space area covered by a coordinate
    /// in a cubemap with a particular `resolution`
    pub fn edge_length<N: RealField>(resolution: u32) -> N {
        na::convert::<_, N>(2.0) / na::convert::<_, N>(resolution as f64)
    }

    /// Returns a grid of resolution^2 directions contained by these coords, in scan-line order
    ///
    /// - `face_resolution` represents the number of coordinates along an edge of the cubemap
    /// - `chunk_resolution` represents the number of samples along an edge of this specific coordinate
    ///
    /// Edge/corner samples lie *directly* on the edge/corner, and hence are *not* the centers of
    /// traditionally addressed texels. In other words, the first sample has position (0, 0), not
    /// (0.5/w, 0.5/h), and the last has position (1, 1), not (1 - 0.5/w, 1 - 0.5/h). This allows
    /// for seamless interpolation in the neighborhood of chunk edges/corners without needing access
    /// to data for neighboring chunks.
    pub fn samples(&self, face_resolution: u32, chunk_resolution: u32) -> SampleIter {
        SampleIter {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
        }
    }

    /// SIMD variant of `samples`.
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "simd")]
    pub fn samples_ps<S: Simd>(
        &self,
        face_resolution: u32,
        chunk_resolution: u32,
    ) -> SampleIterSimd<S> {
        SampleIterSimd {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
            _simd: PhantomData,
        }
    }
}

/// Edge of a quad
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Edge {
    NX = 0,
    NY = 1,
    PX = 2,
    PY = 3,
}

impl Edge {
    /// Iterator over all `Edge`s
    pub fn iter() -> impl Iterator<Item = Edge> {
        [Edge::NX, Edge::NY, Edge::PX, Edge::PY].iter().cloned()
    }
}

impl Neg for Edge {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Edge::*;
        match self {
            PX => NX,
            PY => NY,
            NX => PX,
            NY => PY,
        }
    }
}

/// Iterator over sample points distributed in a regular grid across a chunk, including its edges
#[derive(Debug)]
pub struct SampleIter {
    coords: Coords,
    face_resolution: u32,
    chunk_resolution: u32,
    index: u32,
}

impl Iterator for SampleIter {
    type Item = na::Unit<na::Vector3<f32>>;
    fn next(&mut self) -> Option<na::Unit<na::Vector3<f32>>> {
        if self.index >= self.chunk_resolution * self.chunk_resolution {
            return None;
        }
        let max = self.chunk_resolution - 1;
        let coords = if max == 0 {
            na::Point2::new(0.5, 0.5)
        } else {
            let step = 1.0 / max as f32;
            let (x, y) = (
                self.index % self.chunk_resolution,
                self.index / self.chunk_resolution,
            );
            na::Point2::new(x as f32, y as f32) * step
        };
        let dir = self.coords.direction(self.face_resolution, &coords);
        self.index += 1;
        Some(dir)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.chunk_resolution * self.chunk_resolution;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SampleIter {
    fn len(&self) -> usize {
        let total = self.chunk_resolution * self.chunk_resolution;
        (total - self.index) as usize
    }
}

/// Iterator over sample points distributed in a regular grid across a chunk, including its edges
///
/// Hand-vectorized, returning batches of each dimension in a separate register.
#[cfg(feature = "simd")]
#[derive(Debug)]
pub struct SampleIterSimd<S> {
    coords: Coords,
    face_resolution: u32,
    chunk_resolution: u32,
    index: u32,
    _simd: PhantomData<S>,
}

#[cfg(feature = "simd")]
impl<S: Simd> Iterator for SampleIterSimd<S> {
    type Item = [S::Vf32; 3];
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunk_resolution * self.chunk_resolution {
            return None;
        }
        unsafe {
            let edge_length = S::set1_ps(Coords::edge_length::<f32>(self.face_resolution));
            let origin_on_face_x = S::fmsub_ps(
                S::set1_ps(self.coords.x as f32),
                edge_length,
                S::set1_ps(1.0),
            );
            let origin_on_face_y = S::fmsub_ps(
                S::set1_ps(self.coords.y as f32),
                edge_length,
                S::set1_ps(1.0),
            );
            let max = self.chunk_resolution - 1;
            let (offset_x, offset_y) = if max == 0 {
                let v = S::set1_ps(0.5) * edge_length;
                (v, v)
            } else {
                let step = edge_length / S::set1_ps(max as f32);
                let mut xs = S::setzero_ps();
                for i in 0..S::VF32_WIDTH {
                    xs[i] = ((self.index + i as u32) % self.chunk_resolution) as f32;
                }
                let mut ys = S::setzero_ps();
                for i in 0..S::VF32_WIDTH {
                    ys[i] = ((self.index + i as u32) / self.chunk_resolution) as f32;
                }
                (xs * step, ys * step)
            };
            let pos_on_face_x = origin_on_face_x + offset_x;
            let pos_on_face_y = origin_on_face_y + offset_y;

            let len = S::sqrt_ps(S::fmadd_ps(
                pos_on_face_y,
                pos_on_face_y,
                S::fmadd_ps(pos_on_face_x, pos_on_face_x, S::set1_ps(1.0)),
            ));
            let dir_x = pos_on_face_x / len;
            let dir_y = pos_on_face_y / len;
            let dir_z = S::set1_ps(1.0) / len;

            self.index += S::VF32_WIDTH as u32;
            Some([dir_x, dir_y, dir_z])
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.chunk_resolution * self.chunk_resolution;
        let remaining = (total - self.index) as usize;
        let x = (remaining + S::VF32_WIDTH - 1) / S::VF32_WIDTH;
        (x, Some(x))
    }
}

#[cfg(feature = "simd")]
impl<S: Simd> ExactSizeIterator for SampleIterSimd<S> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

/// Map real coordinates in [0, 1)^2 to integer coordinates in [0, n)^2 such that each integer
/// covers exactly the same distance
fn discretize(resolution: usize, texcoords: na::Point2<f32>) -> (usize, usize) {
    let texcoords = texcoords * resolution as f32;
    let max = resolution - 1;
    (
        na::clamp(texcoords.x as usize, 0, max),
        na::clamp(texcoords.y as usize, 0, max),
    )
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

    #[test]
    fn face_neighbors() {
        // Opposite faces are incident to opposite edges
        assert_eq!(Face::PX.neighbors()[0b00].1, -Face::NX.neighbors()[0b10].1);
        assert_eq!(Face::PX.neighbors()[0b01].1, -Face::NX.neighbors()[0b01].1);
        assert_eq!(Face::PX.neighbors()[0b10].1, -Face::NX.neighbors()[0b00].1);
        assert_eq!(Face::PX.neighbors()[0b11].1, -Face::NX.neighbors()[0b11].1);

        assert_eq!(Face::PY.neighbors()[0b00].1, -Face::NY.neighbors()[0b00].1);
        assert_eq!(Face::PY.neighbors()[0b01].1, -Face::NY.neighbors()[0b11].1);
        assert_eq!(Face::PY.neighbors()[0b10].1, -Face::NY.neighbors()[0b10].1);
        assert_eq!(Face::PY.neighbors()[0b11].1, -Face::NY.neighbors()[0b01].1);

        assert_eq!(Face::PZ.neighbors()[0b00].1, -Face::NZ.neighbors()[0b10].1);
        assert_eq!(Face::PZ.neighbors()[0b01].1, -Face::NZ.neighbors()[0b01].1);
        assert_eq!(Face::PZ.neighbors()[0b10].1, -Face::NZ.neighbors()[0b00].1);
        assert_eq!(Face::PZ.neighbors()[0b11].1, -Face::NZ.neighbors()[0b11].1);
    }

    #[test]
    fn face_neighbor_axes() {
        // Neighboring faces correctly track whether the axes they intersect on in their local
        // reference frames are parallel or antiparallel
        for face in Face::iter() {
            for (edge, &(neighbor, neighbor_edge, parallel)) in Edge::iter().zip(face.neighbors()) {
                let local = face.basis()
                    * match edge {
                        Edge::PX | Edge::NX => na::Vector3::y(),
                        Edge::PY | Edge::NY => na::Vector3::x(),
                    };
                let neighbor = neighbor.basis()
                    * match neighbor_edge {
                        Edge::PX | Edge::NX => na::Vector3::y(),
                        Edge::PY | Edge::NY => na::Vector3::x(),
                    };
                let sign = if parallel { 1.0 } else { -1.0 };
                assert_eq!(local, sign * neighbor);
            }
        }
    }

    #[test]
    fn face_coord_sanity() {
        assert_eq!(
            Face::coords(&na::Vector3::x()),
            (Face::PX, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&na::Vector3::y()),
            (Face::PY, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&na::Vector3::z()),
            (Face::PZ, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::x()),
            (Face::NX, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::y()),
            (Face::NY, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::z()),
            (Face::NZ, na::Point2::new(0.5, 0.5))
        );
    }

    #[test]
    fn coord_neighborhood() {
        use Face::*;
        for face in Face::iter() {
            let center = face.basis() * na::Vector3::z();
            assert_eq!(
                Coords::neighborhood(1, center, 0.1).collect::<Vec<_>>(),
                vec![Coords { x: 0, y: 0, face }]
            );
            assert_eq!(
                Coords::neighborhood(3, center, 0.1).collect::<Vec<_>>(),
                vec![Coords { x: 1, y: 1, face }]
            );
            let xs = Coords::neighborhood(2, center, 0.1).collect::<Vec<_>>();
            assert_eq!(xs.len(), 4);
            for expected in (0..2).flat_map(|y| (0..2).map(move |x| Coords { x, y, face })) {
                assert!(xs.contains(&expected));
            }
        }
        assert_eq!(
            Coords::neighborhood(1, na::Vector3::new(1.0, 1.0, 1.0), 0.1).collect::<Vec<_>>(),
            vec![
                Coords {
                    x: 0,
                    y: 0,
                    face: PX
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: PY
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: PZ
                }
            ]
        );
        assert_eq!(
            Coords::neighborhood(1, na::Vector3::new(1.0, 1.0, 0.0), 0.1).collect::<Vec<_>>(),
            vec![
                Coords {
                    x: 0,
                    y: 0,
                    face: PX
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: PY
                },
            ]
        );
        assert_eq!(
            Coords::neighborhood(5, na::Vector3::new(1.0, 1.0, 1.0), 0.1).count(),
            3
        );
    }

    #[test]
    fn discretize_sanity() {
        assert_eq!(discretize(100, &na::Point2::new(1.0, 1.0)), (99, 99));
        assert_eq!(discretize(100, &na::Point2::new(0.0, 0.0)), (0, 0));
        assert_eq!(discretize(100, &na::Point2::new(0.990, 0.990)), (99, 99));
        assert_eq!(discretize(100, &na::Point2::new(0.989, 0.989)), (98, 98));
        assert_eq!(discretize(100, &na::Point2::new(0.010, 0.010)), (1, 1));
        assert_eq!(discretize(100, &na::Point2::new(0.009, 0.009)), (0, 0));

        assert_eq!(discretize(2, &na::Point2::new(0.49, 0.49)), (0, 0));
        assert_eq!(discretize(2, &na::Point2::new(0.50, 0.50)), (1, 1));
    }
}
