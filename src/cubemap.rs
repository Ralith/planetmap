use std::cmp::Ordering;
#[cfg(feature = "simd")]
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Neg};
use std::{alloc, fmt, mem, ptr};

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
#[derive(Debug)]
#[repr(C)]
pub struct CubeMap<T> {
    resolution: u32,
    data: [T],
}

impl<T> CubeMap<T> {
    /// Access a cubemap in existing memory
    ///
    /// # Safety
    ///
    /// `ptr` must address a `u32` containing the cubemap's resolution, positioned just before the
    /// cubemap's data. The addressed memory must outlive `'a` and have no outstanding unique
    /// borrows.
    pub unsafe fn from_raw<'a>(ptr: *const u32) -> &'a Self {
        let dim = ptr.read() as usize;
        let len = dim * dim * 6;
        &*(ptr::slice_from_raw_parts(ptr, len) as *const Self)
    }

    /// Uniquely access a cubemap in existing memory
    ///
    /// # Safety
    ///
    /// `ptr` must address a `u32` containing the cubemap's resolution, positioned just before the
    /// cubemap's data. The addressed memory must outlive `'a` and have no outstanding borrows.
    pub unsafe fn from_raw_mut<'a>(ptr: *mut u32) -> &'a mut Self {
        let dim = ptr.read() as usize;
        let len = dim * dim * 6;
        &mut *(ptr::slice_from_raw_parts_mut(ptr, len) as *mut Self)
    }

    /// Construct a cube map with faces containing `resolution * resolution` slots, each initialized
    /// to `value`.
    pub fn new(resolution: u32, value: T) -> Box<Self>
    where
        T: Clone,
    {
        let payload_len = resolution as usize * resolution as usize * 6;
        let header_layout = alloc::Layout::new::<u32>();
        let (layout, offset) = header_layout
            .extend(
                alloc::Layout::from_size_align(
                    mem::size_of::<T>() * payload_len,
                    mem::align_of::<T>(),
                )
                .unwrap(),
            )
            .unwrap();
        let layout = layout.pad_to_align();
        unsafe {
            let mem = alloc::alloc(layout);
            mem.cast::<u32>().write(resolution);
            let payload = mem.add(offset).cast::<T>();
            for i in 0..payload_len {
                payload.add(i).write(value.clone());
            }
            Box::from_raw(ptr::slice_from_raw_parts_mut(mem, payload_len) as *mut Self)
        }
    }

    /// Copy a cube map from a contiguous slice of `resolution` by `resolution` data, in +X, -X, +Y,
    /// -Y, +Z, -Z order.
    ///
    /// Returns `None` if `data.len()` isn't correct for `resolution`, i.e. `resolution * resolution
    /// * 6`.
    pub fn from_slice(resolution: u32, data: &[T]) -> Option<Box<Self>>
    where
        T: Copy,
    {
        let payload_len = resolution as usize * resolution as usize * 6;
        if data.len() != payload_len {
            return None;
        }

        let align = mem::align_of::<T>().max(4); // Also the size of the header with padding
        let layout =
            alloc::Layout::from_size_align(align + mem::size_of::<T>() * payload_len, align)
                .unwrap();
        unsafe {
            let mem = alloc::alloc(layout);
            mem.cast::<u32>().write(resolution);
            let payload = mem.add(align).cast::<T>();
            for (i, &x) in data.iter().enumerate() {
                payload.add(i).write(x);
            }
            Some(Box::from_raw(
                ptr::slice_from_raw_parts_mut(mem, payload_len) as *mut Self,
            ))
        }
    }

    /// Compute a cube map based on the direction of each slot
    pub fn from_fn(
        resolution: u32,
        mut f: impl FnMut(na::Unit<na::Vector3<f32>>) -> T,
    ) -> Box<Self> {
        let payload_len = resolution as usize * resolution as usize * 6;
        let align = mem::align_of::<T>().max(4); // Also the size of the header with padding
        let layout =
            alloc::Layout::from_size_align(align + mem::size_of::<T>() * payload_len, align)
                .unwrap();
        unsafe {
            let mem = alloc::alloc(layout);
            mem.cast::<u32>().write(resolution);
            let payload = mem.add(align).cast::<T>();
            for i in 0..payload_len {
                payload.add(i).write(f(get_dir(resolution, i).unwrap()));
            }
            Box::from_raw(ptr::slice_from_raw_parts_mut(mem, payload_len) as *mut Self)
        }
    }

    pub fn resolution(&self) -> u32 {
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
        &self.data
    }
}

impl<T> AsMut<[T]> for CubeMap<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Index<Face> for CubeMap<T> {
    type Output = [T];
    fn index(&self, face: Face) -> &[T] {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * face as usize;
        &self.data[offset..offset + face_size]
    }
}

impl<T> IndexMut<Face> for CubeMap<T> {
    fn index_mut(&mut self, face: Face) -> &mut [T] {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * face as usize;
        &mut self.data[offset..offset + face_size]
    }
}

impl<T> Index<Coords> for CubeMap<T> {
    type Output = T;
    fn index(&self, coord: Coords) -> &T {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * coord.face as usize;
        &self.data[offset + self.resolution as usize * coord.y as usize + coord.x as usize]
    }
}

impl<T> IndexMut<Coords> for CubeMap<T> {
    fn index_mut(&mut self, coord: Coords) -> &mut T {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * coord.face as usize;
        &mut self.data[offset + self.resolution as usize * coord.y as usize + coord.x as usize]
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

fn index(resolution: u32, x: &na::Vector3<f32>) -> usize {
    let resolution = resolution as usize;
    let (face, texcoords) = Face::coords(x);
    let texel = discretize(resolution, texcoords);
    face as usize * resolution * resolution + texel.1 * resolution + texel.0
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
    resolution: u32,
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
        let total = self.resolution as usize * self.resolution as usize;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        let total = self.resolution as usize * self.resolution as usize;
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
    resolution: u32,
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
        let total = self.resolution as usize * self.resolution as usize;
        let remaining = (total - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize {
        let total = self.resolution as usize * self.resolution as usize;
        (total - self.index) as usize
    }
}

fn get_dir(resolution: u32, index: usize) -> Option<na::Unit<na::Vector3<f32>>> {
    let face_size = resolution as usize * resolution as usize;
    if index >= face_size * 6 {
        return None;
    }
    let face = [Face::Px, Face::Nx, Face::Py, Face::Ny, Face::Pz, Face::Nz][index / face_size];
    let rem = index % face_size;
    let y = (rem / resolution as usize) as u32;
    let x = (rem % resolution as usize) as u32;
    Some(Coords { x, y, face }.center(resolution))
}

/// Face of a cube map, identified by direction
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Face {
    /// The face in the +X direction
    Px,
    /// The face in the -X direction
    Nx,
    /// The face in the +Y direction
    Py,
    /// The face in the -Y direction
    Ny,
    /// The face in the +Z direction
    Pz,
    /// The face in the -Z direction
    Nz,
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Face::*;
        let s = match *self {
            Px => "+X",
            Nx => "-X",
            Py => "+Y",
            Ny => "-Y",
            Pz => "+Z",
            Nz => "-Z",
        };
        f.write_str(s)
    }
}

impl Neg for Face {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Face::*;
        match self {
            Px => Nx,
            Py => Ny,
            Pz => Nz,
            Nx => Px,
            Ny => Py,
            Nz => Pz,
        }
    }
}

impl Face {
    /// Find the face that intersects a vector originating at the center of a cube
    pub fn from_vector<N: RealField + PartialOrd>(x: &na::Vector3<N>) -> Self {
        let (&value, &axis) = x
            .iter()
            .zip(&[Face::Px, Face::Py, Face::Pz])
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
            Px => (na::Vector3::z(), -na::Vector3::y(), na::Vector3::x()),
            Nx => (-na::Vector3::z(), -na::Vector3::y(), -na::Vector3::x()),
            Py => (na::Vector3::x(), -na::Vector3::z(), na::Vector3::y()),
            Ny => (na::Vector3::x(), na::Vector3::z(), -na::Vector3::y()),
            Pz => (na::Vector3::x(), na::Vector3::y(), na::Vector3::z()),
            Nz => (-na::Vector3::x(), na::Vector3::y(), -na::Vector3::z()),
        };
        na::Rotation3::from_matrix_unchecked(na::Matrix3::from_columns(&[x, y, z]))
    }

    /// Iterator over all `Face`s
    pub fn iter() -> impl Iterator<Item = Face> {
        const VALUES: &[Face] = &[Face::Px, Face::Nx, Face::Py, Face::Ny, Face::Pz, Face::Nz];
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
            Px => &[
                (Nz, Edge::Nx, false),
                (Py, Edge::Px, false),
                (Pz, Edge::Px, false),
                (Ny, Edge::Px, true),
            ],
            Nx => &[
                (Pz, Edge::Nx, false),
                (Py, Edge::Nx, true),
                (Nz, Edge::Px, false),
                (Ny, Edge::Nx, false),
            ],
            Py => &[
                (Nx, Edge::Ny, true),
                (Pz, Edge::Py, true),
                (Px, Edge::Ny, false),
                (Nz, Edge::Py, false),
            ],
            Ny => &[
                (Nx, Edge::Py, false),
                (Nz, Edge::Ny, false),
                (Px, Edge::Py, true),
                (Pz, Edge::Ny, true),
            ],
            Pz => &[
                (Nx, Edge::Nx, false),
                (Ny, Edge::Py, true),
                (Px, Edge::Px, false),
                (Py, Edge::Ny, true),
            ],
            Nz => &[
                (Px, Edge::Nx, false),
                (Ny, Edge::Ny, false),
                (Nx, Edge::Px, false),
                (Py, Edge::Py, false),
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
            let (neighboring_face, neighbor_edge, parallel_axis) = face.neighbors()[edge];
            let other = match edge {
                Edge::Nx | Edge::Px => y,
                Edge::Ny | Edge::Py => x,
            };
            let other = if parallel_axis { other } else { max - other };
            let (x, y) = match neighbor_edge {
                Edge::Nx => (0, other),
                Edge::Ny => (other, 0),
                Edge::Px => (max, other),
                Edge::Py => (other, max),
            };
            Coords {
                x,
                y,
                face: neighboring_face,
            }
        };
        [
            if x == 0 {
                neighbor_chunk(face, Edge::Nx)
            } else {
                Coords { x: x - 1, y, face }
            },
            if y == 0 {
                neighbor_chunk(face, Edge::Ny)
            } else {
                Coords { x, y: y - 1, face }
            },
            if x == max {
                neighbor_chunk(face, Edge::Px)
            } else {
                Coords { x: x + 1, y, face }
            },
            if y == max {
                neighbor_chunk(face, Edge::Py)
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
    Nx = 0,
    Ny = 1,
    Px = 2,
    Py = 3,
}

impl Edge {
    /// Iterator over all `Edge`s
    pub fn iter() -> impl Iterator<Item = Edge> {
        [Edge::Nx, Edge::Ny, Edge::Px, Edge::Py].iter().cloned()
    }
}

impl Neg for Edge {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Edge::*;
        match self {
            Px => Nx,
            Py => Ny,
            Nx => Px,
            Ny => Py,
        }
    }
}

impl<T> Index<Edge> for [T] {
    type Output = T;
    fn index(&self, edge: Edge) -> &T {
        &self[edge as usize]
    }
}

impl<T> IndexMut<Edge> for [T] {
    fn index_mut(&mut self, edge: Edge) -> &mut T {
        &mut self[edge as usize]
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

            let basis = self.coords.face.basis::<f32>();
            let basis = basis.matrix();
            let x = S::fmadd_ps(
                S::set1_ps(basis.m11),
                dir_x,
                S::fmadd_ps(S::set1_ps(basis.m12), dir_y, S::set1_ps(basis.m13) * dir_z),
            );
            let y = S::fmadd_ps(
                S::set1_ps(basis.m21),
                dir_x,
                S::fmadd_ps(S::set1_ps(basis.m22), dir_y, S::set1_ps(basis.m23) * dir_z),
            );
            let z = S::fmadd_ps(
                S::set1_ps(basis.m31),
                dir_x,
                S::fmadd_ps(S::set1_ps(basis.m32), dir_y, S::set1_ps(basis.m33) * dir_z),
            );

            self.index += S::VF32_WIDTH as u32;
            Some([x, y, z])
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
    use approx::*;

    #[test]
    fn index_sanity() {
        const RES: u32 = 2048;
        assert_eq!(
            index(RES, &na::Vector3::x_axis()) as u32,
            (RES / 2) * (RES + 1)
        );
    }

    #[test]
    fn iter() {
        for res in 0..8 {
            let map = CubeMap::new(res, 0);
            assert_eq!(map.iter().count() as u32, res * res * 6);
        }
    }

    #[test]
    fn addressing_roundtrip() {
        const RES: u32 = 2049; // must be odd for there to be a point exactly on the axes
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
        assert_eq!(Face::Px.neighbors()[0b00].1, -Face::Nx.neighbors()[0b10].1);
        assert_eq!(Face::Px.neighbors()[0b01].1, -Face::Nx.neighbors()[0b01].1);
        assert_eq!(Face::Px.neighbors()[0b10].1, -Face::Nx.neighbors()[0b00].1);
        assert_eq!(Face::Px.neighbors()[0b11].1, -Face::Nx.neighbors()[0b11].1);

        assert_eq!(Face::Py.neighbors()[0b00].1, -Face::Ny.neighbors()[0b00].1);
        assert_eq!(Face::Py.neighbors()[0b01].1, -Face::Ny.neighbors()[0b11].1);
        assert_eq!(Face::Py.neighbors()[0b10].1, -Face::Ny.neighbors()[0b10].1);
        assert_eq!(Face::Py.neighbors()[0b11].1, -Face::Ny.neighbors()[0b01].1);

        assert_eq!(Face::Pz.neighbors()[0b00].1, -Face::Nz.neighbors()[0b10].1);
        assert_eq!(Face::Pz.neighbors()[0b01].1, -Face::Nz.neighbors()[0b01].1);
        assert_eq!(Face::Pz.neighbors()[0b10].1, -Face::Nz.neighbors()[0b00].1);
        assert_eq!(Face::Pz.neighbors()[0b11].1, -Face::Nz.neighbors()[0b11].1);
    }

    #[test]
    fn face_neighbor_axes() {
        // Neighboring faces correctly track whether the axes they intersect on in their local
        // reference frames are parallel or antiparallel
        for face in Face::iter() {
            for (edge, &(neighbor, neighbor_edge, parallel)) in Edge::iter().zip(face.neighbors()) {
                let local = face.basis()
                    * match edge {
                        Edge::Px | Edge::Nx => na::Vector3::y(),
                        Edge::Py | Edge::Ny => na::Vector3::x(),
                    };
                let neighbor = neighbor.basis()
                    * match neighbor_edge {
                        Edge::Px | Edge::Nx => na::Vector3::y(),
                        Edge::Py | Edge::Ny => na::Vector3::x(),
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
            (Face::Px, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&na::Vector3::y()),
            (Face::Py, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&na::Vector3::z()),
            (Face::Pz, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::x()),
            (Face::Nx, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::y()),
            (Face::Ny, na::Point2::new(0.5, 0.5))
        );
        assert_eq!(
            Face::coords(&-na::Vector3::z()),
            (Face::Nz, na::Point2::new(0.5, 0.5))
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
                    face: Px
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Py
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Pz
                }
            ]
        );
        assert_eq!(
            Coords::neighborhood(1, na::Vector3::new(1.0, 1.0, 0.0), 0.1).collect::<Vec<_>>(),
            vec![
                Coords {
                    x: 0,
                    y: 0,
                    face: Px
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Py
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
        assert_eq!(discretize(100, na::Point2::new(1.0, 1.0)), (99, 99));
        assert_eq!(discretize(100, na::Point2::new(0.0, 0.0)), (0, 0));
        assert_eq!(discretize(100, na::Point2::new(0.990, 0.990)), (99, 99));
        assert_eq!(discretize(100, na::Point2::new(0.989, 0.989)), (98, 98));
        assert_eq!(discretize(100, na::Point2::new(0.010, 0.010)), (1, 1));
        assert_eq!(discretize(100, na::Point2::new(0.009, 0.009)), (0, 0));

        assert_eq!(discretize(2, na::Point2::new(0.49, 0.49)), (0, 0));
        assert_eq!(discretize(2, na::Point2::new(0.50, 0.50)), (1, 1));
    }

    #[test]
    fn from_raw() {
        let data = [1, 0, 1, 2, 3, 4, 5];
        let x = unsafe { CubeMap::<u32>::from_raw(data.as_ptr()) };
        assert_eq!(x.resolution(), 1);
        assert_eq!(x.as_ref().len(), 6);
        for i in 0..6 {
            assert_eq!(x.as_ref()[i], data[i + 1]);
        }
    }

    #[test]
    fn samples_sanity() {
        const COORDS: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Py,
        };
        assert_abs_diff_eq!(COORDS.samples(1, 1).next().unwrap(), na::Vector3::y_axis());
        let corners = COORDS
            .samples(1, 2)
            .map(|x| x.into_inner())
            .collect::<Vec<_>>();
        let corner = na::Unit::new_normalize(na::Vector3::new(1.0, 1.0, 1.0));
        assert_abs_diff_eq!(
            corners[..],
            [
                na::Vector3::new(-corner.x, corner.y, corner.z),
                na::Vector3::new(corner.x, corner.y, corner.z),
                na::Vector3::new(-corner.x, corner.y, -corner.z),
                na::Vector3::new(corner.x, corner.y, -corner.z),
            ][..]
        );
    }

    #[test]
    fn neighboring_samples_align() {
        const LEFT: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        const RIGHT: Coords = Coords {
            x: 1,
            y: 0,
            face: Face::Pz,
        };
        let left = LEFT.samples(2, 2).collect::<Vec<_>>();
        let right = RIGHT.samples(2, 2).collect::<Vec<_>>();
        assert_abs_diff_eq!(left[1], right[0]);
        assert_abs_diff_eq!(left[3], right[2]);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn simd_samples_consistent() {
        use simdeez::scalar::Scalar;

        const COORDS: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Py,
        };
        const FACE_RES: u32 = 1;
        const CHUNK_RES: u32 = 17;
        let scalar = COORDS.samples(FACE_RES, CHUNK_RES);
        let simd = COORDS.samples_ps::<Scalar>(FACE_RES, CHUNK_RES);
        assert_eq!(simd.len(), scalar.len());
        for (scalar, [x, y, z]) in scalar.zip(simd) {
            dbg!(x.0, y.0, z.0);
            assert_abs_diff_eq!(
                scalar,
                na::Unit::new_unchecked(na::Vector3::new(x.0, y.0, z.0))
            );
        }
    }
}
