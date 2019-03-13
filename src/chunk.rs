use std::cmp::Ordering;
#[cfg(feature = "simd")]
use std::marker::PhantomData;
use std::ops::Neg;
use std::{fmt, mem};

#[cfg(feature = "simd")]
use simdeez::Simd;

use na::Real;

use crate::addressing;

/// A node of a quadtree on a particular cubemap face
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Chunk {
    /// Coordinates within the set of nodes at this depth in the quadtree
    pub coords: Coords,
    /// Depth in the quadtree
    pub depth: u8,
}

impl Chunk {
    /// The top-level chunk corresponding to a particular cubemap face
    pub fn root(face: Face) -> Self {
        Self {
            coords: Coords { x: 0, y: 0, face },
            depth: 0,
        }
    }

    /// Compute the chunk at `depth` that intersects the vector from the origin towards `dir`
    pub fn from_vector(depth: u8, dir: &na::Vector3<f32>) -> Self {
        Self {
            coords: Coords::from_vector(2u32.pow(depth as u32), dir),
            depth,
        }
    }

    /// The smallest chunk that contains this chunk
    pub fn parent(&self) -> Option<Self> {
        let depth = self.depth.checked_sub(1)?;
        Some(Self {
            coords: Coords {
                x: self.coords.x / 2,
                y: self.coords.y / 2,
                face: self.coords.face,
            },
            depth,
        })
    }

    /// Iterator over the path from this chunk to the root.
    pub fn path(self) -> Path {
        Path { chunk: self }
    }

    /// The largest chunks contained by this chunk
    pub fn children(&self) -> [Self; 4] {
        let depth = self.depth + 1;
        let (x, y) = (self.coords.x * 2, self.coords.y * 2);
        let face = self.coords.face;
        [
            Chunk {
                coords: Coords { x, y, face },
                depth,
            },
            Chunk {
                coords: Coords { x, y: y + 1, face },
                depth,
            },
            Chunk {
                coords: Coords { x: x + 1, y, face },
                depth,
            },
            Chunk {
                coords: Coords {
                    x: x + 1,
                    y: y + 1,
                    face,
                },
                depth,
            },
        ]
    }

    /// Chunks that share an edge with this chunk
    pub fn neighbors(&self) -> [Self; 4] {
        let x = self.coords.neighbors(self.resolution());
        let depth = self.depth;
        [
            Chunk {
                coords: x[0],
                depth,
            },
            Chunk {
                coords: x[1],
                depth,
            },
            Chunk {
                coords: x[2],
                depth,
            },
            Chunk {
                coords: x[3],
                depth,
            },
        ]
    }

    /// Length of one of this chunk's edges before mapping onto the sphere
    pub fn edge_length<N: Real>(&self) -> N {
        Coords::edge_length(self.resolution())
    }

    /// Location of the center of this chunk on the surface of the sphere
    ///
    /// Transform by face.basis() to get world origin
    pub fn origin_on_face<N: Real>(&self) -> na::Unit<na::Vector3<N>> {
        let size = self.edge_length::<N>();
        let vec = na::Vector3::new(
            (na::convert::<_, N>(self.coords.x as f64) + na::convert::<_, N>(0.5)) * size
                - na::convert(1.0),
            (na::convert::<_, N>(self.coords.y as f64) + na::convert::<_, N>(0.5)) * size
                - na::convert(1.0),
            na::convert(1.0),
        );
        na::Unit::new_normalize(vec)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    pub fn samples(&self, resolution: u32) -> SampleIter {
        self.coords.samples(self.resolution(), resolution)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "simd")]
    pub fn samples_ps<S: Simd>(&self, resolution: u32) -> SampleIterSimd<S> {
        self.coords.samples_ps(self.resolution(), resolution)
    }

    /// Compute the single-precision origin of the chunk relative to its side of the sphere, and a
    /// transform from chunk-local space around that origin to view space.
    ///
    /// By computing the worldview matrix with double precision and rounding down, this allows
    /// vertices on a chunk to be efficiently and seamlessly computed by a GPU for planet-sized
    /// spheres.
    pub fn worldview(
        &self,
        sphere_radius: f64,
        view: &na::IsometryMatrix3<f64>,
    ) -> (na::Point3<f32>, na::IsometryMatrix3<f32>) {
        let origin =
            na::convert::<_, na::Vector3<f32>>(sphere_radius * self.origin_on_face().into_inner());
        let world = self.coords.face.basis()
            * na::Translation3::from(na::convert::<_, na::Vector3<f64>>(origin));
        (na::Point3::from(origin), na::convert(view * world))
    }

    /// Compute the direction identified by a [0..1]^2 vector on this chunk
    pub fn direction<N: Real>(&self, coords: &na::Point2<N>) -> na::Unit<na::Vector3<N>> {
        self.coords.direction(self.resolution(), coords)
    }

    /// Number of chunks at this depth along a cubemap edge, i.e. 2^depth
    pub fn resolution(&self) -> u32 { 2u32.pow(self.depth as u32) }
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
        let (x, y) = addressing::discretize(resolution as usize, &unit_coords);
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
                let local = face.basis().inverse() * &direction;
                let local = local.xy() / local.z;
                let theta_m_x = local.x.atan();
                let x_lower = (theta_m_x - theta).tan();
                let x_upper = (theta_m_x + theta).tan();
                let theta_m_y = local.y.atan();
                let y_lower = (theta_m_y - theta).tan();
                let y_upper = (theta_m_y + theta).tan();
                (face, (x_lower, y_lower), (x_upper, y_upper))
            })
            .filter(|(_, lower, upper)| {
                lower.0 <= 1.0 && lower.1 <= 1.0 && upper.0 >= -1.0 && upper.1 >= -1.0
            })
            .flat_map(move |(face, lower, upper)| {
                let (x_lower, y_lower) = addressing::discretize(
                    resolution as usize,
                    &na::Point2::new(remap(lower.0), remap(lower.1)),
                );
                let (x_upper, y_upper) = addressing::discretize(
                    resolution as usize,
                    &na::Point2::new(remap(upper.0), remap(upper.1)),
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
    pub fn center<N: Real>(&self, resolution: u32) -> na::Unit<na::Vector3<N>> {
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
    pub fn direction<N: Real>(&self, resolution: u32, coords: &na::Point2<N>) -> na::Unit<na::Vector3<N>> {
        let edge_length = Self::edge_length::<N>(resolution);
        let origin_on_face = na::Point2::from(
            na::convert::<_, na::Vector2<N>>(na::Vector2::new(
                self.x as f64,
                self.y as f64,
            )) * edge_length,
        ) - na::convert::<_, na::Vector2<N>>(na::Vector2::new(1.0, 1.0));
        let pos_on_face = origin_on_face + coords.coords * edge_length;
        self.face.direction(&pos_on_face)
    }

    /// Length of the edge in cubemap space of the region covered by these coordinates
    pub fn edge_length<N: Real>(resolution: u32) -> N {
        na::convert::<_, N>(2.0) / na::convert::<_, N>(resolution as f64)
    }

    /// Returns a grid of resolution^2 directions contained by these coords, in scan-line order
    pub fn samples(&self, face_resolution: u32, chunk_resolution: u32) -> SampleIter {
        SampleIter {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
        }
    }

    /// Returns a grid of resolution^2 directions contained by these coords, in scan-line order
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "simd")]
    pub fn samples_ps<S: Simd>(&self, face_resolution: u32, chunk_resolution: u32) -> SampleIterSimd<S> {
        SampleIterSimd {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
            _simd: PhantomData,
        }
    }

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
    pub fn from_vector<N: Real + PartialOrd>(x: &na::Vector3<N>) -> Self {
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
    pub fn coords<N: Real>(x: &na::Vector3<N>) -> (Face, na::Point2<N>) {
        let face = Self::from_vector(x);
        let wrt_face = face.basis().inverse() * x;
        (
            face,
            na::Point2::from(wrt_face.xy() * (na::convert::<_, N>(0.5) / wrt_face.z))
                + na::convert::<_, na::Vector2<N>>(na::Vector2::new(0.5, 0.5)),
        )
    }

    /// Transform from face space (facing +Z) to sphere space (facing the named axis).
    pub fn basis<N: Real>(&self) -> na::Rotation3<N> {
        use self::Face::*;
        let (x, y, z) = match *self {
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
    fn neighbors(&self) -> &'static [(Face, Edge, bool); 4] {
        use self::Face::*;
        match *self {
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
    pub fn direction<N: Real>(&self, coords: &na::Point2<N>) -> na::Unit<na::Vector3<N>> {
        let dir_z = na::Unit::new_normalize(na::Vector3::new(coords.x, coords.y, N::one()));
        self.basis() * dir_z
    }
}

/// Boundary of a `Chunk`
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
            let (x, y) = (self.index % self.chunk_resolution, self.index / self.chunk_resolution);
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

pub struct Path {
    chunk: Chunk,
}

impl Iterator for Path {
    type Item = Chunk;
    fn next(&mut self) -> Option<Chunk> {
        let parent = self.chunk.parent()?;
        Some(mem::replace(&mut self.chunk, parent))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunk.depth as usize, Some(self.chunk.depth as usize))
    }
}

impl ExactSizeIterator for Path {
    fn len(&self) -> usize {
        self.chunk.depth as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
    fn neighbors() {
        use self::Face::*;
        // Every face chunk's neighbors are the face's neighbors
        for face in Face::iter() {
            for (&(neighbor_face, _, _), &neighbor_chunk) in
                face.neighbors().iter().zip(&Chunk::root(face).neighbors())
            {
                assert_eq!(Chunk::root(neighbor_face), neighbor_chunk);
            }
        }

        let chunk = Chunk::root(Face::PZ).children()[0];
        assert_eq!(
            chunk.neighbors()[0],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: NX
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[1],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: NY
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[2],
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 0,
                    face: PZ
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[3],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: PZ
                },
                depth: 1,
            }
        );

        let chunk = Chunk {
            depth: 1,
            coords: Coords {
                x: 1,
                y: 0,
                face: PX,
            },
        };
        assert_eq!(chunk.neighbors()[0b01].neighbors()[0b10], chunk);

        // Every chunk is its neighbor's neighbor
        // Depth 2 to ensure we get all the interesting cases
        for grandparent in Face::iter().map(Chunk::root) {
            for parent in grandparent.children().iter() {
                for chunk in parent.children().iter() {
                    for &neighbor in chunk.neighbors().iter() {
                        assert!(neighbor.neighbors().iter().any(|x| x == chunk));
                    }
                }
            }
        }
    }

    #[test]
    fn parents() {
        for grandparent in Face::iter().map(Chunk::root) {
            for &parent in grandparent.children().iter() {
                assert_eq!(parent.parent(), Some(grandparent));
                for chunk in parent.children().iter() {
                    assert_eq!(chunk.parent(), Some(parent));
                }
            }
        }
    }

    #[test]
    fn sample_count() {
        let chunk = Chunk::root(Face::PZ);
        for i in 0..4 {
            assert_eq!(chunk.samples(i).count(), (i * i) as usize);
        }
    }

    #[test]
    fn sample_sanity() {
        assert_eq!(
            Chunk::root(Face::PZ).samples(1).next().unwrap(),
            na::Vector3::z_axis()
        );

        let chunk = Chunk::root(Face::PZ).children()[1];
        assert!(chunk.samples(2).any(|x| x == na::Vector3::z_axis()));

        // Every face's
        for face in Face::iter() {
            // immediate children
            for chunk in &Chunk::root(face).children() {
                // each have one sample at exactly the center of the face
                assert_eq!(
                    chunk
                        .samples(2)
                        .filter(|&x| x == face.basis() * na::Vector3::z_axis())
                        .count(),
                    1
                );
                // and another at a corner
                let corner = 1.0 / 3.0f32.sqrt();
                assert_eq!(
                    chunk
                        .samples(2)
                        .filter(|&x| x.x.abs() == corner
                            && x.y.abs() == corner
                            && x.z.abs() == corner)
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn sample_lod_boundaries() {
        let chunk = Chunk {
            depth: 10,
            coords: Coords {
                x: 12,
                y: 34,
                face: Face::PZ,
            },
        };
        let children = chunk.children();
        let neighbor = chunk.neighbors()[0];
        assert_eq!(
            children[0]
                .samples(5)
                .map(|child_vert| neighbor
                    .samples(5)
                    .filter(|&neighbor_vert| neighbor_vert == child_vert)
                    .count())
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn origin_sanity() {
        assert_eq!(
            Chunk::root(Face::PZ).origin_on_face::<f32>(),
            na::Vector3::z_axis()
        );
    }

    #[test]
    #[cfg(feature = "simd")]
    fn simd_samples() {
        type S = simdeez::sse2::Sse2;

        let chunk = Chunk::root(Face::PZ);

        let mut samples = chunk.samples(5);
        for [x, y, z] in chunk.samples_ps::<S>(5) {
            for i in 0..S::VF32_WIDTH {
                let reference = if let Some(v) = samples.next() {
                    v
                } else {
                    break;
                };
                assert_eq!(x[i], reference.x);
                assert_eq!(y[i], reference.y);
                assert_eq!(z[i], reference.z);
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
}
