use std::{fmt, mem};

/// Face of a cube, identified by normal vector
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Face {
    PX,
    NX,
    PY,
    NY,
    PZ,
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

impl Face {
    /// Transform from face space (facing +Z) to sphere space (facing the named axis).
    pub fn basis(&self) -> na::Rotation3<f64> {
        use self::Face::*;
        let (x, y, z) = match *self {
            PX => (na::Vector3::z(), -na::Vector3::y(), na::Vector3::x()),
            NX => (-na::Vector3::z(), -na::Vector3::y(), -na::Vector3::x()),
            PY => (na::Vector3::x(), -na::Vector3::z(), na::Vector3::y()),
            NY => (na::Vector3::x(), na::Vector3::z(), -na::Vector3::y()),
            PZ => (na::Vector3::x(), na::Vector3::y(), na::Vector3::z()),
            NZ => (-na::Vector3::x(), na::Vector3::y(), -na::Vector3::z()),
        };
        na::Rotation3::<f64>::from_matrix_unchecked(na::Matrix3::from_columns(&[x, y, z]))
    }

    pub fn iter() -> impl Iterator<Item = Face> {
        const VALUES: &[Face] = &[Face::PX, Face::NX, Face::PY, Face::NY, Face::PZ, Face::NZ];
        VALUES.iter().cloned()
    }

    /// Neighboring faces wrt. local axes [-x, -y, +x, +y].
    ///
    /// Returns the neighboring face, the edge of that face, and whether the axis shared with that face is parallel or antiparallel.
    ///
    /// Index by `sign << 1 | axis`.
    pub fn neighbors(&self) -> &'static [(Face, Edge, bool); 4] {
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
}

/// Boundary of a chunk
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Edge {
    NX = 0,
    NY = 1,
    PX = 2,
    PY = 3,
}

impl Edge {
    pub fn iter() -> impl Iterator<Item = Edge> {
        [Edge::NX, Edge::NY, Edge::PX, Edge::PY].iter().cloned()
    }
}

impl ::std::ops::Neg for Edge {
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

/// A bounded manifold domain on the surface of a sphere that a square grid of samples may be
/// continuously mapped to
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Chunk {
    pub coords: (u32, u32),
    pub depth: u8,
    pub face: Face,
}

impl Chunk {
    /// The top-level chunk corresponding to a particular cube face
    pub fn root(face: Face) -> Self {
        Self {
            coords: (0, 0),
            depth: 0,
            face,
        }
    }

    /// The smallest chunk that contains this chunk
    pub fn parent(&self) -> Option<Self> {
        let depth = self.depth.checked_sub(1)?;
        Some(Self {
            coords: (self.coords.0 / 2, self.coords.1 / 2),
            depth,
            face: self.face,
        })
    }

    /// Iterator over the path from this chunk to the root.
    pub fn path(self) -> Path {
        Path { chunk: self }
    }

    /// The largest chunks contained by this chunk
    pub fn children(&self) -> [Self; 4] {
        let depth = self.depth + 1;
        let coords = (self.coords.0 * 2, self.coords.1 * 2);
        let face = self.face;
        [
            Chunk {
                coords,
                depth,
                face,
            },
            Chunk {
                coords: (coords.0, coords.1 + 1),
                depth,
                face,
            },
            Chunk {
                coords: (coords.0 + 1, coords.1),
                depth,
                face,
            },
            Chunk {
                coords: (coords.0 + 1, coords.1 + 1),
                depth,
                face,
            },
        ]
    }

    /// Chunks that share an edge with this chunk
    pub fn neighbors(&self) -> [Self; 4] {
        let Chunk {
            face,
            depth,
            coords,
        } = *self;
        let max = 2u32.pow(self.depth as u32) - 1;
        let neighbor_chunk = |face: Face, edge: Edge| {
            let (neighboring_face, neighbor_edge, parallel_axis) = face.neighbors()[edge as usize];
            let other = match edge {
                Edge::NX | Edge::PX => coords.1,
                Edge::NY | Edge::PY => coords.0,
            };
            let other = if parallel_axis { other } else { max - other };
            let coords = match neighbor_edge {
                Edge::NX => (0, other),
                Edge::NY => (other, 0),
                Edge::PX => (max, other),
                Edge::PY => (other, max),
            };
            Chunk {
                face: neighboring_face,
                depth,
                coords,
            }
        };
        [
            if coords.0 == 0 {
                neighbor_chunk(face, Edge::NX)
            } else {
                Chunk {
                    face,
                    depth,
                    coords: (coords.0 - 1, coords.1),
                }
            },
            if coords.1 == 0 {
                neighbor_chunk(face, Edge::NY)
            } else {
                Chunk {
                    face,
                    depth,
                    coords: (coords.0, coords.1 - 1),
                }
            },
            if coords.0 == max {
                neighbor_chunk(face, Edge::PX)
            } else {
                Chunk {
                    face,
                    depth,
                    coords: (coords.0 + 1, coords.1),
                }
            },
            if coords.1 == max {
                neighbor_chunk(face, Edge::PY)
            } else {
                Chunk {
                    face,
                    depth,
                    coords: (coords.0, coords.1 + 1),
                }
            },
        ]
    }

    /// Length of one of this chunk's edges before mapping onto the sphere
    pub fn edge_length(&self) -> f64 {
        2.0 / 2u32.pow(self.depth as u32) as f64
    }

    /// Location of the center of this chunk on the surface of the sphere
    ///
    /// Transform by face.basis() to get world origin
    pub fn origin_on_face(&self) -> na::Unit<na::Vector3<f64>> {
        let size = self.edge_length() as f64;
        let vec = na::Vector3::new(
            (self.coords.0 as f64 + 0.5) * size - 1.0,
            (self.coords.1 as f64 + 0.5) * size - 1.0,
            1.0,
        );
        na::Unit::new_normalize(vec)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    pub fn samples(self, resolution: u32) -> SampleIter {
        SampleIter {
            chunk: self,
            resolution,
            next: (0, 0),
        }
    }
}

#[derive(Debug)]
pub struct SampleIter {
    chunk: Chunk,
    resolution: u32,
    next: (u32, u32),
}

impl SampleIter {
    fn seq(&self) -> usize {
        self.next.0 as usize + self.next.1 as usize * self.resolution as usize
    }
}

impl Iterator for SampleIter {
    type Item = na::Unit<na::Vector3<f64>>;
    fn next(&mut self) -> Option<na::Unit<na::Vector3<f64>>> {
        if self.seq() == self.resolution as usize * self.resolution as usize {
            return None;
        }
        let edge_length = self.chunk.edge_length();
        let origin_on_face =
            na::Vector2::new(self.chunk.coords.0 as f64, self.chunk.coords.1 as f64) * edge_length
                - na::Vector2::new(1.0, 1.0);
        let step = self.chunk.edge_length() / (self.resolution - 1) as f64;
        let pos_on_face =
            origin_on_face + na::Vector2::new(self.next.0 as f64, self.next.1 as f64) * step;
        let dir_z = na::Unit::new_normalize(na::Vector3::new(pos_on_face.x, pos_on_face.y, 1.0));
        let dir = self.chunk.face.basis() * dir_z;
        let x = self.next.0;
        self.next.0 = (x + 1) % self.resolution;
        self.next.1 += (x + 1) / self.resolution;
        Some(dir)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.resolution as usize * self.resolution as usize;
        let remaining = total - self.seq();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SampleIter {
    fn len(&self) -> usize {
        let total = self.resolution as usize * self.resolution as usize;
        total - self.seq()
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
                coords: (0, 1),
                depth: 1,
                face: NX
            }
        );
        assert_eq!(
            chunk.neighbors()[1],
            Chunk {
                coords: (0, 1),
                depth: 1,
                face: NY
            }
        );
        assert_eq!(
            chunk.neighbors()[2],
            Chunk {
                coords: (1, 0),
                depth: 1,
                face: PZ
            }
        );
        assert_eq!(
            chunk.neighbors()[3],
            Chunk {
                coords: (0, 1),
                depth: 1,
                face: PZ
            }
        );

        let chunk = Chunk {
            face: PX,
            depth: 1,
            coords: (1, 0),
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
                let corner = 1.0 / 3.0f64.sqrt();
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
            face: Face::PZ,
            depth: 10,
            coords: (12, 34),
        };
        let children = chunk.children();
        let neighbor = chunk.neighbors()[0];
        assert_eq!(children[0].samples(5).map(|child_vert| neighbor.samples(5).filter(|&neighbor_vert| neighbor_vert == child_vert).count()).sum::<usize>(), 3);
    }

    #[test]
    fn origin_sanity() {
        assert_eq!(
            Chunk::root(Face::PZ).origin_on_face(),
            na::Vector3::z_axis()
        );
    }
}
