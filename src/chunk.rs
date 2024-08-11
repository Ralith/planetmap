use na::{RealField, SimdRealField};

use crate::cubemap::SampleIterSimd;
use crate::cubemap::{Coords, Edge, Face, SampleIter};

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

    /// Select all `Chunk`s of a certain depth intersecting a cone opening towards `direction` with
    /// half-angle `theta`
    pub fn neighborhood(
        depth: u8,
        direction: na::Vector3<f32>,
        theta: f32,
    ) -> impl Iterator<Item = Self> {
        Coords::neighborhood(2u32.pow(depth.into()), direction, theta)
            .map(move |coords| Self { coords, depth })
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

    /// Iterator over the path from this chunk to the root, inclusive
    pub fn path(self) -> Path {
        Path { chunk: Some(self) }
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
                coords: x[Edge::Nx],
                depth,
            },
            Chunk {
                coords: x[Edge::Ny],
                depth,
            },
            Chunk {
                coords: x[Edge::Px],
                depth,
            },
            Chunk {
                coords: x[Edge::Py],
                depth,
            },
        ]
    }

    /// Length of one of this chunk's edges before mapping onto the sphere
    pub fn edge_length<N: RealField>(&self) -> N {
        Coords::edge_length(self.resolution())
    }

    /// Location of the center of this chunk on the surface of the sphere
    ///
    /// Shorthand for `direction(&[0.5; 2].into())`.
    pub fn origin<N: RealField + Copy>(&self) -> na::Unit<na::Vector3<N>> {
        let size = self.edge_length::<N>();
        let vec = na::Vector2::new(self.coords.x, self.coords.y)
            .map(f64::from)
            .cast::<N>()
            .map(|x| (x + na::convert(0.5)) * size - na::convert(1.0))
            .map(crate::cubemap::warp);
        self.coords.face.basis() * na::Unit::new_normalize(vec.push(na::convert(1.0)))
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    pub fn samples(&self, resolution: u32) -> SampleIter {
        self.coords.samples(self.resolution(), resolution)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    pub fn samples_ps<S>(&self, resolution: u32) -> SampleIterSimd<S>
    where
        S: SimdRealField + Copy,
        S::Element: RealField + Copy,
    {
        self.coords.samples_ps(self.resolution(), resolution)
    }

    /// Compute the direction identified by a [0..1]^2 vector on this chunk
    pub fn direction<N: RealField + Copy>(
        &self,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        self.coords.direction(self.resolution(), coords)
    }

    /// Number of chunks at this depth along a cubemap edge, i.e. 2^depth
    pub fn resolution(&self) -> u32 {
        2u32.pow(self.depth as u32)
    }
}

pub struct Path {
    chunk: Option<Chunk>,
}

impl Iterator for Path {
    type Item = Chunk;
    fn next(&mut self) -> Option<Chunk> {
        let chunk = self.chunk?;
        self.chunk = chunk.parent();
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl ExactSizeIterator for Path {
    fn len(&self) -> usize {
        self.chunk.map_or(0, |x| x.depth as usize + 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use na::SimdValue;

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

        let chunk = Chunk::root(Face::Pz).children()[0];
        assert_eq!(
            chunk.neighbors()[0],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Nx
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
                    face: Ny
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
                    face: Pz
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
                    face: Pz
                },
                depth: 1,
            }
        );

        let chunk = Chunk {
            depth: 1,
            coords: Coords {
                x: 1,
                y: 0,
                face: Px,
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
        let chunk = Chunk::root(Face::Pz);
        for i in 0..4 {
            assert_eq!(chunk.samples(i).count(), (i * i) as usize);
        }
    }

    #[test]
    fn sample_sanity() {
        assert_eq!(
            Chunk::root(Face::Pz).samples(1).next().unwrap(),
            na::Vector3::z_axis()
        );

        let chunk = Chunk::root(Face::Pz).children()[1];
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
                        .filter(|&x| abs_diff_eq!(x.x.abs(), corner)
                            && abs_diff_eq!(x.y.abs(), corner)
                            && abs_diff_eq!(x.z.abs(), corner))
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
                face: Face::Pz,
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
        assert_eq!(Chunk::root(Face::Pz).origin::<f32>(), na::Vector3::z_axis());

        let chunk = Chunk {
            coords: Coords {
                x: 12,
                y: 47,
                face: Face::Px,
            },
            depth: 12,
        };
        assert_eq!(
            chunk,
            Chunk::from_vector(chunk.depth, chunk.origin().as_ref())
        );

        assert_abs_diff_eq!(chunk.origin(), chunk.direction(&na::Point2::new(0.5, 0.5)));
    }

    #[test]
    fn simd_samples() {
        type S = simba::simd::WideF32x4;

        let chunk = Chunk::root(Face::Pz);

        let mut samples = chunk.samples(5);
        for coords in chunk.samples_ps::<S>(5) {
            let [x, y, z] = coords.map(<[f32; S::LANES]>::from);
            for i in 0..S::LANES {
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
    fn path() {
        let leaf = Chunk {
            coords: Coords {
                x: 1,
                y: 1,
                face: Face::Py,
            },
            depth: 1,
        };
        assert_eq!(
            leaf.path().collect::<Vec<_>>(),
            [leaf, leaf.parent().unwrap()]
        );
    }
}
