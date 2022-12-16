use hashbrown::HashMap;
use std::ops::{Index, IndexMut};

use slab::Slab;

use crate::cubemap::{Edge, Face};
use crate::Chunk;

struct Slot {
    chunk: Chunk,
    /// Whether the slot is ready for reading
    ready: bool,
    /// Sequence number of the most recent frame this slot was rendered in
    in_frame: u64,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum depth of quad-tree traversal
    pub max_depth: u8,
    /// Upper bound for `Neighborhood` fields
    ///
    /// Should be set to the base 2 logarithm of the number of quads along the edge of a chunk to
    /// reduce stitching artifacts across extreme LoD boundaries.
    pub max_neighbor_delta: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 12,
            max_neighbor_delta: u8::max_value(),
        }
    }
}

impl Config {
    /// Number of slots needed by this config to represent maximum detail for a worst-case viewpoint
    pub fn slots_needed(&self) -> usize {
        Face::iter()
            .map(Chunk::root)
            .map(|x| self.slots_needed_inner(&x))
            .sum()
    }

    fn slots_needed_inner(&self, chunk: &Chunk) -> usize {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        if chunk.depth == self.max_depth || !needs_subdivision(chunk, &viewpoint) {
            return 1;
        }
        chunk
            .children()
            .iter()
            .map(|x| self.slots_needed_inner(x))
            .sum::<usize>()
            + 1
    }
}

/// Helper for streaming `Chunk`-oriented LoD into a fixed-size cache
pub struct Manager {
    chunks: Slab<Slot>,
    index: HashMap<Chunk, u32>,
    config: Config,
    /// Frame counter used to update `Slot.in_frame`
    frame: u64,
    render: Vec<(Chunk, Neighborhood, u32)>,
}

impl Manager {
    /// Create a manager for a cache of `slots` slots
    pub fn new(slots: usize, config: Config) -> Self {
        Self {
            chunks: Slab::with_capacity(slots),
            index: HashMap::with_capacity(slots),
            config,
            frame: 0,
            render: Vec::with_capacity(slots),
        }
    }

    /// Compute slots to render for a given set of viewpoints, and to load for improved detail in
    /// future passes.
    ///
    /// Viewpoints should be positioned with regard to the sphere's origin, and scaled such
    /// viewpoints on the surface are 1.0 units from the sphere's origin.
    ///
    /// Writes unavailable chunks needed for target quality to `transfer`.
    pub fn update(&mut self, viewpoints: &[na::Point3<f64>], transfer: &mut Vec<Chunk>) {
        transfer.clear();
        self.render.clear();
        self.walk(viewpoints, transfer);

        // Make room for transfers by discarding chunks that we don't currently need.
        let mut available = self.chunks.capacity() - self.chunks.len();
        for idx in 0..self.chunks.capacity() {
            if available >= transfer.len() {
                break;
            }
            if !self.chunks.contains(idx)
                || !self.chunks[idx].ready
                || self.chunks[idx].in_frame == self.frame
            {
                continue;
            }
            let old = self.chunks.remove(idx);
            self.index.remove(&old.chunk);
            available += 1;
        }
        transfer.truncate(available);
        self.frame += 1;
    }

    /// Chunks that can be rendered immediately, with their LoD neighborhood and slot index
    #[inline]
    pub fn renderable(&self) -> &[(Chunk, Neighborhood, u32)] {
        &self.render
    }

    /// Allocate a slot for writing
    ///
    /// Determines where in the cache to transfer chunk data to. Returns `None` if the cache is
    /// full.
    pub fn allocate(&mut self, chunk: Chunk) -> Option<u32> {
        if self.chunks.len() == self.chunks.capacity() {
            return None;
        }
        let slot = self.chunks.insert(Slot {
            chunk,
            ready: false,
            in_frame: 0,
        }) as u32;
        let old = self.index.insert(chunk, slot);
        debug_assert!(
            old.is_none(),
            "a slot has already been allocated for this chunk"
        );
        Some(slot)
    }

    /// Indicate that a previously `allocate`d slot can now be safely read or reused
    pub fn release(&mut self, slot: u32) {
        debug_assert!(
            !self.chunks[slot as usize].ready,
            "slot must be allocated to be released"
        );
        self.chunks[slot as usize].ready = true;
    }

    fn get(&self, chunk: &Chunk) -> Option<u32> {
        self.index.get(chunk).cloned()
    }

    fn walk(&mut self, viewpoints: &[na::Point3<f64>], transfers: &mut Vec<Chunk>) {
        // Gather the set of chunks we can should render and want to transfer
        for chunk in Face::iter().map(Chunk::root) {
            let slot = self.get(&chunk);
            // Kick off the loop for each face's quadtree
            self.walk_inner(
                viewpoints,
                transfers,
                ChunkState {
                    chunk,
                    slot,
                    renderable: slot.map_or(false, |idx| self.chunks[idx as usize].ready),
                },
            );
        }

        // Compute the neighborhood of each rendered chunk.
        for &mut (chunk, ref mut neighborhood, _) in &mut self.render {
            for (edge, neighbor) in Edge::iter().zip(chunk.neighbors().iter()) {
                for x in neighbor.path() {
                    let slot = match self.index.get(&x) {
                        None => continue,
                        Some(&x) => x,
                    };
                    let state = &self.chunks[slot as usize];
                    if state.ready && state.in_frame == self.frame {
                        neighborhood[edge] =
                            (chunk.depth - x.depth).min(self.config.max_neighbor_delta);
                        break;
                    }
                }
            }
        }
    }

    /// Walk the quadtree below `chunk`, recording chunks to render and transfer.
    fn walk_inner(
        &mut self,
        viewpoints: &[na::Point3<f64>],
        transfers: &mut Vec<Chunk>,
        chunk: ChunkState,
    ) {
        // If this chunk is already associated with a cache slot, preserve that slot; otherwise,
        // tell the caller we want it.
        if let Some(idx) = chunk.slot {
            self.chunks[idx as usize].in_frame = self.frame;
        } else {
            transfers.push(chunk.chunk);
        }

        let subdivide = chunk.chunk.depth < self.config.max_depth
            && viewpoints
                .iter()
                .any(|v| needs_subdivision(&chunk.chunk, v));
        if !subdivide {
            if chunk.renderable {
                self.render
                    .push((chunk.chunk, Neighborhood::default(), chunk.slot.unwrap()));
            }
            return;
        }

        let children = chunk.chunk.children();
        let child_slots = [
            self.get(&children[Edge::Nx]),
            self.get(&children[Edge::Ny]),
            self.get(&children[Edge::Px]),
            self.get(&children[Edge::Py]),
        ];
        // The children of this chunk might be rendered if:
        let children_renderable = chunk.renderable // this subtree should be rendered at all, and
            && child_slots                         // every child is already resident in the cache
            .iter()
            .all(|slot| slot.map_or(false, |x| self.chunks[x as usize].ready));
        // If this subtree should be rendered and the children can't be rendered, this chunk must be rendered.
        if chunk.renderable && !children_renderable {
            self.render
                .push((chunk.chunk, Neighborhood::default(), chunk.slot.unwrap()));
        }
        // Recurse into the children
        for (&child, &slot) in children.iter().zip(child_slots.iter()) {
            self.walk_inner(
                viewpoints,
                transfers,
                ChunkState {
                    chunk: child,
                    renderable: children_renderable,
                    slot,
                },
            );
        }
    }
}

/// For each edge of a particular chunk, this represents the number of LoD levels higher that chunk
/// is than its neighbor on that edge.
///
/// Smoothly interpolating across chunk boundaries requires careful attention to these values. In
/// particular, any visualization of chunk data should take care to be continuous at the edge with
/// regard to an adjacent lower-detail level if discontinuities are undesirable. For example,
/// terrain using heightmapped tiles should weld together a subset of the vertices on an edge shared
/// with a lower-detail chunk.
///
/// Note that increases in LoD are not represented here; it is always the responsibility of the
/// higher-detail chunk to account for neighboring lower-detail chunks.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Neighborhood {
    /// Decrease in LoD in the local -X direction
    pub nx: u8,
    /// Decrease in LoD in the local -Y direction
    pub ny: u8,
    /// Decrease in LoD in the local +X direction
    pub px: u8,
    /// Decrease in LoD in the local +Y direction
    pub py: u8,
}

impl Index<Edge> for Neighborhood {
    type Output = u8;
    fn index(&self, edge: Edge) -> &u8 {
        use Edge::*;
        match edge {
            Nx => &self.nx,
            Ny => &self.ny,
            Px => &self.px,
            Py => &self.py,
        }
    }
}

impl IndexMut<Edge> for Neighborhood {
    fn index_mut(&mut self, edge: Edge) -> &mut u8 {
        use Edge::*;
        match edge {
            Nx => &mut self.nx,
            Ny => &mut self.ny,
            Px => &mut self.px,
            Py => &mut self.py,
        }
    }
}

struct ChunkState {
    chunk: Chunk,
    /// Cache slot associated with this chunk, whether or not it's ready
    slot: Option<u32>,
    /// Whether the subtree at this chunk will be rendered
    renderable: bool,
}

fn needs_subdivision(chunk: &Chunk, viewpoint: &na::Point3<f64>) -> bool {
    // Half-angle of the cone whose edges are simultaneously tangent to the edges of a pair of
    // circles inscribed on a chunk at depth D and a neighbor of that chunk at depth D+1. Setting a
    // threshold larger than this leads to LoD deltas greater than 1 across edges.
    let max_half_angle = 1.0f64.atan2(10.0f64.sqrt());

    let center = na::Point3::from(chunk.origin().into_inner());
    if center.coords.dot(&viewpoint.coords) < 0.0 {
        return false;
    }
    let distance = na::distance(&center, viewpoint);
    // Half-angle of the circular cone from the camera containing an inscribed sphere
    let half_angle = (chunk.edge_length::<f64>() * 0.5).atan2(distance);
    half_angle >= max_half_angle
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transfer_completeness() {
        let mut mgr = Manager::new(2048, Config::default());
        let mut transfers = Vec::new();
        mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
        assert_eq!(mgr.renderable().len(), 0);
        for &transfer in &transfers {
            let slot = mgr.allocate(transfer).unwrap();
            mgr.release(slot);
        }
        mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
        assert_eq!(transfers.len(), 0);
        assert_ne!(mgr.renderable().len(), 0);
    }

    #[test]
    fn slots_needed() {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        for max_depth in 0..12 {
            let config = Config {
                max_depth,
                ..Config::default()
            };
            let needed = config.slots_needed();
            let mut mgr = Manager::new(2048, config);
            let mut transfers = Vec::new();
            mgr.update(&[viewpoint], &mut transfers);
            assert_eq!(transfers.len(), needed);
        }
    }

    #[test]
    fn neighborhood() {
        use crate::cubemap::Coords;
        let mut mgr = Manager::new(2048, Config::default());
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        assert_eq!(mgr.renderable().len(), 0);
        // Get +X to LoD 1, +Y to LoD 0
        for &chunk in &[
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 0,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Py,
                },
                depth: 0,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 1,
                    face: Face::Px,
                },
                depth: 1,
            },
        ] {
            assert!(transfers.contains(&chunk));
            let slot = mgr.allocate(chunk).unwrap();
            mgr.release(slot);
        }

        // Validate output
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        assert_eq!(mgr.renderable().len(), 5);
        use std::collections::HashMap;
        let neighbors = mgr
            .renderable()
            .iter()
            .map(|&(chunk, neighbors, _slot)| (chunk, neighbors))
            .collect::<HashMap<_, _>>();
        let neighborhood = *neighbors
            .get(&Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            })
            .unwrap();
        assert_ne!(
            neighborhood,
            Neighborhood {
                nx: 0,
                ny: 0,
                px: 0,
                py: 0
            }
        );
    }

    #[test]
    fn incremental_transfer() {
        let mut mgr = Manager::new(2048, Config::default());
        let viewpoint = na::Point3::from(na::Vector3::new(0.0, 1.0, 0.0));
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        let expected = transfers.len();
        const BATCH_SIZE: usize = 17;
        let mut actual = 0;
        while !transfers.is_empty() {
            for &chunk in transfers.iter().take(BATCH_SIZE) {
                actual += 1;
                let slot = mgr.allocate(chunk).unwrap();
                mgr.release(slot);
            }
            transfers.clear();
            mgr.update(&[viewpoint], &mut transfers);
            assert_eq!(transfers.len(), expected - actual);
        }
    }

    #[test]
    fn subdivision_sanity() {
        let chunk = Chunk {
            depth: 10,
            coords: crate::cubemap::Coords {
                x: 12,
                y: 34,
                face: Face::Px,
            },
        };
        // Verify that a chunk containing the viewpoint gets subdivided
        assert!(needs_subdivision(
            &chunk,
            &na::Point3::from(chunk.direction(&[0.5; 2].into()).into_inner())
        ));
    }
}
