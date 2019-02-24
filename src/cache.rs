use fxhash::{FxHashMap, FxHashSet};
use std::ops::{Index, IndexMut};

use na;
use slab::Slab;

use crate::{chunk, Chunk};

struct Slot {
    chunk: Chunk,
    /// Whether the slot is ready for reading
    ready: bool,
}

/// Helper for streaming `Chunk`-oriented LoD into a fixed-size cache
pub struct Manager {
    chunks: Slab<Slot>,
    index: FxHashMap<Chunk, u32>,
}

impl Manager {
    /// Create a manager for a cache of `capacity` slots
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            chunks: Slab::with_capacity(capacity),
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Compute slots to render for a given point of view, and to load for improved detail
    pub fn update(&mut self, viewpoints: &[na::Point3<f64>]) -> State {
        let mut walker = Walker::with_capacity(self.chunks.capacity());
        walker.walk(&self, viewpoints);
        
        // Make room for transfers
        let mut available = self.chunks.capacity() - self.chunks.len();
        for idx in walker.used.iter().cloned().enumerate()
            .filter_map(|(idx, used)| if used { None } else { Some(idx) })
        {
            if available >= walker.out.transfer.len() { break; }
            if self.chunks.contains(idx as usize) && self.chunks[idx as usize].ready {
                let old = self.chunks.remove(idx as usize);
                self.index.remove(&old.chunk);
                available += 1;
            }
        }
        walker.out.transfer.truncate(available);

        walker.out
    }

    /// Allocate a slot for writing
    ///
    /// Determines where in the cache to transfer chunk data to. Returns `None` if the cache is
    /// full.
    pub fn allocate(&mut self, chunk: Chunk) -> Option<u32> {
        if self.chunks.len() == self.chunks.capacity() {
            return None;
        }
        let slot = self.chunks.insert(Slot { chunk, ready: false }) as u32;
        self.index.insert(chunk, slot);
        Some(slot)
    }

    /// Indicate that a previously `allocate`d slot can now be safely read or reused
    pub fn release(&mut self, slot: u32) {
        self.chunks[slot as usize].ready = true;
    }

    fn get(&self, chunk: &Chunk) -> Option<u32> {
        self.index.get(chunk).cloned()
    }
}

/// Result of a `Manager::update` operation
pub struct State {
    /// Chunks that can be rendered immediately, with their LoD neighborhood and slot index
    pub render: Vec<(Chunk, Neighborhood, u32)>,
    /// Chunks that should be loaded to improve the detail supplied by the `render` set in a future
    /// `Manager::update` call for a similar `viewpoint`.
    pub transfer: Vec<Chunk>,
}

/// For each edge of a particular chunk, this represents the number of LoD levels higher that chunk
/// is than its neighbor on that edge.
///
/// Smoothly interpolating across chunk boundaries requires careful attention to these values. In
/// particular, any visualization of chunk data should take care to be continuous at the edge with
/// regard to an adjacent lower-detail level if discontinuities are undesirable.
///
/// Note that increases in LoD are not represented here; it is always the responsibility of the
/// higher-detail chunk to account for neighboring lower-detail chunks.
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

impl Index<chunk::Edge> for Neighborhood {
    type Output = u8;
    fn index(&self, edge: chunk::Edge) -> &u8 {
        use chunk::Edge::*;
        match edge {
            NX => &self.nx,
            NY => &self.ny,
            PX => &self.px,
            PY => &self.py,
        }
    }
}

impl IndexMut<chunk::Edge> for Neighborhood {
    fn index_mut(&mut self, edge: chunk::Edge) -> &mut u8 {
        use chunk::Edge::*;
        match edge {
            NX => &mut self.nx,
            NY => &mut self.ny,
            PX => &mut self.px,
            PY => &mut self.py,
        }
    }
}

struct Walker {
    out: State,
    used: Vec<bool>,
}

impl Walker {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            out: State {
                render: Vec::new(),
                transfer: Vec::new(),
            },
            used: vec![false; capacity],
        }
    }

    fn walk(&mut self, mgr: &Manager, viewpoints: &[na::Point3<f64>]) {
        // Gather the set of chunks we can should render and want to transfer
        for chunk in chunk::Face::iter().map(Chunk::root) {
            let slot = mgr.get(&chunk);
            self.walk_inner(mgr, viewpoints,
                            ChunkState {
                                chunk,
                                slot,
                                subdivide: viewpoints.iter().any(|v| needs_subdivision(&chunk, v)),
                                renderable: slot.map_or(false, |idx| mgr.chunks[idx as usize].ready),
                            }
            );
        }

        // Compute the LoD delta neighborhood of each rendered chunk
        let rendering = self.out.render.iter().map(|&(chunk, _, _)| chunk).collect::<FxHashSet<_>>();
        for &mut (chunk, ref mut neighborhood, _) in &mut self.out.render {
            use chunk::Edge;
            for (edge, neighbor) in Edge::iter().zip(chunk.neighbors().iter()) {
                if let Some(neighbor) = neighbor.path().find(|x| rendering.contains(&x)) {
                    neighborhood[edge] = chunk.depth - neighbor.depth;
                }
            }
        }
    }

    fn walk_inner(
        &mut self,
        mgr: &Manager,
        viewpoints: &[na::Point3<f64>],
        chunk: ChunkState,
    ) {
        if let Some(idx) = chunk.slot {
            self.used[idx as usize] = true;
        } else {
            self.out.transfer.push(chunk.chunk);
        }

        let children = chunk.chunk.children();
        let child_slots = [
            mgr.get(&children[0]),
            mgr.get(&children[1]),
            mgr.get(&children[2]),
            mgr.get(&children[3]),
        ];
        let children_renderable = chunk.renderable
            && chunk.subdivide
            && child_slots
            .iter()
            .all(|slot| slot.map_or(false, |x| mgr.chunks[x as usize].ready));
        if chunk.renderable && !children_renderable {
            let neighborhood = Neighborhood {
                nx: 0,
                ny: 0,
                px: 0,
                py: 0,
            }; // Filled in later
            self.out
                .render
                .push((chunk.chunk, neighborhood, chunk.slot.unwrap()));
        }
        const MAX_DEPTH: u8 = 12;
        if chunk.subdivide {
            for (&child, &slot) in children
                .iter()
                .zip(child_slots.iter())
            {
                self.walk_inner(mgr, viewpoints, ChunkState {
                    chunk: child,
                    renderable: children_renderable,
                    slot,
                    subdivide: child.depth < MAX_DEPTH
                        && viewpoints.iter().any(|v| needs_subdivision(&child, v)),
                });
            )
        }
    }
}

struct ChunkState {
    chunk: Chunk,
    /// Whether this chunk should be subdivided
    subdivide: bool,
    /// Cache slot associated with this chunk, whether or not it's ready
    slot: Option<u32>,
    /// True iff this chunk or its children will be rendered
    renderable: bool,
}

fn needs_subdivision(chunk: &Chunk, viewpoint: &na::Point3<f64>) -> bool {
    // Angle of a cone that intersects inscribed circles on a chunk of depth D and a neighbor of
    // depth D+1. Setting a threshold larger than this leads to LoD deltas greater than 1 across
    // edges.
    let max_half_angle = 1.0f64.atan2(10.0f64.sqrt());

    let center = na::Point3::from(chunk.face.basis() * chunk.origin_on_face().into_inner());
    let distance = na::distance(&center, viewpoint);
    // Half-angle of the circular cone from the camera containing an inscribed sphere
    let half_angle = (chunk.edge_length() * 0.5).atan2(distance);
    half_angle >= max_half_angle
}
