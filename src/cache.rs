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

#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum depth of quad-tree traversal
    pub max_depth: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self { max_depth: 12 }
    }
}

impl Config {
    /// Number of slots needed by this config to represent maximum detail for a worst-case viewpoint
    pub fn slots_needed(&self) -> usize {
        chunk::Face::iter()
            .map(Chunk::root)
            .map(|x| self.slots_needed_inner(&x))
            .sum()
    }

    fn slots_needed_inner(&self, chunk: &Chunk) -> usize {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        if chunk.depth == self.max_depth || !needs_subdivision(&chunk, &viewpoint) {
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
    index: FxHashMap<Chunk, u32>,
    config: Config,
}

impl Manager {
    /// Create a manager for a cache of `slots` slots
    pub fn new(slots: usize, config: Config) -> Self {
        Self {
            chunks: Slab::with_capacity(slots),
            index: FxHashMap::with_capacity_and_hasher(slots, Default::default()),
            config,
        }
    }

    /// Compute slots to render for a given set of viewpoints, and to load for improved detail in
    /// future passes.
    ///
    /// Viewpoints should be positioned with regard to the sphere's origin, and scaled such
    /// viewpoints on the surface are 1.0 units from the sphere's origin.
    pub fn update(&mut self, viewpoints: &[na::Point3<f64>]) -> State {
        let mut walker = Walker::with_capacity(self.chunks.capacity());
        walker.walk(&self, viewpoints);

        // Make room for transfers by discarding chunks that we don't currently need.
        let mut available = self.chunks.capacity() - self.chunks.len();
        for idx in walker
            .used
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(idx, used)| if used { None } else { Some(idx) })
        {
            if available >= walker.out.transfer.len() {
                break;
            }
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
        let slot = self.chunks.insert(Slot {
            chunk,
            ready: false,
        }) as u32;
        let old = self.index.insert(chunk, slot);
        debug_assert!(old.is_none(), "a slot has already been allocated for this chunk");
        Some(slot)
    }

    /// Indicate that a previously `allocate`d slot can now be safely read or reused
    pub fn release(&mut self, slot: u32) {
        debug_assert!(!self.chunks[slot as usize].ready, "slot must be allocated to be released");
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
/// regard to an adjacent lower-detail level if discontinuities are undesirable. For example,
/// terrain using heightmapped tiles should weld together a subset of the vertices on an edge shared
/// with a lower-detail chunk.
///
/// Note that increases in LoD are not represented here; it is always the responsibility of the
/// higher-detail chunk to account for neighboring lower-detail chunks.
#[derive(Default)]
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
            // Kick off the loop for each face's quadtree
            self.walk_inner(
                mgr,
                viewpoints,
                ChunkState {
                    chunk,
                    slot,
                    renderable: slot.map_or(false, |idx| mgr.chunks[idx as usize].ready),
                },
            );
        }

        // Compute the LoD delta neighborhood of each rendered chunk
        let rendering = self
            .out
            .render
            .iter()
            .map(|&(chunk, _, _)| chunk)
            .collect::<FxHashSet<_>>();
        for &mut (chunk, ref mut neighborhood, _) in &mut self.out.render {
            use chunk::Edge;
            for (edge, neighbor) in Edge::iter().zip(chunk.neighbors().iter()) {
                // Compute the LoD difference to the rendered neighbor on this edge, if any
                if let Some(neighbor) = neighbor.path().find(|x| rendering.contains(&x)) {
                    neighborhood[edge] = chunk.depth - neighbor.depth;
                }
            }
        }
    }

    /// Walk the quadtree below `chunk`, recording chunks to render and transfer.
    fn walk_inner(&mut self, mgr: &Manager, viewpoints: &[na::Point3<f64>], chunk: ChunkState) {
        // If this chunk is already associated with a cache slot, preserve that slot; otherwise,
        // tell the caller we want it.
        if let Some(idx) = chunk.slot {
            self.used[idx as usize] = true;
        } else {
            self.out.transfer.push(chunk.chunk);
        }

        let subdivide = chunk.chunk.depth < mgr.config.max_depth
            && viewpoints
                .iter()
                .any(|v| needs_subdivision(&chunk.chunk, v));
        if !subdivide {
            if chunk.renderable {
                self.out
                    .render
                    .push((chunk.chunk, Neighborhood::default(), chunk.slot.unwrap()));
            }
            return;
        }

        let children = chunk.chunk.children();
        let child_slots = [
            mgr.get(&children[0]),
            mgr.get(&children[1]),
            mgr.get(&children[2]),
            mgr.get(&children[3]),
        ];
        // The children of this chunk might be rendered if:
        let children_renderable = chunk.renderable // this subtree should be rendered at all, and
            && child_slots                         // every child is already resident in the cache
            .iter()
            .all(|slot| slot.map_or(false, |x| mgr.chunks[x as usize].ready));
        // If this subtree should be rendered and the children can't be rendered, this chunk must be rendered.
        if chunk.renderable && !children_renderable {
            self.out
                .render
                .push((chunk.chunk, Neighborhood::default(), chunk.slot.unwrap()));
        }
        // Recurse into the children
        for (&child, &slot) in children.iter().zip(child_slots.iter()) {
            self.walk_inner(
                mgr,
                viewpoints,
                ChunkState {
                    chunk: child,
                    renderable: children_renderable,
                    slot,
                },
            );
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

    let center = na::Point3::from(chunk.face.basis() * chunk.origin_on_face().into_inner());
    if center.coords.dot(&viewpoint.coords) < 0.0 {
        return false;
    }
    let distance = na::distance(&center, viewpoint);
    // Half-angle of the circular cone from the camera containing an inscribed sphere
    let half_angle = (chunk.edge_length() * 0.5).atan2(distance);
    half_angle >= max_half_angle
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transfer_completeness() {
        let mut mgr = Manager::new(2048, Config::default());
        let state = mgr.update(&[na::Point3::from(na::Vector3::z())]);
        assert_eq!(state.render.len(), 0);
        for transfer in state.transfer {
            let slot = mgr.allocate(transfer).unwrap();
            mgr.release(slot);
        }
        let state = mgr.update(&[na::Point3::from(na::Vector3::z())]);
        assert_eq!(state.transfer.len(), 0);
        assert_ne!(state.render.len(), 0);
    }

    #[test]
    fn slots_needed() {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        for max_depth in 0..12 {
            let config = Config { max_depth };
            let needed = config.slots_needed();
            let mut mgr = Manager::new(2048, config);
            let state = mgr.update(&[viewpoint]);
            assert_eq!(state.transfer.len(), needed);
        }
    }
}
