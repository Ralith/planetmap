//! Collision detection for radial heightmaps
//!
//! Implement [`Terrain`] for your heightmap, then create colliders using it with a [`Planet`]. A
//! query dispatcher that handles both planets and standard shapes can be constructed with
//! `PlanetDispatcher.chain(DefaultQueryDispatcher)`.

use std::sync::{Arc, Mutex};

use hashbrown::hash_map;
use hashbrown::HashMap;
use parry3d_f64::{
    bounding_volume::{Aabb, BoundingSphere, BoundingVolume},
    mass_properties::MassProperties,
    math::{Isometry, Point, Real, Vector},
    query::{
        visitors::BoundingVolumeIntersectionsVisitor, ClosestPoints, Contact, ContactManifold,
        ContactManifoldsWorkspace, DefaultQueryDispatcher, NonlinearRigidMotion,
        PersistentQueryDispatcher, PointProjection, PointQuery, QueryDispatcher, Ray, RayCast,
        RayIntersection, TypedWorkspaceData, Unsupported, WorkspaceData, TOI,
    },
    shape::{FeatureId, HalfSpace, Shape, ShapeType, SimdCompositeShape, Triangle, TypedShape},
    utils::IsometryOpt,
};

use crate::{
    cubemap::{Coords, Edge},
    lru_slab::{LruSlab, SlotId},
};

/// Height data source for `Planet`
pub trait Terrain: Send + Sync + 'static {
    /// Generate a `resolution * resolution` grid of heights wrt. sea level
    fn sample(&self, coords: &Coords, out: &mut [f32]);
    /// Number of blocks of samples (chunks) along the edge of a cubemap face
    fn face_resolution(&self) -> u32;
    /// Number of samples along the edge of a single chunk. `sample` will be supplied buffers sized
    /// based for this. Must be at least 2.
    fn chunk_resolution(&self) -> u32;
    /// The maximum value that will ever be written by `sample`
    fn max_height(&self) -> f32;
    /// The minimum value that will ever be written by `sample`
    fn min_height(&self) -> f32;
}

/// Trivial `Terrain` impl
#[derive(Debug, Copy, Clone)]
pub struct FlatTerrain {
    face_resolution: u32,
    chunk_resolution: u32,
}

impl FlatTerrain {
    pub fn new(face_resolution: u32, chunk_resolution: u32) -> Self {
        Self {
            face_resolution,
            chunk_resolution,
        }
    }
}

impl Terrain for FlatTerrain {
    fn face_resolution(&self) -> u32 {
        self.face_resolution
    }

    fn chunk_resolution(&self) -> u32 {
        self.chunk_resolution
    }

    fn max_height(&self) -> f32 {
        0.0
    }
    fn min_height(&self) -> f32 {
        0.0
    }

    fn sample(&self, _: &Coords, out: &mut [f32]) {
        for x in out {
            *x = 0.0;
        }
    }
}

/// A fixed-resolution partially-resident radial heightmap
///
/// Generates height data on-demand via `Terrain`, preserving it in a fixed-size LRU cache.
pub struct Planet {
    terrain: Arc<dyn Terrain>,
    radius: f64,
    chunk_resolution: u32,
    // Future work: could preallocate an arena for height samples
    cache: Mutex<Cache>,
}

impl Planet {
    /// Construct a new collision shape for a radial heightmap defined by `terrain`
    ///
    /// `terrain` - source of height samples
    /// `cache_size` - maximum number of chunks of height data to keep in memory
    /// `radius` - distance from origin of points with height 0
    pub fn new(terrain: Arc<dyn Terrain>, cache_size: u32, radius: f64) -> Self {
        Self {
            chunk_resolution: terrain.chunk_resolution(),
            terrain,
            radius,
            cache: Mutex::new(Cache::new(cache_size)),
        }
    }

    fn max_radius(&self) -> f64 {
        self.radius + self.terrain.max_height() as f64
    }
    fn min_radius(&self) -> f64 {
        self.radius + self.terrain.min_height() as f64
    }

    fn sample(&self, coords: &Coords) -> Box<[f32]> {
        let mut samples =
            vec![0.0; self.chunk_resolution as usize * self.chunk_resolution as usize]
                .into_boxed_slice();
        self.terrain.sample(coords, &mut samples[..]);
        samples
    }

    fn feature_id(&self, slot: SlotId, triangle: u32) -> u32 {
        slot.0 * self.chunk_resolution * self.chunk_resolution + triangle
    }

    /// Applies the function `f` to all the triangles intersecting the given sphere. Exits early on
    /// `false` return.
    pub fn map_elements_in_local_sphere(
        &self,
        bounds: &BoundingSphere,
        aabb: &Aabb,
        mut f: impl FnMut(&Coords, SlotId, u32, &Triangle) -> bool,
    ) {
        let dir = bounds.center().coords;
        let distance = dir.norm();
        let cache = &mut *self.cache.lock().unwrap();
        // Iterate over each overlapping chunk
        'outer: for chunk_coords in Coords::neighborhood(
            self.terrain.face_resolution(),
            dir.cast(),
            bounds.radius().atan2(distance) as f32,
        ) {
            let (slot, data) = cache.get(self, &chunk_coords);
            if self.radius + data.max as f64 + bounds.radius() < distance
                || self.radius + data.min as f64 - bounds.radius() > distance
            {
                // Short-circuit if `other` is way above or below this chunk
                continue;
            }
            let patch = Patch::new(&chunk_coords, self.terrain.face_resolution());
            for quad in patch.quads_within(aabb, self.chunk_resolution) {
                for (index, triangle) in
                    quad.triangles(self.radius, self.chunk_resolution, &data.samples)
                {
                    if !f(&chunk_coords, slot, index, &triangle) {
                        break 'outer;
                    }
                }
            }
        }
    }
}

impl Clone for Planet {
    fn clone(&self) -> Self {
        Self {
            terrain: self.terrain.clone(),
            cache: Mutex::new(self.cache.lock().unwrap().clone()),
            ..*self
        }
    }
}

impl RayCast for Planet {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_toi: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        // Find the chunk containing the ray origin
        let mut chunk =
            Coords::from_vector(self.terrain.face_resolution(), &ray.origin.coords.cast());
        let mut patch = Patch::new(&chunk, self.terrain.face_resolution());
        let mut ray = *ray;

        // Walk along the ray until we hit something
        let cache = &mut *self.cache.lock().unwrap();
        loop {
            let (slot, data) = cache.get(self, &chunk);
            // FIXME: Rays can sometimes slip between bounding planes and the outer edge of a
            // quad. To avoid this, we should extend the quad to form a narrow skirt around the
            // chunk, or check neighboring chunks when very close to a boundary.

            let mut maybe_hit = None;
            let edge = walk_patch(self.chunk_resolution - 1, &patch, &ray, max_toi, |quad| {
                let tris = quad
                    .displace(self.radius, self.chunk_resolution, &data.samples)
                    .triangles();
                let Some((tri, mut hit)) = tris.into_iter()
                    .enumerate()
                    .filter_map(|(i, t)| Some((i, t.cast_local_ray_and_get_normal(&ray, max_toi, solid)?)))
                    .min_by(|a, b| a.1.toi.total_cmp(&b.1.toi))
                else {
                    return true;
                };
                let quad_index = quad.position.y * self.chunk_resolution + quad.position.x;
                let index = (quad_index << 1) | tri as u32;
                hit.feature = FeatureId::Face(self.feature_id(slot, index));
                maybe_hit = Some(hit);
                false
            });

            if let Some(hit) = maybe_hit {
                return Some(hit);
            }

            match edge {
                None => return None,
                Some((edge, toi)) => {
                    chunk = chunk.neighbors(self.terrain.face_resolution())[edge];
                    patch = Patch::new(&chunk, self.terrain.face_resolution());
                    ray.origin += ray.dir * toi;
                }
            }
        }
    }
}

/// `quad` is row-major vectors from the origin
fn raycast_quad_edges(
    ray: &Ray,
    [a, b, c, d]: &[na::Vector3<f64>; 4],
    max_toi: f64,
) -> Option<(Edge, f64)> {
    let edges: [(Edge, [&na::Vector3<f64>; 2]); 4] = [
        (Edge::Nx, [c, a]),
        (Edge::Ny, [a, b]),
        (Edge::Px, [b, d]),
        (Edge::Py, [d, c]),
    ];

    let mut closest = None;
    for &(edge, [v1, v2]) in edges.iter() {
        // Construct inward-facing edge planes
        let plane = HalfSpace {
            normal: na::Unit::new_normalize(v1.cross(v2)),
        };
        // Eliminate planes behind the ray
        if plane.normal.as_ref().dot(&ray.dir) >= 0.0 {
            continue;
        }
        if let Some(hit) = plane.cast_local_ray(ray, max_toi, true) {
            closest = Some(match closest {
                None => (edge, hit),
                Some((_, toi)) if hit < toi => (edge, hit),
                Some(x) => x,
            });
        }
    }
    closest
}

impl PointQuery for Planet {
    fn project_local_point(&self, pt: &Point<Real>, solid: bool) -> PointProjection {
        if solid && pt.coords.norm_squared() < self.min_radius() * self.min_radius() {
            return PointProjection {
                is_inside: true,
                point: *pt,
            };
        }
        // TODO: Handle `solid` near the surface
        self.project_local_point_and_get_feature(pt).0
    }

    fn project_local_point_and_get_feature(
        &self,
        pt: &Point<Real>,
    ) -> (PointProjection, FeatureId) {
        // TODO: Optimize/fix this by projecting `pt` onto the cubemap, then scanning *outward* from
        // the quad containing the projected point until all remaining triangles must be further
        // than the closest triangle found so far, regardless of height
        let coords = Coords::from_vector(self.terrain.face_resolution(), &pt.coords.cast());
        let distance2 = |x: &na::Point3<f64>| na::distance_squared(x, pt);
        let cache = &mut *self.cache.lock().unwrap();
        let (slot, data) = cache.get(self, &coords);
        let patch = Patch::new(&coords, self.terrain.face_resolution());
        let (idx, nearest) = patch
            .triangles(self.radius, self.chunk_resolution, &data.samples)
            .map(|(i, tri)| (i, tri.project_local_point(pt, false)))
            .min_by(|(_, x), (_, y)| {
                distance2(&x.point)
                    .partial_cmp(&distance2(&y.point))
                    .unwrap()
            })
            .unwrap();
        // TODO: Check neighborhood, so we don't miss as many cliff faces
        (nearest, FeatureId::Face(self.feature_id(slot, idx)))
    }
}

impl Shape for Planet {
    fn compute_local_aabb(&self) -> Aabb {
        Aabb::from_half_extents(Point::origin(), Vector::repeat(self.max_radius()))
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        BoundingSphere::new(Point::origin(), self.max_radius())
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        parry3d_f64::shape::Ball {
            radius: self.radius,
        }
        .mass_properties(density)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(0)
    }

    fn ccd_thickness(&self) -> Real {
        0.0
    }

    fn ccd_angular_thickness(&self) -> Real {
        0.0
    }

    fn clone_box(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct Cache {
    slots: LruSlab<ChunkData>,
    index: HashMap<Coords, SlotId>,
}

impl Cache {
    pub fn new(capacity: u32) -> Self {
        Self {
            slots: LruSlab::with_capacity(capacity),
            index: HashMap::with_capacity(capacity as usize),
        }
    }

    pub fn get(&mut self, planet: &Planet, coords: &Coords) -> (SlotId, &ChunkData) {
        let (slot, old) = match self.index.entry(*coords) {
            hash_map::Entry::Occupied(e) => (*e.get(), None),
            hash_map::Entry::Vacant(e) => {
                let old = if self.slots.len() == self.slots.capacity() {
                    let lru = self.slots.lru().unwrap();
                    Some(self.slots.remove(lru).coords)
                } else {
                    None
                };
                let slot = self
                    .slots
                    .insert(ChunkData::new(*coords, planet.sample(coords)));
                e.insert(slot);
                (slot, old)
            }
        };
        if let Some(old) = old {
            self.index.remove(&old);
        }
        (slot, self.slots.get_mut(slot))
    }
}

#[derive(Debug, Clone)]
struct ChunkData {
    samples: Box<[f32]>,
    coords: Coords,
    min: f32,
    max: f32,
}

impl ChunkData {
    fn new(coords: Coords, samples: Box<[f32]>) -> Self {
        let mut iter = samples.iter().cloned();
        let first = iter.next().expect("empty sample array");
        let mut min = first;
        let mut max = first;
        for sample in iter {
            if sample < min {
                min = sample;
            } else if sample > max {
                max = sample;
            }
        }
        Self {
            samples,
            coords,
            min,
            max,
        }
    }
}

/// A [`PersistentQueryDispatcher`] that handles `Planet` shapes
pub struct PlanetDispatcher;

// TODO: Fill these in
impl QueryDispatcher for PlanetDispatcher {
    fn intersection_test(
        &self,
        pos12: &Isometry<Real>,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<bool, Unsupported> {
        if let Some(p1) = g1.downcast_ref::<Planet>() {
            return Ok(intersects(pos12, p1, g2));
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            return Ok(intersects(&pos12.inverse(), p2, g1));
        }
        Err(Unsupported)
    }

    fn distance(
        &self,
        _pos12: &Isometry<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
    ) -> Result<Real, Unsupported> {
        Err(Unsupported)
    }

    fn contact(
        &self,
        _pos12: &Isometry<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        Err(Unsupported)
    }

    fn closest_points(
        &self,
        _pos12: &Isometry<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        Err(Unsupported)
    }

    fn time_of_impact(
        &self,
        pos12: &Isometry<Real>,
        vel12: &Vector<Real>,
        g1: &dyn Shape,
        g2: &dyn Shape,
        max_toi: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<TOI>, Unsupported> {
        if let Some(p1) = g1.downcast_ref::<Planet>() {
            return Ok(compute_toi(
                pos12,
                vel12,
                p1,
                g2,
                max_toi,
                stop_at_penetration,
                false,
            ));
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            return Ok(compute_toi(
                &pos12.inverse(),
                &-vel12,
                p2,
                g1,
                max_toi,
                stop_at_penetration,
                true,
            ));
        }
        Err(Unsupported)
    }

    fn nonlinear_time_of_impact(
        &self,
        motion1: &NonlinearRigidMotion,
        g1: &dyn Shape,
        motion2: &NonlinearRigidMotion,
        g2: &dyn Shape,
        start_time: Real,
        end_time: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<TOI>, Unsupported> {
        if let Some(p1) = g1.downcast_ref::<Planet>() {
            return Ok(compute_nonlinear_toi(
                motion1,
                p1,
                motion2,
                g2,
                start_time,
                end_time,
                stop_at_penetration,
                false,
            ));
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            return Ok(compute_nonlinear_toi(
                motion2,
                p2,
                motion1,
                g1,
                start_time,
                end_time,
                stop_at_penetration,
                false,
            ));
        }
        Err(Unsupported)
    }
}

fn intersects(pos12: &Isometry<Real>, planet: &Planet, other: &dyn Shape) -> bool {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    let bounds = other.compute_bounding_sphere(pos12);
    let aabb = other.compute_aabb(pos12);
    let mut intersects = false;
    planet.map_elements_in_local_sphere(&bounds, &aabb, |_, _, _, triangle| {
        intersects = dispatcher
            .intersection_test(pos12, triangle, other)
            .unwrap_or(false);
        !intersects
    });
    intersects
}

fn compute_toi(
    pos12: &Isometry<Real>,
    vel12: &Vector<Real>,
    planet: &Planet,
    other: &dyn Shape,
    max_toi: Real,
    stop_at_penetration: bool,
    flipped: bool,
) -> Option<TOI> {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    // TODO: Raycast vs. minkowski sum of chunk bounds and bounding sphere?
    let aabb = {
        let start = other.compute_aabb(pos12);
        let end = start.transform_by(&Isometry::from_parts((max_toi * vel12).into(), na::one()));
        start.merged(&end)
    };
    let mut closest = None::<TOI>;
    planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |_, _, _, triangle| {
        let impact = if flipped {
            dispatcher.time_of_impact(
                &pos12.inverse(),
                &-vel12,
                other,
                triangle,
                max_toi,
                stop_at_penetration,
            )
        } else {
            dispatcher.time_of_impact(pos12, vel12, triangle, other, max_toi, stop_at_penetration)
        };
        if let Ok(Some(impact)) = impact {
            closest = Some(match closest {
                None => impact,
                Some(x) if impact.toi < x.toi => impact,
                Some(x) => x,
            });
        }
        true
    });
    closest
}

#[allow(clippy::too_many_arguments)] // that's just what it takes
fn compute_nonlinear_toi(
    motion_planet: &NonlinearRigidMotion,
    planet: &Planet,
    motion_other: &NonlinearRigidMotion,
    other: &dyn Shape,
    start_time: Real,
    end_time: Real,
    stop_at_penetration: bool,
    flipped: bool,
) -> Option<TOI> {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    // TODO: Select chunks/triangles more conservatively, as discussed in compute_toi
    let aabb = {
        let start_pos = motion_planet.position_at_time(start_time).inverse()
            * motion_other.position_at_time(start_time);
        let end_pos = motion_planet.position_at_time(end_time).inverse()
            * motion_other.position_at_time(end_time);
        let start = other.compute_aabb(&start_pos);
        let end = other.compute_aabb(&end_pos);
        start.merged(&end)
    };
    let mut closest = None::<TOI>;
    planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |_, _, _, triangle| {
        let impact = if flipped {
            dispatcher.nonlinear_time_of_impact(
                motion_other,
                other,
                motion_planet,
                triangle,
                start_time,
                end_time,
                stop_at_penetration,
            )
        } else {
            dispatcher.nonlinear_time_of_impact(
                motion_planet,
                triangle,
                motion_other,
                other,
                start_time,
                end_time,
                stop_at_penetration,
            )
        };
        if let Ok(Some(impact)) = impact {
            closest = Some(match closest {
                None => impact,
                Some(x) if impact.toi < x.toi => impact,
                Some(x) => x,
            });
        }
        true
    });
    closest
}

impl<ManifoldData, ContactData> PersistentQueryDispatcher<ManifoldData, ContactData>
    for PlanetDispatcher
where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    fn contact_manifolds(
        &self,
        pos12: &Isometry<Real>,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        if let Some(p1) = g1.downcast_ref::<Planet>() {
            if let Some(composite) = g2.as_composite_shape() {
                compute_manifolds_vs_composite(
                    pos12,
                    &pos12.inverse(),
                    p1,
                    composite,
                    prediction,
                    manifolds,
                    workspace,
                    false,
                );
            } else {
                compute_manifolds(pos12, p1, g2, prediction, manifolds, workspace, false);
            }
            return Ok(());
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            if let Some(composite) = g2.as_composite_shape() {
                compute_manifolds_vs_composite(
                    &pos12.inverse(),
                    pos12,
                    p2,
                    composite,
                    prediction,
                    manifolds,
                    workspace,
                    true,
                );
            } else {
                compute_manifolds(
                    &pos12.inverse(),
                    p2,
                    g1,
                    prediction,
                    manifolds,
                    workspace,
                    true,
                );
            }
            return Ok(());
        }
        Err(Unsupported)
    }

    fn contact_manifold_convex_convex(
        &self,
        _pos12: &Isometry<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _prediction: Real,
        _manifold: &mut ContactManifold<ManifoldData, ContactData>,
    ) -> Result<(), Unsupported> {
        // Planets aren't guaranteed to be convex, so we have no cases to handle here
        Err(Unsupported)
    }
}

fn compute_manifolds<ManifoldData, ContactData>(
    pos12: &Isometry<Real>,
    planet: &Planet,
    other: &dyn Shape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    let workspace = workspace
        .get_or_insert_with(|| ContactManifoldsWorkspace(Box::<Workspace>::default()))
        .0
        .downcast_mut::<Workspace>()
        .unwrap();
    let dispatcher = DefaultQueryDispatcher; // TODO after https://github.com/dimforge/parry/issues/8

    workspace.phase ^= true;
    let phase = workspace.phase;

    let bounds = other.compute_bounding_sphere(pos12).loosened(prediction);
    let aabb = other.compute_aabb(pos12).loosened(prediction);
    let mut old_manifolds = std::mem::take(manifolds);
    planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, slot, index, triangle| {
        let tri_state = match workspace.state.entry((coords, index)) {
            hash_map::Entry::Occupied(e) => {
                let tri_state = e.into_mut();

                let manifold = old_manifolds[tri_state.manifold_index].take();
                tri_state.manifold_index = manifolds.len();
                tri_state.phase = phase;
                manifolds.push(manifold);

                tri_state
            }
            hash_map::Entry::Vacant(e) => {
                let tri_state = TriangleState {
                    manifold_index: manifolds.len(),
                    phase,
                };

                let id = planet.feature_id(slot, index);
                let (id1, id2) = if flipped { (0, id) } else { (id, 0) };
                manifolds.push(ContactManifold::with_data(
                    id1,
                    id2,
                    ManifoldData::default(),
                ));

                e.insert(tri_state)
            }
        };

        let manifold = &mut manifolds[tri_state.manifold_index];

        // TODO: Nonconvex, postprocess contact `fid`s once parry's feature ID story is worked out
        if flipped {
            let _ = dispatcher.contact_manifold_convex_convex(
                &pos12.inverse(),
                other,
                triangle,
                prediction,
                manifold,
            );
        } else {
            let _ = dispatcher
                .contact_manifold_convex_convex(pos12, triangle, other, prediction, manifold);
        }
        true
    });

    workspace.state.retain(|_, x| x.phase == phase);
}

/// Narrow-phase collision detection state for `Planet`
#[derive(Default, Clone)]
pub struct Workspace {
    state: HashMap<(Coords, u32), TriangleState>,
    phase: bool,
}

impl WorkspaceData for Workspace {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData {
        TypedWorkspaceData::Custom(0)
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct TriangleState {
    manifold_index: usize,
    phase: bool,
}

#[allow(clippy::too_many_arguments)] // that's just what it takes
fn compute_manifolds_vs_composite<ManifoldData, ContactData>(
    pos12: &Isometry<Real>,
    pos21: &Isometry<Real>,
    planet: &Planet,
    other: &dyn SimdCompositeShape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    let workspace = workspace
        .get_or_insert_with(|| ContactManifoldsWorkspace(Box::<WorkspaceVsComposite>::default()))
        .0
        .downcast_mut::<WorkspaceVsComposite>()
        .unwrap();
    let dispatcher = DefaultQueryDispatcher; // TODO after https://github.com/dimforge/parry/issues/8

    workspace.phase ^= true;
    let phase = workspace.phase;

    let bvh = other.qbvh();

    let bounds = bvh
        .root_aabb()
        .bounding_sphere()
        .transform_by(pos12)
        .loosened(prediction);
    let aabb = bvh.root_aabb().transform_by(pos12).loosened(prediction);
    let mut old_manifolds = std::mem::take(manifolds);
    planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, slot, index, triangle| {
        let tri_aabb = triangle.compute_aabb(pos21).loosened(prediction);

        let mut visit = |&composite_subshape: &u32| {
            other.map_part_at(
                composite_subshape,
                &mut |composite_part_pos, composite_part_shape| {
                    let key = CompositeKey {
                        chunk_coords: coords,
                        triangle: index,
                        composite_subshape,
                    };
                    // TODO: Dedup wrt. convex case
                    let tri_state = match workspace.state.entry(key) {
                        hash_map::Entry::Occupied(e) => {
                            let tri_state = e.into_mut();

                            let manifold = old_manifolds[tri_state.manifold_index].take();
                            tri_state.manifold_index = manifolds.len();
                            tri_state.phase = phase;
                            manifolds.push(manifold);

                            tri_state
                        }
                        hash_map::Entry::Vacant(e) => {
                            let mut manifold = ContactManifold::new();
                            let id = planet.feature_id(slot, index);
                            if flipped {
                                manifold.subshape1 = composite_subshape;
                                manifold.subshape2 = id;
                                manifold.subshape_pos1 = composite_part_pos.copied();
                            } else {
                                manifold.subshape1 = id;
                                manifold.subshape2 = composite_subshape;
                                manifold.subshape_pos2 = composite_part_pos.copied();
                            };

                            let tri_state = TriangleState {
                                manifold_index: manifolds.len(),
                                phase,
                            };
                            manifolds.push(manifold);
                            e.insert(tri_state)
                        }
                    };

                    let manifold = &mut manifolds[tri_state.manifold_index];

                    if flipped {
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &composite_part_pos.inv_mul(pos21),
                            composite_part_shape,
                            triangle,
                            prediction,
                            manifold,
                        );
                    } else {
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &composite_part_pos.prepend_to(pos12),
                            triangle,
                            composite_part_shape,
                            prediction,
                            manifold,
                        );
                    }
                },
            );
            true
        };
        let mut visitor = BoundingVolumeIntersectionsVisitor::new(&tri_aabb, &mut visit);
        bvh.traverse_depth_first(&mut visitor);

        true
    });

    workspace.state.retain(|_, x| x.phase == phase);
}

/// Narrow-phase collision detection state for `Planet`
#[derive(Default, Clone)]
pub struct WorkspaceVsComposite {
    state: HashMap<CompositeKey, TriangleState>,
    phase: bool,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct CompositeKey {
    chunk_coords: Coords,
    triangle: u32,
    composite_subshape: u32,
}

impl WorkspaceData for WorkspaceVsComposite {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData {
        TypedWorkspaceData::Custom(0)
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}

/// Quad defined by a chunk pre-displacement
///
/// Generally neither flat nor square.
#[derive(Copy, Clone, Debug)]
struct Patch {
    // (0, 0) a--b
    //        |\ |
    //        | \|
    //        c--d (1,1)
    a: na::Vector3<f64>,
    b: na::Vector3<f64>,
    c: na::Vector3<f64>,
    d: na::Vector3<f64>,
}

impl Patch {
    pub fn new(coords: &Coords, face_resolution: u32) -> Self {
        Self {
            a: coords
                .direction(face_resolution, &[0.0, 0.0].into())
                .into_inner(),
            b: coords
                .direction(face_resolution, &[1.0, 0.0].into())
                .into_inner(),
            c: coords
                .direction(face_resolution, &[0.0, 1.0].into())
                .into_inner(),
            d: coords
                .direction(face_resolution, &[1.0, 1.0].into())
                .into_inner(),
        }
    }

    /// Map a point from patch space to a direction in sphere space
    fn get(&self, p: &na::Point2<f64>) -> na::Vector3<f64> {
        // Extend the triangle into a parallelogram, then bilinearly interpolate. This guarantees a
        // numerically exact result at each vertex, because in that case every vertex's contribution
        // is multiplied by 0 or 1 exactly and then summed. This precision ensures that there won't
        // be cracks between patches.
        let (b, c) = if p.x > p.y {
            (self.b, self.a + self.d - self.b)
        } else {
            (self.a + self.d - self.c, self.c)
        };
        let y0 = self.a * (1.0 - p.x) + b * p.x;
        let y1 = c * (1.0 - p.x) + self.d * p.x;
        y0 * (1.0 - p.y) + y1 * p.y
    }

    /// Map a direction in sphere space to a point in patch space
    #[inline(always)]
    fn project(&self, dir: &na::Vector3<f64>) -> na::Point2<f64> {
        // Project onto each triangle, then select the in-bounds result
        #[inline(always)]
        fn project(
            p: &na::Vector3<f64>,
            x: na::Vector3<f64>,
            y: na::Vector3<f64>,
            dir: &na::Vector3<f64>,
        ) -> na::Point2<f64> {
            // t * dir = p + u * x + v * y
            // -p = x * u + y * v - t * dir
            //    = [x y dir] [u v -t]^T
            // [u v -t]^T = [x y dir]^-1 . -p
            let m = na::Matrix3::from_columns(&[x, y, *dir]);
            (-(m.try_inverse().unwrap().fixed_view::<2, 3>(0, 0) * p)).into()
        }

        let left = project(&self.a, self.d - self.c, self.c - self.a, dir);
        let result = if left.x <= left.y {
            left
        } else {
            project(&self.a, self.b - self.a, self.d - self.b, dir)
        };
        result.map(|x| x.clamp(0.0, 1.0))
    }

    fn quads(&self, chunk_resolution: u32) -> impl Iterator<Item = Quad> + '_ {
        let quad_resolution = chunk_resolution - 1;
        (0..quad_resolution).flat_map(move |y| {
            (0..quad_resolution).map(move |x| Quad::new(self, quad_resolution, [x, y].into()))
        })
    }

    fn quads_within<'a>(
        &'a self,
        aabb: &Aabb,
        chunk_resolution: u32,
    ) -> impl Iterator<Item = Quad> + 'a {
        self.quads_within_inner(aabb, chunk_resolution)
            .into_iter()
            .flatten()
    }

    fn quads_within_inner<'a>(
        &'a self,
        aabb: &Aabb,
        chunk_resolution: u32,
    ) -> Option<impl Iterator<Item = Quad> + 'a> {
        let verts = aabb.vertices();
        let v0 = self.project(&verts[0].coords).coords;
        let (lower, upper) = verts[1..]
            .iter()
            .map(|v| self.project(&v.coords).coords)
            .fold((v0, v0), |(lower, upper), p| {
                (lower.zip_map(&p, f64::min), upper.zip_map(&p, f64::max))
            });
        if lower.iter().any(|&v| v == 1.0) || upper.iter().any(|&v| v == 0.0) {
            return None;
        }
        let quad_resolution = chunk_resolution - 1;
        let discretize = |x: f64| ((x * quad_resolution as f64) as u32).min(quad_resolution - 1);
        // FIXME: wrong units! Reuse bounding
        let lower = lower.map(discretize);
        let upper = upper.map(discretize);
        Some((lower.y..=upper.y).flat_map(move |y| {
            (lower.x..=upper.x).map(move |x| Quad::new(self, quad_resolution, [x, y].into()))
        }))
    }

    fn triangles<'a>(
        &'a self,
        radius: f64,
        chunk_resolution: u32,
        samples: &'a [f32],
    ) -> impl Iterator<Item = (u32, Triangle)> + 'a {
        self.quads(chunk_resolution).flat_map(move |quad| {
            let index = quad.index(chunk_resolution);
            quad.displace(radius, chunk_resolution, samples)
                .triangles()
                .into_iter()
                .enumerate()
                .map(move |(i, tri)| ((index << 1) | i as u32, tri))
        })
    }
}

/// Identifies a pair of triangles within a patch
struct Quad {
    /// Row-major order
    corners: [na::Vector3<f64>; 4],
    position: na::Point2<u32>,
}

impl Quad {
    fn new(patch: &Patch, resolution: u32, position: na::Point2<u32>) -> Self {
        let offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        Self {
            corners: offsets.map(|x| {
                patch
                    .get(&((position.cast::<f64>() + na::Vector2::from(x)) / f64::from(resolution)))
            }),
            position,
        }
    }

    fn index(&self, chunk_resolution: u32) -> u32 {
        self.position.y * chunk_resolution + self.position.x
    }

    fn displace(&self, radius: f64, chunk_resolution: u32, chunk_samples: &[f32]) -> DisplacedQuad {
        let offsets = [[0, 0], [1, 0], [0, 1], [1, 1]];
        let mut result = self.corners;
        for (v, offset) in result.iter_mut().zip(offsets) {
            let sample = self.position + na::Vector2::from(offset);
            let displacement = chunk_samples[(sample.y * chunk_resolution + sample.x) as usize];
            // We deliberately don't normalize `v` in `v * radius` because we're displacing a
            // subdivided patch, not the surface of the sphere directly.
            *v = *v * radius + v.normalize() * f64::from(displacement);
        }
        DisplacedQuad {
            corners: result.map(na::Point3::from),
        }
    }

    fn triangles<'a>(
        &'a self,
        radius: f64,
        chunk_resolution: u32,
        samples: &'a [f32],
    ) -> impl Iterator<Item = (u32, Triangle)> + 'a {
        let index = self.index(chunk_resolution);
        self.displace(radius, chunk_resolution, samples)
            .triangles()
            .into_iter()
            .enumerate()
            .map(move |(i, tri)| ((index << 1) | i as u32, tri))
    }
}

struct DisplacedQuad {
    /// Row-major order
    corners: [na::Point3<f64>; 4],
}

impl DisplacedQuad {
    fn triangles(&self) -> [Triangle; 2] {
        let [p0, p1, p2, p3] = self.corners;
        [Triangle::new(p0, p1, p3), Triangle::new(p3, p2, p0)]
    }
}

/// Invoke `f` on the row-major corners of every quad in `patch` along `ray`, which much originate
/// within `patch`. Returns the patch edge reached and the ray toi at which the edge was reached, if
/// any.
///
/// - `ray` must start within `patch`
/// - `f` returns whether to continue
fn walk_patch(
    quad_resolution: u32,
    patch: &Patch,
    ray: &Ray,
    max_toi: f64,
    mut f: impl FnMut(&Quad) -> bool,
) -> Option<(Edge, f64)> {
    let quad_resolution_f = quad_resolution as f64;
    let start = patch.project(&ray.origin.coords);
    let mut quad = start.map(|x| {
        (x * quad_resolution_f)
            .trunc()
            .clamp(0.0, quad_resolution_f - 1.0)
    });
    loop {
        let candidate = Quad::new(patch, quad_resolution, quad.map(|x| x as u32));
        if !f(&candidate) {
            return None;
        }
        // Find the next quad along the ray
        let Some((edge, toi)) = raycast_quad_edges(ray, &candidate.corners, max_toi) else {
            return None;
        };
        quad += edge.direction().into_inner();
        if quad.x >= quad_resolution_f
            || quad.y >= quad_resolution_f
            || quad.x < 0.0
            || quad.y < 0.0
        {
            // Reached the edge of the patch
            return Some((edge, toi));
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use parry3d_f64::{query::TOIStatus, shape::Ball};

    use crate::cubemap::Face;

    use super::*;

    #[test]
    fn triangles() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2)), 32, 1.0);
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        let samples = planet.sample(&coords);
        let patch = Patch::new(&coords, planet.terrain.face_resolution());
        assert_eq!(
            patch
                .triangles(planet.radius, planet.chunk_resolution, &samples)
                .count(),
            2
        );
        let expected = 1.0 / 3.0f64.sqrt();
        for (_, tri) in patch.triangles(planet.radius, planet.chunk_resolution, &samples) {
            assert!(tri.normal().unwrap().z > 0.0);
            for vert in &[tri.a, tri.b, tri.c] {
                assert!(vert.z > 0.0);
                for coord in &vert.coords {
                    assert_eq!(coord.abs(), expected);
                }
                assert_relative_eq!(vert.coords.norm(), 1.0);
            }
        }
    }

    fn ball_contacts(planet: &Planet, pos: Point<Real>, radius: Real) -> usize {
        let dispatcher = PlanetDispatcher.chain(DefaultQueryDispatcher);
        let ball = Ball { radius };
        let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
        let mut workspace = None;
        dispatcher
            .contact_manifolds(
                &Isometry::translation(pos.x, pos.y, pos.z),
                planet,
                &ball,
                0.0,
                &mut manifolds,
                &mut workspace,
            )
            .unwrap();
        manifolds.iter().map(|m| m.contacts().len()).sum()
    }

    #[test]
    fn end_to_end() {
        const PLANET_RADIUS: f64 = 6371e3;

        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        // We add 0.1 to PLANET_RADIUS in positive tests below to hack around the issue in
        // https://github.com/dimforge/parry/pull/148. Can be removed once fix is released.
        assert!(ball_contacts(&planet, Point::new(2.0, PLANET_RADIUS + 0.1, 0.0), 1.0) >= 2,
                "a ball lying on an axis of a planet with an even number of chunks per face overlaps with at least four triangles");
        assert_eq!(
            ball_contacts(&planet, Point::new(0.0, PLANET_RADIUS + 2.0, 0.0), 1.0),
            0
        );
        assert!(ball_contacts(&planet, Point::new(-1.0, PLANET_RADIUS + 0.1, 0.0), 1.0) > 0);

        for i in 0..10 {
            use std::f64;
            let rot = na::UnitQuaternion::from_axis_angle(
                &na::Vector3::z_axis(),
                (i as f64 / 1000.0) * f64::consts::PI * 1e-4,
            );
            let pos = Point::from(rot * na::Vector3::new(0.0, PLANET_RADIUS + 0.1, 0.0));
            assert!(ball_contacts(&planet, dbg!(pos), 1.0) > 0);
        }
    }

    // Ensure absence of a collision hole arising from mistakenly considering chunk centers *not* to
    // be offset by 0.5 / face_resolution from edges of cubemap faces.
    #[test]
    fn coordinate_center_regression() {
        const PLANET_RADIUS: f64 = 6371e3;
        const BALL_RADIUS: f64 = 50.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        let pos = Point::from(
            Vector::<Real>::new(-5_195_083.148, 3_582_099.812, -877_091.267).normalize()
                * PLANET_RADIUS,
        );

        assert!(ball_contacts(&planet, pos, BALL_RADIUS) > 0);
    }

    #[test]
    fn toi_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        let impact = PlanetDispatcher
            .time_of_impact(
                &Isometry::translation(PLANET_RADIUS + DISTANCE, 0.0, 0.0),
                &Vector::new(-1.0, 0.0, 0.0),
                &planet,
                &ball,
                100.0,
                false,
            )
            .unwrap()
            .expect("toi not found");
        assert_eq!(impact.status, TOIStatus::Converged);
        assert_relative_eq!(impact.toi, DISTANCE - ball.radius);
        assert_relative_eq!(impact.witness1, Point::new(PLANET_RADIUS, 0.0, 0.0));
        assert_relative_eq!(impact.witness2, Point::new(-ball.radius, 0.0, 0.0));
        assert_relative_eq!(impact.normal1, Vector::x_axis());
        assert_relative_eq!(impact.normal2, -Vector::x_axis());
    }

    #[test]
    fn ray_direct() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );
        let hit = planet
            .cast_local_ray_and_get_normal(
                &Ray {
                    origin: Point::new(PLANET_RADIUS + DISTANCE, 1.0, 1.0),
                    dir: -Vector::x(),
                },
                100.0,
                true,
            )
            .expect("hit not found");
        assert_relative_eq!(hit.toi, DISTANCE, epsilon = 1e-3);
        assert_relative_eq!(hit.normal, Vector::x_axis(), epsilon = 1e-3);

        let hit = planet.cast_local_ray_and_get_normal(
            &Ray {
                origin: Point::new(PLANET_RADIUS + DISTANCE, 1.0, 1.0),
                dir: Vector::x(),
            },
            100.0,
            true,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_perp() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        for &dir in [Vector::x(), Vector::y(), -Vector::x(), -Vector::y()].iter() {
            let hit = planet.cast_local_ray_and_get_normal(
                &Ray {
                    origin: Point::new(1.0, 1.0, PLANET_RADIUS + DISTANCE),
                    dir,
                },
                10000.0,
                true,
            );
            assert!(hit.is_none());
        }
    }

    #[test]
    fn ray_glancing() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 1000.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );
        planet
            .cast_local_ray_and_get_normal(
                &Ray {
                    origin: Point::new(1.0, 1.0, PLANET_RADIUS + DISTANCE),
                    dir: na::Vector3::new(1.5, 1.5, -1.0).normalize(),
                },
                1e5,
                true,
            )
            .expect("hit not found");
    }

    #[test]
    fn intersects_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        assert!(PlanetDispatcher
            .intersection_test(
                &Isometry::translation(PLANET_RADIUS, 0.0, 0.0),
                &planet,
                &ball,
            )
            .unwrap());
        assert!(!PlanetDispatcher
            .intersection_test(
                &Isometry::translation(PLANET_RADIUS + ball.radius * 2.0, 0.0, 0.0),
                &planet,
                &ball,
            )
            .unwrap());
    }

    #[test]
    fn nonlinear_toi_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
            32,
            PLANET_RADIUS,
        );

        let toi = PlanetDispatcher
            .nonlinear_time_of_impact(
                &NonlinearRigidMotion::constant_position(na::one()),
                &planet,
                &NonlinearRigidMotion {
                    start: Isometry::translation(PLANET_RADIUS + ball.radius + 0.5, 0.0, 0.0),
                    local_center: na::Point3::origin(),
                    linvel: -na::Vector3::x(),
                    angvel: na::zero(),
                },
                &ball,
                0.0,
                1.0,
                true,
            )
            .unwrap()
            .expect("no hit");
        assert_eq!(toi.status, TOIStatus::Converged);
        assert_relative_eq!(toi.toi, 0.5);
        assert_relative_eq!(toi.witness1, na::Point3::new(PLANET_RADIUS, 0.0, 0.0));
        assert_relative_eq!(toi.witness2, na::Point3::new(-ball.radius, 0.0, 0.0));
        assert_relative_eq!(toi.normal1, na::Vector3::x_axis());
        assert_relative_eq!(toi.normal2, -na::Vector3::x_axis());

        // Same configuration as above, but too far to hit within the allotted time
        let toi = PlanetDispatcher
            .nonlinear_time_of_impact(
                &NonlinearRigidMotion::constant_position(na::one()),
                &planet,
                &NonlinearRigidMotion {
                    start: Isometry::translation(PLANET_RADIUS + ball.radius + 1.5, 0.0, 0.0),
                    local_center: na::Point3::origin(),
                    linvel: -na::Vector3::x(),
                    angvel: na::zero(),
                },
                &ball,
                0.0,
                1.0,
                true,
            )
            .unwrap();
        assert!(toi.is_none());
    }

    #[test]
    fn patch_interpolation() {
        let res = 2u32.pow(12);
        let chunk = Coords {
            x: 12,
            y: 47,
            face: Face::Px,
        };
        let patch = Patch::new(&chunk, res);

        // Verify that the corners are consistent. All other points won't be, since
        // `Coords::direction` interpolates on the sphere.
        for coords in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
            // Exact equality is intended here, as a prerequisite for neighboring patches to be
            // seamless.
            assert_eq!(
                patch.get(&coords.into()),
                chunk.direction(res, &coords.into()).into_inner()
            );
        }
    }

    #[test]
    fn patch_projection() {
        let res = 2u32.pow(12);
        let chunk = Coords {
            x: 12,
            y: 47,
            face: Face::Px,
        };
        let patch = Patch::new(&chunk, res);

        let coords = [0.1, 0.4].into();
        assert_abs_diff_eq!(patch.project(&patch.get(&coords)), coords, epsilon = 1e-4);

        let coords = [0.9, 0.7].into();
        assert_abs_diff_eq!(patch.project(&patch.get(&coords)), coords, epsilon = 1e-4);
    }

    #[test]
    fn patch_projection_2() {
        // Regression test for a case that needs the second quadratic solution
        let patch = Patch {
            a: [1.0, 1.1, 0.0].into(),
            b: [1.0, 1.2, 1.3].into(),
            c: [1.0, -0.1, -0.1].into(),
            d: [1.0, 0.0, 1.0].into(),
        };

        let p = patch.project(&na::Vector3::new(1.0, 0.1, 0.1));
        assert!(p.x >= 0.0 && p.x <= 1.0);
        assert!(p.y >= 0.0 && p.y <= 1.0);
    }

    #[test]
    fn patch_projection_3() {
        // Regression test for a case that needs the second quadratic solution
        let patch = Patch {
            a: [1.0, 2.0, 1.0].into(),
            b: [1.0, 2.0, 2.0].into(),
            c: [1.0, 1.0, 1.0].into(),
            d: [1.0, 1.0, 2.0].into(),
        };

        assert_abs_diff_eq!(
            patch.project(&na::Vector3::new(1.0, 1.1, 1.1)),
            na::Point2::new(0.1, 0.9)
        );
        assert_abs_diff_eq!(
            patch.get(&na::Point2::new(0.1, 0.9)),
            na::Vector3::new(1.0, 1.1, 1.1)
        );
    }

    #[test]
    fn patch_raycast() {
        const RESOLUTION: u32 = 2;
        const Z: f64 = 1e6;
        // RESOLUTION x RESOLUTION square at z=1
        let patch = Patch {
            a: [0.0, 0.0, Z].into(),
            b: [RESOLUTION as f64, 0.0, Z].into(),
            c: [0.0, RESOLUTION as f64, Z].into(),
            d: [RESOLUTION as f64, RESOLUTION as f64, Z].into(),
        };

        let check_ray = |origin, direction, expected_quads: &[[u32; 2]], expected_edge| {
            let mut i = 0;
            let result = walk_patch(
                RESOLUTION,
                &patch,
                &Ray::new(origin, direction),
                100.0,
                |quad| {
                    let expected = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].map(|offset| {
                        (na::Vector2::from(expected_quads[i]).cast::<f64>()
                            + na::Vector2::from(offset))
                        .push(Z)
                    });
                    i += 1;
                    for (actual, expected) in quad.corners.into_iter().zip(&expected) {
                        assert_abs_diff_eq!(actual, expected);
                    }
                    true
                },
            );
            assert_eq!(result.map(|x| x.0), expected_edge);
        };

        check_ray(
            [0.5, 0.5, Z].into(),
            [1.0, 0.0, 0.0].into(),
            &[[0, 0], [1, 0]],
            Some(Edge::Px),
        );
        check_ray(
            [0.5, 0.5, Z].into(),
            [-1.0, 0.0, 0.0].into(),
            &[[0, 0]],
            Some(Edge::Nx),
        );
        check_ray(
            [1.5, 0.5, Z].into(),
            [-1.0, 0.0, 0.0].into(),
            &[[1, 0], [0, 0]],
            Some(Edge::Nx),
        );
        check_ray(
            [1.5, 1.5, Z].into(),
            [-1.0, 0.0, 0.0].into(),
            &[[1, 1], [0, 1]],
            Some(Edge::Nx),
        );
        check_ray(
            [1.5, 1.5, Z].into(),
            [0.0, 1.0, 0.0].into(),
            &[[1, 1]],
            Some(Edge::Py),
        );
        check_ray(
            [1.5, 1.5, Z].into(),
            [0.0, -1.0, 0.0].into(),
            &[[1, 1], [1, 0]],
            Some(Edge::Ny),
        );
    }
}
