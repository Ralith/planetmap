//! Collision detection for radial heightmaps
//!
//! Implement [`Terrain`] for your heightmap, then create colliders using it with a [`Planet`]. A
//! query dispatcher that handles both planets and standard shapes can be constructed with
//! `PlanetDispatcher.chain(DefaultQueryDispatcher)`.

use std::sync::{Arc, Mutex};

use hashbrown::hash_map;
use hashbrown::HashMap;
use parry3d_f64::{
    bounding_volume::{BoundingSphere, BoundingVolume, AABB},
    mass_properties::MassProperties,
    math::{Isometry, Point, Real, Vector},
    query::{
        ClosestPoints, Contact, ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher,
        NonlinearRigidMotion, PersistentQueryDispatcher, PointProjection, PointQuery,
        QueryDispatcher, Ray, RayCast, RayIntersection, TypedWorkspaceData, Unsupported,
        WorkspaceData, TOI,
    },
    shape::{FeatureId, Shape, ShapeType, Triangle, TypedShape},
};

use crate::{
    cubemap::{Coords, Edge},
    lru_slab::{LruSlab, SlotId},
};

/// Height data source for `Planet`
pub trait Terrain: Send + Sync + 'static {
    /// Generate a `resolution * resolution` grid of heights wrt. sea level
    fn sample(&self, resolution: u32, coords: &Coords, out: &mut [f32]);
    /// Number of blocks of samples along the edge of a cubemap face
    fn face_resolution(&self) -> u32;
    /// The maximum value that will ever be written by `sample`
    fn max_height(&self) -> f32;
    /// The minimum value that will ever be written by `sample`
    fn min_height(&self) -> f32;
}

/// Perfect sphere `Terrain` impl
#[derive(Debug, Copy, Clone)]
pub struct FlatTerrain {
    face_resolution: u32,
}

impl FlatTerrain {
    pub fn new(face_resolution: u32) -> Self {
        Self { face_resolution }
    }
}

impl Terrain for FlatTerrain {
    fn face_resolution(&self) -> u32 {
        self.face_resolution
    }
    fn max_height(&self) -> f32 {
        0.0
    }
    fn min_height(&self) -> f32 {
        0.0
    }

    fn sample(&self, _: u32, _: &Coords, out: &mut [f32]) {
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
    /// `chunk_resolution` - number of heightfield samples along the edge of a chunk
    pub fn new(
        terrain: Arc<dyn Terrain>,
        cache_size: u32,
        radius: f64,
        chunk_resolution: u32,
    ) -> Self {
        assert!(chunk_resolution > 1, "chunks must be at least 2x2");
        Self {
            terrain,
            radius,
            chunk_resolution,
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
        self.terrain
            .sample(self.chunk_resolution, &coords, &mut samples[..]);
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
        mut f: impl FnMut(&Coords, SlotId, u32, &Triangle) -> bool,
    ) {
        let dir = bounds.center().coords;
        let distance = dir.norm();
        let cache = &mut *self.cache.lock().unwrap();
        for coords in Coords::neighborhood(
            self.terrain.face_resolution(),
            na::convert(dir),
            bounds.radius().atan2(distance) as f32,
        ) {
            let (slot, data) = cache.get(self, &coords);
            if self.radius as f64 + data.max as f64 + bounds.radius() < distance {
                // Short-circuit if `other` is way above this chunk
                continue;
            }
            // Future work: should be able to filter triangles before actually computing them
            for (i, triangle) in ChunkTriangles::new(self, coords, &data.samples)
                .enumerate()
                .filter(|(_, tri)| {
                    tri.bounding_sphere(&Isometry::identity())
                        .intersects(&bounds)
                })
            {
                if !f(&coords, slot, i as u32, &triangle) {
                    break;
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
        let res = self.terrain.face_resolution();
        let mut coords = Coords::from_vector(res, &ray.origin.coords.cast());

        // Walk along the ray until we hit something
        let cache = &mut *self.cache.lock().unwrap();
        loop {
            let (slot, data) = cache.get(self, &coords);
            // FIXME: Rays can sometimes slip between the bounding planes of a chunk and the outer
            // edge of the triangles within. To avoid this, we should extend boundary triangles to
            // form a narrow skirt around the chunk, or check neighboring chunks when very close to
            // a boundary.

            // TODO: We could probably improve perf here by replacing the exhaustive ChunkTriangles
            // scan with something that just walks along the path of the ray, similar to how chunks
            // are traversed. Maybe even the same code?
            if let Some((index, mut hit)) = ChunkTriangles::new(self, coords, &data.samples)
                .filter_map(|tri| tri.cast_local_ray_and_get_normal(ray, max_toi, solid))
                .enumerate()
                .min_by(|(_, x), (_, y)| x.toi.partial_cmp(&y.toi).unwrap())
            {
                hit.feature = FeatureId::Face(self.feature_id(slot, index as u32));
                return Some(hit);
            }

            match raycast_edges(res, &coords, ray, max_toi) {
                None => return None,
                Some((edge, _)) => {
                    coords = coords.neighbors(res)[edge];
                }
            }
        }
    }
}

/// Find the edge of `chunk` crossed by a ray from `origin` towards `direction`, if any. Assumes
/// that `ray` passes through the cone defined by `chunk`.
fn raycast_edges(
    resolution: u32,
    chunk: &Coords,
    ray: &Ray,
    max_toi: Real,
) -> Option<(Edge, Real)> {
    use parry3d_f64::shape::HalfSpace;

    let edges: [(Edge, [[f64; 2]; 2]); 4] = [
        (Edge::Nx, [[0.0, 1.0], [0.0, 0.0]]),
        (Edge::Ny, [[0.0, 0.0], [1.0, 0.0]]),
        (Edge::Px, [[1.0, 0.0], [1.0, 1.0]]),
        (Edge::Py, [[1.0, 1.0], [0.0, 1.0]]),
    ];

    let mut closest = None;
    for &(edge, corners) in edges.iter() {
        // Construct inward-facing edge planes
        let v1 = chunk.direction(resolution, &na::Point2::from(corners[0]));
        let v2 = chunk.direction(resolution, &na::Point2::from(corners[1]));
        let plane = HalfSpace {
            normal: na::Unit::new_normalize(v1.cross(&v2)),
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
        self.project_local_point_and_get_feature(pt).0
    }

    fn project_local_point_and_get_feature(
        &self,
        pt: &Point<Real>,
    ) -> (PointProjection, FeatureId) {
        let coords = Coords::from_vector(self.terrain.face_resolution(), &na::convert(pt.coords));
        let distance2 = |x: &na::Point3<f64>| na::distance_squared(x, pt);
        let cache = &mut *self.cache.lock().unwrap();
        let (slot, data) = cache.get(self, &coords);
        let (idx, nearest) = ChunkTriangles::new(self, coords, &data.samples)
            .map(|tri| tri.project_local_point(pt, false))
            .enumerate()
            .min_by(|(_, x), (_, y)| {
                distance2(&x.point)
                    .partial_cmp(&distance2(&y.point))
                    .unwrap()
            })
            .unwrap();
        // TODO: Check neighborhood, so we don't miss as many cliff faces
        (nearest, FeatureId::Face(self.feature_id(slot, idx as u32)))
    }
}

impl Shape for Planet {
    fn compute_local_aabb(&self) -> AABB {
        AABB::from_half_extents(Point::origin(), Vector::repeat(self.max_radius()))
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
struct ChunkTriangles<'a> {
    planet: &'a Planet,
    samples: &'a [f32],
    coords: Coords,
    /// LSB identifies the triangle within a quad, remaining bits identify a quad by position in the
    /// row-major sequence
    index: u32,
}

impl<'a> ChunkTriangles<'a> {
    fn new(planet: &'a Planet, coords: Coords, samples: &'a [f32]) -> Self {
        Self {
            planet,
            samples,
            coords,
            index: 0,
        }
    }

    fn vertex(&self, x: u32, y: u32) -> Point<Real> {
        let height = self.samples[(y * self.planet.chunk_resolution + x) as usize];
        let quad_resolution = (self.planet.chunk_resolution - 1) as f64;
        let unit_coords = na::Point2::new(x as f64 / quad_resolution, y as f64 / quad_resolution);
        let dir = self
            .coords
            .direction(self.planet.terrain.face_resolution(), &unit_coords);
        na::Point3::from(dir.into_inner() * (self.planet.radius as f64 + height as f64))
    }

    fn get(&self) -> Triangle {
        let quad_resolution = self.planet.chunk_resolution - 1;

        let quad_index = self.index >> 1;
        let y = quad_index / quad_resolution;
        let x = quad_index % quad_resolution;
        let p0 = self.vertex(x, y);
        let p1 = self.vertex(x + 1, y);
        let p2 = self.vertex(x + 1, y + 1);
        let p3 = self.vertex(x, y + 1);
        let left = (self.index & 1) == 0;
        if left {
            Triangle::new(p0, p1, p2)
        } else {
            Triangle::new(p2, p3, p0)
        }
    }
}

impl Iterator for ChunkTriangles<'_> {
    type Item = Triangle;
    fn next(&mut self) -> Option<Triangle> {
        // Number of quads along a chunk edge
        let quad_resolution = self.planet.chunk_resolution - 1;
        // Two triangles per quad
        if self.index == quad_resolution * quad_resolution * 2 {
            return None;
        }
        let tri = self.get();
        self.index += 1;
        Some(tri)
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
                    .insert(ChunkData::new(*coords, planet.sample(&coords)));
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
    ) -> Result<Option<TOI>, Unsupported> {
        if let Some(p1) = g1.downcast_ref::<Planet>() {
            return Ok(compute_toi(pos12, vel12, p1, g2, max_toi, false));
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            return Ok(compute_toi(
                &pos12.inverse(),
                &-vel12,
                p2,
                g1,
                max_toi,
                true,
            ));
        }
        Err(Unsupported)
    }

    fn nonlinear_time_of_impact(
        &self,
        _motion1: &NonlinearRigidMotion,
        _g1: &dyn Shape,
        _motion2: &NonlinearRigidMotion,
        _g2: &dyn Shape,
        _start_time: Real,
        _end_time: Real,
        _stop_at_penetration: bool,
    ) -> Result<Option<TOI>, Unsupported> {
        Err(Unsupported)
    }
}

fn intersects(pos12: &Isometry<Real>, planet: &Planet, other: &dyn Shape) -> bool {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    let bounds = other.compute_bounding_sphere(pos12);
    let mut intersects = false;
    planet.map_elements_in_local_sphere(&bounds, |_, _, _, triangle| {
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
    flipped: bool,
) -> Option<TOI> {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    // TODO: To improve perf with large max_toi, select chunks similarly to ray casts, maybe using
    // ray-sphere from hit point vs. ray through chunk vertex to detect secondary hits, from which
    // no recursive traversal proceeds?
    let bounds = {
        let start = other.compute_aabb(pos12);
        let end = start.transform_by(&Isometry::from_parts((max_toi * vel12).into(), na::one()));
        start.merged(&end).bounding_sphere()
    };
    let mut closest = None::<TOI>;
    planet.map_elements_in_local_sphere(&bounds, |_, _, _, triangle| {
        let impact = if flipped {
            dispatcher.time_of_impact(&pos12.inverse(), &-vel12, other, triangle, max_toi)
        } else {
            dispatcher.time_of_impact(pos12, vel12, triangle, other, max_toi)
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
            compute_manifolds(pos12, p1, g2, prediction, manifolds, workspace, false);
            return Ok(());
        }
        if let Some(p2) = g2.downcast_ref::<Planet>() {
            compute_manifolds(
                &pos12.inverse(),
                p2,
                g1,
                prediction,
                manifolds,
                workspace,
                true,
            );
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
        .get_or_insert_with(|| ContactManifoldsWorkspace(Box::new(Workspace::default())))
        .0
        .downcast_mut::<Workspace>()
        .unwrap();
    let dispatcher = DefaultQueryDispatcher; // TODO after https://github.com/dimforge/parry/issues/8

    workspace.color ^= true;
    let color = workspace.color;

    let bounds = other.compute_bounding_sphere(pos12).loosened(prediction);
    let mut old_manifolds = std::mem::replace(manifolds, Vec::new());
    planet.map_elements_in_local_sphere(&bounds, |&coords, slot, index, triangle| {
        let tri_state = match workspace.state.entry((coords, index)) {
            hash_map::Entry::Occupied(e) => {
                let tri_state = e.into_mut();

                let manifold = old_manifolds[tri_state.manifold_index].take();
                tri_state.manifold_index = manifolds.len();
                tri_state.color = color;
                manifolds.push(manifold);

                tri_state
            }
            hash_map::Entry::Vacant(e) => {
                let tri_state = TriangleState {
                    manifold_index: manifolds.len(),
                    workspace: None,
                    color,
                };

                let id = planet.feature_id(slot, index) as u32;
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

    workspace.state.retain(|_, x| x.color == color);
}

/// Narrow-phase collision detection state for `Planet`
#[derive(Default, Clone)]
pub struct Workspace {
    state: HashMap<(Coords, u32), TriangleState>,
    color: bool,
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
    workspace: Option<ContactManifoldsWorkspace>,
    color: bool,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use parry3d_f64::{query::TOIStatus, shape::Ball};

    use crate::cubemap::Face;

    use super::*;

    #[test]
    fn triangles() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1)), 32, 1.0, 2);
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        let samples = planet.sample(&coords);
        let tris = ChunkTriangles::new(&planet, coords, &samples[..]);
        assert_eq!(tris.clone().count(), 2);
        let expected = 1.0 / 3.0f64.sqrt();
        for tri in tris {
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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
        );

        assert!(ball_contacts(&planet, Point::new(2.0, PLANET_RADIUS as f64, 0.0), 1.0) > 1);
        assert_eq!(
            ball_contacts(
                &planet,
                Point::new(0.0, PLANET_RADIUS as f64 + 2.0, 0.0),
                1.0
            ),
            0
        );
        assert!(ball_contacts(&planet, Point::new(-1.0, PLANET_RADIUS as f64, 0.0), 1.0) > 0);

        for i in 1..10 {
            use std::f64;
            let rot = na::UnitQuaternion::from_axis_angle(
                &na::Vector3::z_axis(),
                (i as f64 / 1000.0) * f64::consts::PI * 1e-4,
            );
            let pos = Point::from(rot * na::Vector3::new(0.0, PLANET_RADIUS as f64, 0.0));
            assert!(ball_contacts(&planet, pos, 1.0) > 0);
        }
    }

    // Ensure absence of a collision hole arising from mistakenly considering chunk centers *not* to
    // be offset by 0.5 / face_resolution from edges of cubemap faces.
    #[test]
    fn coordinate_center_regression() {
        const PLANET_RADIUS: f64 = 6371e3;
        const BALL_RADIUS: f64 = 50.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
        );

        let impact = PlanetDispatcher
            .time_of_impact(
                &Isometry::translation(PLANET_RADIUS + DISTANCE, 0.0, 0.0),
                &Vector::new(-1.0, 0.0, 0.0),
                &planet,
                &ball,
                100.0,
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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
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
        assert_relative_eq!(hit.toi, DISTANCE, epsilon = 1e-4);
        assert_relative_eq!(hit.normal, Vector::x_axis(), epsilon = 1e-4);

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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
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
            Arc::new(FlatTerrain::new(2u32.pow(12))),
            32,
            PLANET_RADIUS,
            17,
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
}
