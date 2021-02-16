//! Collision detection for radial heightmaps
//!
//! Implement [`Terrain`] for your heightmap, then create colliders using it with a [`Planet`]. A
//! query dispatcher that handles both planets and standard shapes can be constructed with
//! `PlanetDispatcher.chain(DefaultQueryDispatcher)`.

use std::sync::{Arc, Mutex};

use hashbrown::hash_map;
use hashbrown::HashMap;
use lru::LruCache;
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

use crate::cubemap::Coords;

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
    cache: Mutex<LruCache<Coords, ChunkData>>,
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
        cache_size: usize,
        radius: f64,
        chunk_resolution: u32,
    ) -> Self {
        assert!(chunk_resolution > 1, "chunks must be at least 2x2");
        Self {
            terrain,
            radius,
            chunk_resolution,
            cache: Mutex::new(LruCache::new(cache_size)),
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

    fn feature_id(&self, _coords: &Coords, _triangle: usize, _tri_feature: FeatureId) -> FeatureId {
        use FeatureId::*;
        // TODO: Maintain an index into the cache, kept alive by live manifold generators, for improved stability
        Unknown
        // match tri_feature {
        //     Vertex(n) => Vertex(triangle << 2 | n),
        //     Edge(n) => Edge(triangle << 2 | n),
        //     Face(_) => Face(triangle),
        //     Unknown => Unknown,
        // }
    }

    /// Applies the function `f` to all the triangles intersecting the given sphere
    pub fn map_elements_in_local_sphere(
        &self,
        bounds: &BoundingSphere,
        mut f: impl FnMut(&Coords, u32, &Triangle),
    ) {
        let dir = bounds.center().coords;
        let distance = dir.norm();
        let cache = &mut *self.cache.lock().unwrap();
        for coords in Coords::neighborhood(
            self.terrain.face_resolution(),
            na::convert(dir),
            bounds.radius().atan2(distance) as f32,
        ) {
            let data = if let Some(x) = cache.get(&coords) {
                x
            } else {
                cache.put(coords, ChunkData::new(self.sample(&coords)));
                cache.get(&coords).unwrap()
            };
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
                f(&coords, i as u32, &triangle)
            }
        }
    }
}

impl Clone for Planet {
    fn clone(&self) -> Self {
        Self {
            terrain: self.terrain.clone(),
            cache: Mutex::new(LruCache::new(self.cache.lock().unwrap().cap())),
            ..*self
        }
    }
}

impl RayCast for Planet {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &Ray,
        _max_toi: Real,
        _solid: bool,
    ) -> Option<RayIntersection> {
        todo!()
    }
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
        let data = if let Some(x) = cache.get(&coords) {
            x
        } else {
            cache.put(coords, ChunkData::new(self.sample(&coords)));
            cache.get(&coords).unwrap()
        };
        let (idx, (nearest, feature)) = ChunkTriangles::new(self, coords, &data.samples)
            .map(|tri| tri.project_local_point_and_get_feature(pt))
            .enumerate()
            .min_by(|(_, (x, _)), (_, (y, _))| {
                distance2(&x.point)
                    .partial_cmp(&distance2(&y.point))
                    .unwrap()
            })
            .unwrap();
        // TODO: Check neighborhood, so we don't miss as many cliff faces
        (nearest, self.feature_id(&coords, idx, feature))
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

#[derive(Debug, Clone)]
struct ChunkData {
    samples: Box<[f32]>,
    min: f32,
    max: f32,
}

impl ChunkData {
    fn new(samples: Box<[f32]>) -> Self {
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
        Self { samples, min, max }
    }
}

/// A [`PersistentQueryDispatcher`] that handles `Planet` shapes
pub struct PlanetDispatcher;

// TODO: Fill these in
impl QueryDispatcher for PlanetDispatcher {
    fn intersection_test(
        &self,
        _pos12: &Isometry<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
    ) -> Result<bool, Unsupported> {
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
        _pos12: &Isometry<Real>,
        _vel12: &Vector<Real>,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _max_toi: Real,
    ) -> Result<Option<TOI>, Unsupported> {
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
    planet.map_elements_in_local_sphere(&bounds, |&coords, index, triangle| {
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

                // TODO: Subshape IDs
                manifolds.push(ContactManifold::with_data(0, 0, ManifoldData::default()));

                e.insert(tri_state)
            }
        };

        let manifold = &mut manifolds[tri_state.manifold_index];

        // TODO: Why can we use convex-convex here?
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
    use parry3d_f64::shape::Ball;

    use crate::cubemap::Face;

    use super::*;

    #[test]
    fn triangles() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1)), 32, 1.0, 2);
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::PZ,
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
}
