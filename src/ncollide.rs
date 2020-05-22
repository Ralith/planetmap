//! Collision detection for radial heightmaps
//!
//! Implement `Terrain` for your heightmap, then create colliders using it with a `Planet`. The
//! `CollisionWorld` must be configured to use `PlanetManifoldGenerator`s for collision detection
//! vs. `Planet`s, for example by using a `NarrowPhase` with a
//! `PlanetDispatcher<DefaultContactDispatcher>`.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use planetmap::ncollide::{PlanetDispatcher, PlanetProximityDispatcher, Planet, FlatTerrain};
//! use ncollide3d::{
//!     narrow_phase::{DefaultContactDispatcher, DefaultProximityDispatcher, NarrowPhase},
//!     shape::ShapeHandle,
//!     pipeline::{
//!         object::{CollisionGroups, GeometricQueryType},
//!         world::CollisionWorld,
//!     },
//! };
//!
//! let mut world = CollisionWorld::new(0.01);
//! world.set_narrow_phase(NarrowPhase::new(
//!     Box::new(PlanetDispatcher::new(DefaultContactDispatcher::new())),
//!     Box::new(PlanetProximityDispatcher::new(DefaultProximityDispatcher::new())),
//! ));
//!
//! world.add(
//!     na::Isometry3::identity(),
//!     ShapeHandle::new(Planet::new(Arc::new(FlatTerrain::new(16)), 32, 1.0, 4)),
//!     CollisionGroups::new(),
//!     GeometricQueryType::Contacts(0.0, 0.0),
//!     0,
//! );
//! ```

use std::sync::{Arc, Mutex};

use hashbrown::hash_map;
use hashbrown::HashMap;
use lru::LruCache;
use na::RealField;
use ncollide3d::{
    bounding_volume::{BoundingSphere, BoundingVolume, HasBoundingVolume, AABB},
    narrow_phase::{ContactAlgorithm, ContactDispatcher, ContactManifoldGenerator},
    query::{
        Contact, ContactKinematic, ContactManifold, ContactPrediction, ContactPreprocessor,
        PointProjection, PointQuery, Ray, RayCast, RayIntersection,
    },
    shape::{FeatureId, Shape, Triangle},
};

use crate::cubemap::Coords;
use ncollide3d::narrow_phase::{ProximityAlgorithm, ProximityDetector, ProximityDispatcher};
use ncollide3d::query::Proximity;

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
    radius: f32,
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
        radius: f32,
        chunk_resolution: u32,
    ) -> Self {
        assert!(chunk_resolution > 1);
        Self {
            terrain,
            radius,
            chunk_resolution,
            cache: Mutex::new(LruCache::new(cache_size)),
        }
    }

    fn max_radius(&self) -> f64 {
        self.radius as f64 + self.terrain.max_height() as f64
    }
    fn min_radius(&self) -> f64 {
        self.radius as f64 + self.terrain.min_height() as f64
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

impl<N: RealField> HasBoundingVolume<N, BoundingSphere<N>> for Planet {
    fn bounding_volume(&self, m: &na::Isometry3<N>) -> BoundingSphere<N> {
        BoundingSphere::new(m * na::Point3::origin(), na::convert(self.max_radius()))
    }
}

impl<N: RealField> HasBoundingVolume<N, AABB<N>> for Planet {
    fn bounding_volume(&self, m: &na::Isometry3<N>) -> AABB<N> {
        let radius = na::convert(self.max_radius());
        AABB::from_half_extents(
            m * na::Point3::origin(),
            na::Vector3::new(radius, radius, radius),
        )
    }
}

impl PointQuery<f64> for Planet {
    fn project_point(
        &self,
        m: &na::Isometry3<f64>,
        pt: &na::Point3<f64>,
        solid: bool,
    ) -> PointProjection<f64> {
        if solid && na::distance_squared(pt, &(m * na::Point3::origin())) < self.min_radius() {
            return PointProjection {
                is_inside: true,
                point: *pt,
            };
        };
        self.project_point_with_feature(m, pt).0
    }

    fn project_point_with_feature(
        &self,
        m: &na::Isometry3<f64>,
        pt: &na::Point3<f64>,
    ) -> (PointProjection<f64>, FeatureId) {
        let local = m.inverse_transform_point(pt);
        let coords =
            Coords::from_vector(self.terrain.face_resolution(), &na::convert(local.coords));
        let distance2 = |x: &na::Point3<f64>| na::distance_squared(x, &local);
        let cache = &mut *self.cache.lock().unwrap();
        let data = if let Some(x) = cache.get(&coords) {
            x
        } else {
            cache.put(coords, ChunkData::new(self.sample(&coords)));
            cache.get(&coords).unwrap()
        };
        let (idx, (nearest, feature)) = ChunkTriangles::new(self, coords, &data.samples)
            .map(|tri| tri.project_point_with_feature(m, &local))
            .enumerate()
            .min_by(|(_, (x, _)), (_, (y, _))| {
                distance2(&x.point)
                    .partial_cmp(&distance2(&y.point))
                    .unwrap()
            })
            .unwrap();
        // TODO: Check neighborhood, so we don't miss as many cliff faces
        (
            PointProjection {
                point: m * nearest.point,
                ..nearest
            },
            self.feature_id(&coords, idx, feature),
        )
    }
}

impl Shape<f64> for Planet {
    #[inline]
    fn aabb(&self, m: &na::Isometry3<f64>) -> AABB<f64> {
        self.bounding_volume(m)
    }

    #[inline]
    fn bounding_sphere(&self, m: &na::Isometry3<f64>) -> BoundingSphere<f64> {
        self.bounding_volume(m)
    }

    #[inline]
    fn tangent_cone_contains_dir(
        &self,
        _fid: FeatureId,
        _m: &na::Isometry3<f64>,
        _deformations: Option<&[f64]>,
        _dir: &na::Unit<na::Vector3<f64>>,
    ) -> bool {
        // TODO: Implementing this properly will improve stability
        false
    }

    #[inline]
    fn as_point_query(&self) -> Option<&dyn PointQuery<f64>> {
        Some(self)
    }
}

impl RayCast<f64> for Planet {
    fn toi_and_normal_with_ray(
        &self,
        _m: &na::Isometry3<f64>,
        _ray: &Ray<f64>,
        _max_toi: f64,
        _solid: bool,
    ) -> Option<RayIntersection<f64>> {
        unimplemented!()
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

    fn vertex(&self, x: u32, y: u32) -> na::Point3<f64> {
        let height = self.samples[(y * self.planet.chunk_resolution + x) as usize];
        let quad_resolution = (self.planet.chunk_resolution - 1) as f64;
        let unit_coords = na::Point2::new(x as f64 / quad_resolution, y as f64 / quad_resolution);
        let dir = self
            .coords
            .direction(self.planet.terrain.face_resolution(), &unit_coords);
        na::Point3::from(dir.into_inner() * (self.planet.radius as f64 + height as f64))
    }

    fn get(&self) -> Triangle<f64> {
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
    type Item = Triangle<f64>;
    fn next(&mut self) -> Option<Triangle<f64>> {
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

/// Narrow-phase collision detection algorithm for `Planet`
pub struct PlanetManifoldGenerator {
    flip: bool,
    state: HashMap<(Coords, usize), TriangleContactData>,
    color: bool,
}

impl PlanetManifoldGenerator {
    /// `flip` - whether the planet is the second shape
    pub fn new(flip: bool) -> Self {
        Self {
            flip,
            state: HashMap::new(),
            color: false,
        }
    }

    fn contact(
        &mut self,
        dispatcher: &dyn ContactDispatcher<f64>,
        planet_transform: &na::Isometry3<f64>,
        planet: &Planet,
        proc1: Option<&dyn ContactPreprocessor<f64>>,
        other_transform: &na::Isometry3<f64>,
        other: &dyn Shape<f64>,
        proc2: Option<&dyn ContactPreprocessor<f64>>,
        prediction: &ContactPrediction<f64>,
        manifold: &mut ContactManifold<f64>,
    ) {
        self.color ^= true;
        let color = self.color;

        let bounds = other
            .bounding_sphere(other_transform)
            .loosened(prediction.linear());
        let dir = planet_transform
            .inverse_transform_point(bounds.center())
            .coords;

        let mut cache_guard = planet.cache.lock().unwrap();
        let triangle_iter = ChunkTriangleBroadphaseIter::new(
            &mut *cache_guard,
            planet,
            *planet_transform,
            dir,
            bounds,
        );

        for (coords, triangle, triangle_index) in triangle_iter {
            let tri = match self.state.entry((coords, triangle_index)) {
                hash_map::Entry::Occupied(mut e) => {
                    e.get_mut().color = color;
                    e.into_mut()
                }
                hash_map::Entry::Vacant(e) => {
                    if let Some(algo) = if !self.flip {
                        dispatcher.get_contact_algorithm(&triangle, other)
                    } else {
                        dispatcher.get_contact_algorithm(other, &triangle)
                    } {
                        e.insert(TriangleContactData { algo, color })
                    } else {
                        // no contact algorithm found
                        return;
                    }
                }
            };
            let proc1 = TriangleContactPreprocessor {
                planet,
                outer: proc1,
                coords,
                triangle: triangle_index,
            };
            if !self.flip {
                tri.algo.generate_contacts(
                    dispatcher,
                    planet_transform,
                    &triangle,
                    Some(&proc1),
                    other_transform,
                    other,
                    proc2,
                    prediction,
                    manifold,
                );
            } else {
                tri.algo.generate_contacts(
                    dispatcher,
                    other_transform,
                    other,
                    proc2,
                    planet_transform,
                    &triangle,
                    Some(&proc1),
                    prediction,
                    manifold,
                );
            }
        }

        self.state.retain(|_, x| x.color == color);
    }
}

impl ContactManifoldGenerator<f64> for PlanetManifoldGenerator {
    fn generate_contacts(
        &mut self,
        dispatcher: &dyn ContactDispatcher<f64>,
        ma: &na::Isometry3<f64>,
        a: &dyn Shape<f64>,
        proc1: Option<&dyn ContactPreprocessor<f64>>,
        mb: &na::Isometry3<f64>,
        b: &dyn Shape<f64>,
        proc2: Option<&dyn ContactPreprocessor<f64>>,
        prediction: &ContactPrediction<f64>,
        manifold: &mut ContactManifold<f64>,
    ) -> bool {
        if !self.flip {
            if let Some(p) = a.as_shape::<Planet>() {
                self.contact(dispatcher, ma, p, proc1, mb, b, proc2, prediction, manifold);
                return true;
            }
        } else {
            if let Some(p) = b.as_shape::<Planet>() {
                self.contact(dispatcher, mb, p, proc2, ma, a, proc1, prediction, manifold);
                return true;
            }
        }
        false
    }
}

/// Narrow-phase collision detection algorithm for `Planet`
pub struct PlanetProximityGenerator {
    flip: bool,
    state: HashMap<(Coords, usize), TriangleProximityData>,
    color: bool,
}

impl PlanetProximityGenerator {
    /// `flip` - whether the planet is the second shape
    pub fn new(flip: bool) -> Self {
        Self {
            flip,
            state: HashMap::new(),
            color: false,
        }
    }

    fn proximity(
        &mut self,
        dispatcher: &dyn ProximityDispatcher<f64>,
        planet_transform: &na::Isometry3<f64>,
        planet: &Planet,
        other_transform: &na::Isometry3<f64>,
        other: &dyn Shape<f64>,
        margin: f64,
    ) -> Option<Proximity> {
        self.color ^= true;
        let color = self.color;

        let bounds = other.bounding_sphere(other_transform);
        let dir = planet_transform
            .inverse_transform_point(bounds.center())
            .coords;

        // used to store Proximity::Disjoint or Proximity::WithinMargin cases (logical OR of the for
        // loop)
        let mut result = Proximity::Disjoint;

        let mut cache_guard = planet.cache.lock().unwrap();
        let triangle_iter = ChunkTriangleBroadphaseIter::new(
            &mut *cache_guard,
            planet,
            *planet_transform,
            dir,
            bounds,
        );

        for (coords, triangle, triangle_index) in triangle_iter {
            let tri = match self.state.entry((coords, triangle_index)) {
                hash_map::Entry::Occupied(mut e) => {
                    e.get_mut().color = color;
                    e.into_mut()
                }
                hash_map::Entry::Vacant(e) => {
                    if let Some(algo) = if !self.flip {
                        dispatcher.get_proximity_algorithm(&triangle, other)
                    } else {
                        dispatcher.get_proximity_algorithm(other, &triangle)
                    } {
                        e.insert(TriangleProximityData { algo, color })
                    } else {
                        // no proximity algorithm found
                        return None;
                    }
                }
            };

            let res = if !self.flip {
                tri.algo.update(
                    dispatcher,
                    planet_transform,
                    &triangle,
                    other_transform,
                    other,
                    margin,
                )
            } else {
                tri.algo.update(
                    dispatcher,
                    other_transform,
                    other,
                    planet_transform,
                    &triangle,
                    margin,
                )
            };

            match res {
                Some(Proximity::WithinMargin) => {
                    result = Proximity::WithinMargin;
                }
                Some(Proximity::Intersecting) => {
                    result = Proximity::Intersecting;
                    break;
                }
                _ => {}
            }
        }

        self.state.retain(|_, x| x.color == color);

        Some(result)
    }
}

impl ProximityDetector<f64> for PlanetProximityGenerator {
    fn update(
        &mut self,
        dispatcher: &dyn ProximityDispatcher<f64>,
        ma: &na::Isometry3<f64>,
        a: &dyn Shape<f64>,
        mb: &na::Isometry3<f64>,
        b: &dyn Shape<f64>,
        margin: f64,
    ) -> Option<Proximity> {
        if !self.flip {
            if let Some(p) = a.as_shape::<Planet>() {
                self.proximity(dispatcher, ma, p, mb, b, margin)
            } else {
                None
            }
        } else {
            if let Some(p) = b.as_shape::<Planet>() {
                self.proximity(dispatcher, mb, p, ma, a, margin)
            } else {
                None
            }
        }
    }
}

/// Used to iterate over every triangle in proximity to the given collider.
struct ChunkTriangleBroadphaseIter<'a> {
    neighborhood: Box<dyn Iterator<Item = Coords>>,
    current_chunk: Option<(Coords, std::iter::Enumerate<ChunkTriangles<'a>>)>,

    planet_cache: &'a mut LruCache<Coords, ChunkData>,
    planet: &'a Planet,
    planet_transform: na::Isometry3<f64>,
    other_collider_bounds: BoundingSphere<f64>,
    distance: f64,
}

impl<'a> ChunkTriangleBroadphaseIter<'a> {
    fn new(
        planet_cache: &'a mut LruCache<Coords, ChunkData>,
        planet: &'a Planet,
        planet_transform: na::Isometry3<f64>,
        dir: na::Vector3<f64>,
        other_collider_bounds: BoundingSphere<f64>,
    ) -> ChunkTriangleBroadphaseIter<'a> {
        let distance = dir.norm();

        ChunkTriangleBroadphaseIter {
            neighborhood: Box::new(Coords::neighborhood(
                planet.terrain.face_resolution(),
                na::convert(dir),
                other_collider_bounds.radius().atan2(distance) as f32,
            )),
            current_chunk: None,

            planet_cache,
            planet,
            planet_transform,
            distance: dir.norm(),
            other_collider_bounds,
        }
    }
}

impl<'a> Iterator for ChunkTriangleBroadphaseIter<'a> {
    type Item = (Coords, Triangle<f64>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut temp_chunk = None;
            let (coords, (triangle_idx, triangle)) = if self.current_chunk.is_some() && {
                let (coords, triangle_iter) = self.current_chunk.as_mut().unwrap();

                temp_chunk = triangle_iter.next().map(|b| (*coords, b));
                temp_chunk.is_some()
            } {
                // Current chunk has triangles left to yield

                temp_chunk.unwrap()
            } else {
                // Current chunk has no more triangles left to yield (or has just been initialized to None)

                loop {
                    let coords = self.neighborhood.next()?;

                    // Also drops the reference to the chunk's sample buffer reference (see below in proof of safety).
                    self.current_chunk = None;

                    let chunk_data = if let Some(_) = self.planet_cache.get(&coords) {
                        self.planet_cache.peek(&coords).unwrap()
                    } else {
                        self.planet_cache
                            .put(coords, ChunkData::new(self.planet.sample(&coords)));

                        self.planet_cache.peek(&coords).unwrap()
                    };

                    // Skip this chunk if `other` is way above or below it
                    if self.planet.radius as f64
                        + chunk_data.max as f64
                        + self.other_collider_bounds.radius()
                        >= self.distance
                        || self.planet.radius as f64 + chunk_data.min as f64
                            - self.other_collider_bounds.radius()
                            < self.distance
                    {
                        continue;
                    } else {
                        // SAFETY: `self.current_chunk` is set to `None` before we mutate the LruCache again.
                        //         The mutably borrowed LruCache is only ever modified in this loop's block.
                        let elided_lifetime_borrow =
                            unsafe { &*(&*chunk_data.samples as *const _) };

                        self.current_chunk = Some((
                            coords,
                            ChunkTriangles::new(self.planet, coords, elided_lifetime_borrow)
                                .enumerate(),
                        ));

                        let (_, triangle_iter) = self.current_chunk.as_mut().unwrap();

                        // Panic safety: We assume that the newly made ChunkTriangles yields at least one triangle
                        break (
                            coords,
                            triangle_iter
                                .next()
                                .expect("ChunkTriangles failed to yield at least one triangle"),
                        );
                    }
                }
            };

            // Skip this triangle if its bounding sphere does not collide with `other`'s bounding sphere
            if triangle
                .bounding_sphere(&self.planet_transform)
                .intersects(&self.other_collider_bounds)
            {
                break Some((coords, triangle, triangle_idx));
            } else {
                continue;
            }
        }
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

struct TriangleContactData {
    algo: ContactAlgorithm<f64>,
    color: bool,
}

struct TriangleProximityData {
    algo: ProximityAlgorithm<f64>,
    color: bool,
}

/// A `ProximityDispatcher` that knows about `Planet`
pub struct PlanetProximityDispatcher<T> {
    inner: T,
}

impl<T> PlanetProximityDispatcher<T> {
    /// Construct a dispatcher that forwards unrecognized shape pairs to `inner`
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: ProximityDispatcher<f64>> ProximityDispatcher<f64> for PlanetProximityDispatcher<T> {
    fn get_proximity_algorithm(
        &self,
        a: &dyn Shape<f64>,
        b: &dyn Shape<f64>,
    ) -> Option<ProximityAlgorithm<f64>> {
        if a.is_shape::<Planet>() {
            return Some(Box::new(PlanetProximityGenerator::new(false)));
        }
        if b.is_shape::<Planet>() {
            return Some(Box::new(PlanetProximityGenerator::new(true)));
        }
        self.inner.get_proximity_algorithm(a, b)
    }
}

/// A `ContactDispatcher` that knows about `Planet`
pub struct PlanetDispatcher<T> {
    inner: T,
}

impl<T> PlanetDispatcher<T> {
    /// Construct a dispatcher that forwards unrecognized shape pairs to `inner`
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: ContactDispatcher<f64>> ContactDispatcher<f64> for PlanetDispatcher<T> {
    fn get_contact_algorithm(
        &self,
        a: &dyn Shape<f64>,
        b: &dyn Shape<f64>,
    ) -> Option<ContactAlgorithm<f64>> {
        if a.is_shape::<Planet>() {
            return Some(Box::new(PlanetManifoldGenerator::new(false)));
        }
        if b.is_shape::<Planet>() {
            return Some(Box::new(PlanetManifoldGenerator::new(true)));
        }
        self.inner.get_contact_algorithm(a, b)
    }
}

struct TriangleContactPreprocessor<'a, N: RealField> {
    planet: &'a Planet,
    outer: Option<&'a dyn ContactPreprocessor<N>>,
    coords: Coords,
    triangle: usize,
}

impl<N: RealField> ContactPreprocessor<N> for TriangleContactPreprocessor<'_, N> {
    fn process_contact(
        &self,
        contact: &mut Contact<N>,
        kinematic: &mut ContactKinematic<N>,
        is_first: bool,
    ) -> bool {
        if is_first {
            kinematic.set_feature1(self.planet.feature_id(
                &self.coords,
                self.triangle,
                kinematic.feature1(),
            ));
        } else {
            kinematic.set_feature2(self.planet.feature_id(
                &self.coords,
                self.triangle,
                kinematic.feature2(),
            ));
        }

        if let Some(x) = self.outer {
            x.process_contact(contact, kinematic, is_first)
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ncollide3d::{
        narrow_phase::{DefaultContactDispatcher, DefaultProximityDispatcher, NarrowPhase},
        pipeline::{
            object::{CollisionGroups, GeometricQueryType},
            world::CollisionWorld,
        },
        shape::{Ball, ShapeHandle},
    };

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
            for vert in &[tri.a(), tri.b(), tri.c()] {
                assert!(vert.z > 0.0);
                for coord in &vert.coords {
                    assert_eq!(coord.abs(), expected);
                }
                assert_relative_eq!(vert.coords.norm(), 1.0);
            }
        }
    }

    #[test]
    fn end_to_end() {
        let mut world = CollisionWorld::new(0.01);
        world.set_narrow_phase(NarrowPhase::new(
            Box::new(PlanetDispatcher::new(DefaultContactDispatcher::new())),
            Box::new(DefaultProximityDispatcher::new()),
        ));

        const PLANET_RADIUS: f32 = 6371e3;
        const BALL_RADIUS: f64 = 1.0;

        world.add(
            na::Isometry3::identity(),
            ShapeHandle::new(Planet::new(
                Arc::new(FlatTerrain::new(2u32.pow(12))),
                32,
                PLANET_RADIUS,
                17,
            )),
            CollisionGroups::new(),
            GeometricQueryType::Contacts(0.0, 0.0),
            0,
        );
        let (ball, _) = world.add(
            na::convert(na::Translation3::new(2.0, PLANET_RADIUS as f64, 0.0)),
            ShapeHandle::new(Ball::new(BALL_RADIUS)),
            CollisionGroups::new(),
            GeometricQueryType::Contacts(0.0, 0.0),
            0,
        );

        world.update();
        assert!(world.contact_pairs(true).count() > 0);

        world
            .get_mut(ball)
            .unwrap()
            .set_position(na::convert(na::Translation3::new(
                0.0,
                PLANET_RADIUS as f64 + BALL_RADIUS * 2.0,
                0.0,
            )));
        world.update();
        assert_eq!(world.contact_pairs(true).count(), 0);

        world
            .get_mut(ball)
            .unwrap()
            .set_position(na::convert(na::Translation3::new(
                -1.0,
                PLANET_RADIUS as f64,
                0.0,
            )));
        world.update();
        assert!(world.contact_pairs(true).count() > 0);

        for i in 1..10 {
            use std::f64;
            let rot = na::UnitQuaternion::from_axis_angle(
                &na::Vector3::z_axis(),
                (i as f64 / 1000.0) * f64::consts::PI * 1e-4,
            );
            let vec = rot * na::Vector3::new(0.0, PLANET_RADIUS as f64, 0.0);
            world
                .get_mut(ball)
                .unwrap()
                .set_position(na::convert(na::Translation3::from(vec)));
            world.update();
            assert!(world.contact_pairs(true).count() > 0);
        }
    }

    // Ensure absence of a collision hole arising from mistakenly considering chunk centers *not* to
    // be offset by 0.5 / face_resolution from edges cubemap faces.
    #[test]
    fn coordinate_center_regression() {
        let mut world = CollisionWorld::new(0.01);
        world.set_narrow_phase(NarrowPhase::new(
            Box::new(PlanetDispatcher::new(DefaultContactDispatcher::new())),
            Box::new(DefaultProximityDispatcher::new()),
        ));

        const PLANET_RADIUS: f32 = 6371e3;
        const BALL_RADIUS: f64 = 50.0;

        world.add(
            na::Isometry3::identity(),
            ShapeHandle::new(Planet::new(
                Arc::new(FlatTerrain::new(2u32.pow(12))),
                32,
                PLANET_RADIUS,
                17,
            )),
            CollisionGroups::new(),
            GeometricQueryType::Contacts(0.01, 0.01),
            0,
        );
        let coords = na::Vector3::<f64>::new(-5_195_083.148, 3_582_099.812, -877_091.267)
            .normalize()
            * PLANET_RADIUS as f64;
        let (_ball, _) = world.add(
            na::convert(na::Translation3::from(coords)),
            ShapeHandle::new(Ball::new(BALL_RADIUS)),
            CollisionGroups::new(),
            GeometricQueryType::Contacts(0.01, 0.01),
            0,
        );

        world.update();
        assert!(world.contact_pairs(true).count() > 0);
    }
}
