use std::sync::{Arc, Mutex};

use hashbrown::hash_map;
use hashbrown::HashMap;
use lru::LruCache;
use na::Real;
use ncollide3d::{
    bounding_volume::{BoundingSphere, HasBoundingVolume, AABB},
    narrow_phase::{ContactAlgorithm, ContactDispatcher, ContactManifoldGenerator},
    query::{
        Contact, ContactKinematic, ContactManifold, ContactPrediction, ContactPreprocessor,
        PointProjection, PointQuery, Ray, RayCast, RayIntersection,
    },
    shape::{FeatureId, Shape, Triangle},
    utils::{IdAllocator, IsometryOps},
};

use crate::chunk::Coords;

/// Height data source for `Planet`
pub trait Terrain: Send + Sync + 'static {
    /// Generate a `resolution * resolution` grid of heights wrt. sea level
    fn sample(&self, resolution: u32, coords: &Coords, out: &mut [f32]);
    /// The maximum value that will ever be written by `sample`
    fn max_height(&self) -> f32;
    /// The minimum value that will ever be written by `sample`
    fn min_height(&self) -> f32;
}

#[derive(Debug, Copy, Clone)]
pub struct FlatTerrain;

impl Terrain for FlatTerrain {
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
pub struct Planet {
    terrain: Arc<dyn Terrain>,
    radius: f32,
    face_resolution: u32,
    chunk_resolution: u32,
    // Could preallocate an arena for height samples
    cache: Mutex<LruCache<Coords, ChunkData>>,
}

impl Planet {
    /// Construct a new collision shape for a radial heightmap defined by `terrain`
    ///
    /// `terrain` - source of height samples
    /// `cache_size` - maximum number of chunks of height data to keep in memory
    /// `radius` - distance from origin of points with height 0
    /// `face_resolution` - number of chunks along the edge of a cubemap face
    /// `chunk_resolution` - number of heightfield samples along the edge of a chunk
    pub fn new(
        terrain: Arc<dyn Terrain>,
        cache_size: usize,
        radius: f32,
        face_resolution: u32,
        chunk_resolution: u32,
    ) -> Self {
        assert!(chunk_resolution > 1);
        Self {
            terrain,
            radius,
            face_resolution,
            chunk_resolution,
            cache: Mutex::new(LruCache::new(cache_size)),
        }
    }

    fn max_radius(&self) -> f64 {
        self.radius as f64 + self.terrain.max_height() as f64
    }
    fn min_radius(&self) -> f64 {
        self.radius as f64 + self.terrain.max_height() as f64
    }

    fn sample(&self, coords: &Coords) -> Box<[f32]> {
        let mut samples =
            vec![0.0; self.chunk_resolution as usize * self.chunk_resolution as usize]
                .into_boxed_slice();
        self.terrain
            .sample(self.chunk_resolution, &coords, &mut samples[..]);
        samples
    }

    fn feature_id(&self, coords: &Coords, triangle: usize, tri_feature: FeatureId) -> FeatureId {
        use FeatureId::*;
        Unknown                 // TODO
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

impl<N: Real> HasBoundingVolume<N, BoundingSphere<N>> for Planet {
    fn bounding_volume(&self, m: &na::Isometry3<N>) -> BoundingSphere<N> {
        BoundingSphere::new(m * na::Point3::origin(), na::convert(self.max_radius()))
    }
}

impl<N: Real> HasBoundingVolume<N, AABB<N>> for Planet {
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
        let coords = Coords::from_vector(self.face_resolution, &na::convert(local.coords));
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
        // TODO: Check neighborhood?
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
        // FIXME
        false
    }

    // #[inline]
    // fn subshape_containing_feature(&self, _id: FeatureId) -> usize {
    //     // FIXME
    //     0
    // }

    #[inline]
    fn as_point_query(&self) -> Option<&PointQuery<f64>> {
        Some(self)
    }
}

impl RayCast<f64> for Planet {
    fn toi_and_normal_with_ray(
        &self,
        _m: &na::Isometry3<f64>,
        _ray: &Ray<f64>,
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
    index: u32,
}

impl<'a> ChunkTriangles<'a> {
    pub fn new(planet: &'a Planet, coords: Coords, samples: &'a [f32]) -> Self {
        Self {
            planet,
            samples,
            coords,
            index: 0,
        }
    }

    fn vertex(&self, x: u32, y: u32) -> na::Point3<f64> {
        let height = self.samples[(y * self.planet.chunk_resolution + x) as usize];
        let unit_coords = na::Point2::new(
            x as f64 / self.planet.chunk_resolution as f64,
            y as f64 / self.planet.chunk_resolution as f64,
        );
        let dir = self
            .coords
            .direction(self.planet.face_resolution, &unit_coords);
        na::Point3::from(dir.into_inner() * (self.planet.radius as f64 + height as f64))
    }

    fn get(&self, index: u32) -> Triangle<f64> {
        let quad_resolution = self.planet.chunk_resolution - 1;

        let quad_index = index >> 1;
        let y = quad_index / quad_resolution;
        let x = quad_index % quad_resolution;
        let left = (index & 1) == 0;
        let p0 = self.vertex(x, y);
        let p1 = self.vertex(x + 1, y);
        let p2 = self.vertex(x + 1, y + 1);
        let p3 = self.vertex(x, y + 1);
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
        let quad_resolution = self.planet.chunk_resolution - 1;
        if self.index == quad_resolution * quad_resolution * 2 {
            return None;
        }
        let tri = self.get(self.index);
        self.index += 1;
        Some(tri)
    }
}

pub struct PlanetManifoldGenerator {
    flip: bool,
    state: HashMap<(Coords, usize), TriangleData>,
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

    fn run(
        &mut self,
        dispatcher: &ContactDispatcher<f64>,
        ma: &na::Isometry3<f64>,
        planet: &Planet,
        proc1: Option<&ContactPreprocessor<f64>>,
        mb: &na::Isometry3<f64>,
        other: &dyn Shape<f64>,
        proc2: Option<&ContactPreprocessor<f64>>,
        prediction: &ContactPrediction<f64>,
        id_alloc: &mut IdAllocator,
        manifold: &mut ContactManifold<f64>,
    ) {
        self.color ^= true;
        let color = self.color;

        let bounds = other.bounding_sphere(mb);
        let dir = ma.inverse_transform_point(bounds.center()).coords;
        let distance = dir.norm();
        // Future work: degenerate to ball for bounding spheres significantly larger than altitude range?
        let cache = &mut *planet.cache.lock().unwrap();
        for coords in Coords::neighborhood(
            planet.face_resolution,
            na::convert(dir),
            bounds.radius().atan2(dir.norm()) as f32,
        ) {
            let data = if let Some(x) = cache.get(&coords) {
                x
            } else {
                cache.put(coords, ChunkData::new(planet.sample(&coords)));
                cache.get(&coords).unwrap()
            };
            if planet.radius as f64 + data.max as f64 + bounds.radius() < distance {
                // Short-circuit if `other` is way above this chunk
                continue;
            }
            for (i, triangle) in ChunkTriangles::new(planet, coords, &data.samples).enumerate() {
                // Could short-circuit here using dot(coord, dir)
                let tri = match self.state.entry((coords, i)) {
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
                            e.insert(TriangleData { algo, color })
                        } else {
                            return;
                        }
                    }
                };
                let proc1 = TriangleContactPreprocessor {
                    planet,
                    outer: proc1,
                    coords,
                    triangle: i,
                };
                if !self.flip {
                    tri.algo.generate_contacts(
                        dispatcher, ma, &triangle, Some(&proc1), mb, other, proc2, prediction, id_alloc,
                        manifold,
                    );
                } else {
                    tri.algo.generate_contacts(
                        dispatcher, mb, other, proc2, ma, &triangle, Some(&proc1), prediction, id_alloc,
                        manifold,
                    );
                }
            }
        }

        self.state.retain(|_, x| x.color == color);
    }
}

impl ContactManifoldGenerator<f64> for PlanetManifoldGenerator {
    fn generate_contacts(
        &mut self,
        d: &ContactDispatcher<f64>,
        ma: &na::Isometry3<f64>,
        a: &dyn Shape<f64>,
        proc1: Option<&ContactPreprocessor<f64>>,
        mb: &na::Isometry3<f64>,
        b: &dyn Shape<f64>,
        proc2: Option<&ContactPreprocessor<f64>>,
        prediction: &ContactPrediction<f64>,
        id_alloc: &mut IdAllocator,
        manifold: &mut ContactManifold<f64>,
    ) -> bool {
        if !self.flip {
            if let Some(p) = a.as_shape::<Planet>() {
                self.run(
                    d, ma, p, proc1, mb, b, proc2, prediction, id_alloc, manifold,
                );
                return true;
            }
        } else {
            if let Some(p) = b.as_shape::<Planet>() {
                self.run(
                    d, mb, p, proc2, ma, a, proc1, prediction, id_alloc, manifold,
                );
                return true;
            }
        }
        false
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

struct TriangleData {
    algo: ContactAlgorithm<f64>,
    color: bool,
}

pub struct PlanetDispatcher<T> {
    inner: T,
}

impl<T> PlanetDispatcher<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: ContactDispatcher<f64>> ContactDispatcher<f64> for PlanetDispatcher<T> {
    fn get_contact_algorithm(
        &self,
        a: &Shape<f64>,
        b: &Shape<f64>,
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

struct TriangleContactPreprocessor<'a, N: Real> {
    planet: &'a Planet,
    outer: Option<&'a ContactPreprocessor<N>>,
    coords: Coords,
    triangle: usize,
}

impl<N: Real> ContactPreprocessor<N> for TriangleContactPreprocessor<'_, N> {
    fn process_contact(
        &self,
        contact: &mut Contact<N>,
        kinematic: &mut ContactKinematic<N>,
        is_first: bool,
    ) -> bool {
        if is_first {
            kinematic.set_feature1(self.planet.feature_id(&self.coords, self.triangle, kinematic.feature1()));
        } else {
            kinematic.set_feature2(self.planet.feature_id(&self.coords, self.triangle, kinematic.feature2()));
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
    use ncollide3d::{
        narrow_phase::{DefaultContactDispatcher, DefaultProximityDispatcher, NarrowPhase},
        shape::{Ball, ShapeHandle},
        world::{CollisionGroups, CollisionWorld, GeometricQueryType},
    };

    use crate::chunk::Face;

    use super::*;

    #[test]
    fn triangles() {
        const CHUNK_RESOLUTION: u32 = 5;
        let planet = Planet::new(Arc::new(FlatTerrain), 32, 1.0, 4, CHUNK_RESOLUTION);
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::PZ,
        };
        let samples = planet.sample(&coords);
        let iter = ChunkTriangles::new(&planet, coords, &samples[..]);
        const QUAD_RESOLUTION: u32 = CHUNK_RESOLUTION - 1;
        assert_eq!(
            iter.clone().count() as u32,
            QUAD_RESOLUTION * QUAD_RESOLUTION * 2
        );
        for tri in iter.clone() {
            for point in tri.vertices() {
                assert!(point.x < 0.0);
                assert!(point.y < 0.0);
                assert!(point.z > 0.0);
            }
        }
    }
}
