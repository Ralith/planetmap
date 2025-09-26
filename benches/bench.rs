use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "parry")]
use parry3d_f64::{
    query::{PointQuery, QueryDispatcher, Ray, RayCast},
    shape::Ball,
};

#[cfg(feature = "parry")]
use planetmap::parry::{FlatTerrain, Planet, PlanetDispatcher};
use planetmap::*;

#[cfg(feature = "parry")]
criterion_group!(benches, cache, face_coords, collision);
#[cfg(not(feature = "parry"))]
criterion_group!(benches, cache, face_coords);
criterion_main!(benches);

fn face_coords(c: &mut Criterion) {
    c.bench_function("get vector coordinates on face", |b| {
        b.iter(|| {
            let v = na::Vector3::new(0.42, 0.17, 0.12);
            std::hint::black_box(cubemap::Face::coords(&v));
        });
    });
}

fn cache(c: &mut Criterion) {
    let mut mgr = CacheManager::new(2048, Default::default());
    let mut transfers = Vec::with_capacity(2048);
    mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
    resolve_transfers(&mut mgr, &mut transfers);
    c.bench_function("best-case update", move |b| {
        let mut transfers = Vec::with_capacity(2048);
        b.iter(|| mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers));
    });

    let mut mgr = CacheManager::new(2048, Default::default());
    mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
    resolve_transfers(&mut mgr, &mut transfers);
    let diagonal1 = na::Point3::from(-na::Vector3::new(1.0, 1.0, 1.0).normalize());
    let diagonal2 = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
    c.bench_function("worst-case update", move |b| {
        let mut which = true;
        b.iter(|| {
            if which {
                mgr.update(&[diagonal1], &mut transfers);
            } else {
                mgr.update(&[diagonal2], &mut transfers);
            }
            resolve_transfers(&mut mgr, &mut transfers);
            which = !which;
        });
    });
}

fn resolve_transfers(mgr: &mut CacheManager, transfers: &mut Vec<Chunk>) {
    for &mut transfer in transfers {
        let slot = mgr.allocate(transfer).unwrap();
        mgr.release(slot);
    }
}

#[cfg(feature = "parry")]
fn collision(c: &mut Criterion) {
    use std::sync::Arc;

    const PLANET_RADIUS: f64 = 6371e3;
    let ball = Ball { radius: 1.0 };
    let planet = Planet::new(
        Arc::new(FlatTerrain::new(2u32.pow(12), 17)),
        32,
        PLANET_RADIUS,
    );

    c.bench_function("intersect", |b| {
        b.iter(|| {
            assert!(PlanetDispatcher
                .intersection_test(
                    &na::Isometry3::translation(PLANET_RADIUS, 0.0, 0.0),
                    &planet,
                    &ball,
                )
                .unwrap());
        });
    });

    c.bench_function("short raycast", |b| {
        b.iter(|| {
            planet.cast_local_ray(
                &Ray {
                    origin: na::Point3::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                    dir: na::Vector3::y(),
                },
                1e1,
                true,
            );
        });
    });

    c.bench_function("long raycast", |b| {
        b.iter(|| {
            planet.cast_local_ray(
                &Ray {
                    origin: na::Point3::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                    dir: na::Vector3::y(),
                },
                1e3,
                true,
            );
        });
    });

    c.bench_function("project point", |b| {
        b.iter(|| {
            planet.project_point(
                &na::Isometry3::identity(),
                &na::Point3::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                true,
            )
        });
    });
}
