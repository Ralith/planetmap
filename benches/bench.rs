use criterion::{criterion_group, criterion_main, Criterion};

use planetmap::*;

criterion_group!(benches, cache);
criterion_main!(benches);

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
