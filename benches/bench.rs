use criterion::{criterion_group, criterion_main, Criterion};

use planetmap::*;

criterion_group!(benches, cache);
criterion_main!(benches);

fn cache(c: &mut Criterion) {
    let mut mgr = CacheManager::with_capacity(2048);
    let state = mgr.update(&[na::Point3::from(na::Vector3::z())]);
    resolve_transfers(&mut mgr, &state.transfer);
    c.bench_function("best-case update", move |b| {
        b.iter(|| mgr.update(&[na::Point3::from(na::Vector3::z())]));
    });

    let mut mgr = CacheManager::with_capacity(2048);
    let state = mgr.update(&[na::Point3::from(na::Vector3::z())]);
    resolve_transfers(&mut mgr, &state.transfer);
    let diagonal1 = na::Point3::from(-na::Vector3::new(1.0, 1.0, 1.0).normalize());
    let diagonal2 = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
    c.bench_function("worst-case update", move |b| {
        let mut which = true;
        b.iter(|| {
            let state = if which {
                mgr.update(&[diagonal1])
            } else {
                mgr.update(&[diagonal2])
            };
            resolve_transfers(&mut mgr, &state.transfer);
            which = !which;
        });
    });
}

fn resolve_transfers(mgr: &mut CacheManager, transfers: &[Chunk]) {
    for &transfer in transfers {
        let slot = mgr.allocate(transfer).unwrap();
        mgr.release(slot);
    }
}
