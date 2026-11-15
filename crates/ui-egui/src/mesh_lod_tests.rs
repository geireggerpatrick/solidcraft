use super::*;
use solidcraft_engine::geom::prelude::*;

/// A sphere mesh of about `2 n²` triangles, each corner its own point.
fn sphere(n: u32) -> Arc<Mesh> {
    use solidcore_record::Vec3;
    let at = |i: u32, j: u32| {
        let (t, p) = (std::f64::consts::PI * f64::from(i) / f64::from(n), std::f64::consts::TAU * f64::from(j) / f64::from(2 * n));
        let (st, sp, cp) =
            (if i == 0 || i == n { 0.0 } else { t.sin() }, if j == 2 * n { 0.0 } else { p.sin() }, if j == 2 * n { 1.0 } else { p.cos() });
        Vec3::new(50.0 * st * cp, 50.0 * st * sp, 50.0 * t.cos())
    };
    let mut m = Mesh::default();
    for i in 0..n {
        for j in 0..2 * n {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            for [x, y, z] in [[a, c, d], [a, b, c]] {
                let k = m.positions.len() as u32;
                let nrm = (y - x).cross(z - x).normalized().unwrap_or(Vec3::Z);
                m.positions.extend([x, y, z]);
                m.normals.extend([nrm; 3]);
                m.triangles.push([k, k + 1, k + 2]);
            }
        }
    }
    Arc::new(m)
}

/// A large mesh's levels are made once, on a worker: asked for, `None` comes back at once and
/// the work starts; when done, the generation moves on and the levels come back, coarser and
/// coarser, each with its error. A small mesh gets none.
#[test]
fn a_large_meshs_levels_are_made_once_on_a_worker() {
    let lods = MeshLods::default();
    assert!(lods.levels(&sphere(20)).is_none(), "small: none");
    assert!(!lods.busy());
    let mesh = sphere(110);
    assert!(mesh.triangles.len() >= MIN_TRIANGLES);
    let before = lods.generation();
    assert!(lods.levels(&mesh).is_none(), "not made in the asking frame");
    let began = std::time::Instant::now();
    let levels = loop {
        if let Some(l) = lods.levels(&mesh) {
            break l;
        }
        assert!(began.elapsed() < std::time::Duration::from_secs(60), "made on a worker");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    assert_ne!(lods.generation(), before);
    assert!(!lods.busy());
    // A quarter and a sixteenth (a sixty-fourth would be under `MIN_LEVEL`).
    assert_eq!(levels.len(), 2, "{} triangles", mesh.triangles.len());
    let mut last = (mesh.triangles.len(), 0.0f32);
    for (tris, error) in levels.iter() {
        let n = tris.indices.len() / 3;
        assert!(n * 3 <= last.0, "{n} after {}", last.0);
        assert!(*error >= last.1 && *error < 2.0, "error {error}");
        assert!(tris.flat);
        last = (n, *error);
    }
    assert!(Arc::ptr_eq(&levels, &lods.levels(&mesh).unwrap()), "made once");
    // Let go when no longer drawn.
    lods.retain(&HashSet::new());
    assert!(lods.levels(&mesh).is_none(), "made again when asked again");
}

/// A smoothly shaded mesh has smoothly shaded levels: on a sphere the corners that meet in a
/// point of a level are lit alike, away from the centre (within eight degrees on the coarsest
/// level), so no triangle shows as a facet.
#[test]
fn a_smooth_meshs_levels_are_shaded_smoothly() {
    let mut mesh = (*sphere(110)).clone();
    for (n, p) in mesh.normals.iter_mut().zip(&mesh.positions) {
        *n = p.normalized().unwrap();
    }
    let levels = make(&mesh);
    assert!(!levels.is_empty());
    for (tris, _) in &levels {
        assert!(!tris.flat);
        let mut worst = 1.0f32;
        let mut lit = std::collections::HashMap::new();
        for (p, n) in tris.corners().flatten() {
            let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            worst = worst.min((p[0] * n[0] + p[1] * n[1] + p[2] * n[2]) / length);
            let first = *lit.entry(p.map(f32::to_bits)).or_insert(n);
            assert!((0..3).all(|k| (first[k] - n[k]).abs() < 1e-4), "one point lit two ways: {first:?} and {n:?}");
        }
        assert!(worst > 8.0f32.to_radians().cos(), "a corner's normal is {} degrees off", worst.acos().to_degrees());
    }
    // A mesh of flat triangles keeps flat levels.
    assert!(make(&sphere(110)).iter().all(|(tris, _)| tris.flat));
}
