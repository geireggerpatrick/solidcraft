//! A generative design study: faces that are held, faces that are loaded, the bodies to build
//! around, and the structure that comes out as a closed triangle mesh.

use crate::voxel::{self as gv, Grid, HalfSpace, Mesh, Outcome, Problem, Settings, V3};

/// The most cells a design space may have.
pub const MAX_CELLS: usize = 3_000_000;
/// How far the structure reaches into the body behind a flat picked face, so the two overlap
/// and a slicer prints them as one part.
pub const OVERLAP: f64 = 0.05;

/// The triangles of one picked face, facing outward.
pub type FaceTris = Vec<[V3; 3]>;

#[derive(Clone, Debug)]
pub struct Study {
    /// Faces that do not move.
    pub held: Vec<FaceTris>,
    /// Faces the force acts on.
    pub load: Vec<FaceTris>,
    /// The force in newtons.
    pub force: V3,
    /// The body to carve the structure out of; `None`: the box around the picked faces.
    pub space: Option<Mesh>,
    /// Closed bodies the structure stays out of. With no `space`, the bodies of the picked
    /// faces belong here.
    pub obstacles: Vec<Mesh>,
    /// How far the design box reaches beyond the picked faces, in mm.
    pub margin: f64,
    /// How many cells to divide the design space into.
    pub voxels: f64,
    /// `min_wall` 0 means a thinnest member that suits the size of the design space, and
    /// `max_deflection` 0 a limit that does.
    pub settings: Settings,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub mesh: Mesh,
    pub outcome: Outcome,
    /// The cell size in mm.
    pub cell: f64,
    pub cells: usize,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: V3) -> Option<V3> {
    let n = dot(a, a).sqrt();
    (n > 1e-12).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

/// The plane of a face whose triangles all lie in one, facing one way: a point and the normal.
pub fn plane_of(tris: &[[V3; 3]]) -> Option<HalfSpace> {
    let mut plane: Option<HalfSpace> = None;
    for t in tris {
        let Some(n) = unit(cross(sub(t[1], t[0]), sub(t[2], t[0]))) else { continue };
        match plane {
            None => plane = Some((t[0], n)),
            Some((_, n0)) if dot(n, n0) < 1.0 - 1e-6 => return None,
            Some(_) => {}
        }
    }
    let (point, normal) = plane?;
    tris.iter().flatten().all(|p| dot(sub(*p, point), normal).abs() < 1e-4).then_some((point, normal))
}

/// The pieces of a mesh that reach one of the planes.
fn standing(mesh: Mesh, planes: &[HalfSpace]) -> Mesh {
    let mut parent: Vec<usize> = (0..mesh.pos.len()).collect();
    fn root(parent: &mut [usize], mut v: usize) -> usize {
        while parent[v] != v {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        v
    }
    for t in &mesh.tri {
        let a = root(&mut parent, t[0]);
        for &v in &t[1..] {
            let b = root(&mut parent, v);
            parent[b] = a;
        }
    }
    let mut stands = vec![false; mesh.pos.len()];
    for v in 0..mesh.pos.len() {
        if planes.iter().any(|(point, normal)| dot(sub(mesh.pos[v], *point), *normal).abs() < 2.0 * OVERLAP + 0.02) {
            let r = root(&mut parent, v);
            stands[r] = true;
        }
    }
    let mut new_id = vec![usize::MAX; mesh.pos.len()];
    let mut out = Mesh::default();
    for t in &mesh.tri {
        if !stands[root(&mut parent, t[0])] {
            continue;
        }
        out.tri.push(t.map(|v| {
            if new_id[v] == usize::MAX {
                new_id[v] = out.pos.len();
                out.pos.push(mesh.pos[v]);
            }
            new_id[v]
        }));
    }
    out
}

struct Face<'a> {
    tris: &'a [[V3; 3]],
    plane: Option<HalfSpace>,
}

fn faces(list: &[FaceTris], flat: bool) -> Vec<Face<'_>> {
    list.iter().map(|tris| Face { tris, plane: if flat { plane_of(tris) } else { None } }).collect()
}

/// Runs the study. `time_left` is asked between steps; when it says no, the best shape so far
/// is finished.
pub fn run(study: &Study, time_left: &mut dyn FnMut() -> bool) -> Result<Report, String> {
    if study.held.is_empty() || study.load.is_empty() || study.held.iter().chain(&study.load).any(Vec::is_empty) {
        return Err("give at least one held face and one loaded face".into());
    }
    if !(study.voxels >= 1.0 && study.voxels <= MAX_CELLS as f64 && study.margin >= 0.0 && study.margin.is_finite()) {
        return Err("the cell count or the margin is out of range".into());
    }
    let boxed = study.space.is_none();
    let held = faces(&study.held, boxed);
    let load = faces(&study.load, boxed);

    // The design space and its grid.
    let (mut grid, clip, within): (Grid, Option<(V3, V3)>, Vec<HalfSpace>) = match &study.space {
        Some(mesh) => {
            let volume = gv::mesh_volume(mesh).abs();
            if !(volume > 1e-6) {
                return Err("the design space must be a closed body".into());
            }
            let h = (volume / study.voxels).cbrt().max(0.05);
            let (lo, hi) = mesh.bbox();
            let cells = (0..3).map(|a| ((hi[a] - lo[a]) / h).ceil().max(1.0)).product::<f64>();
            if !(cells <= MAX_CELLS as f64) {
                return Err("the design space is too thin for this many cells; give fewer".into());
            }
            (gv::voxelize(mesh, h), Some((lo, hi)), Vec::new())
        }
        None => {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for p in held.iter().chain(&load).flat_map(|f| f.tris.iter().flatten()) {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a] - study.margin);
                    hi[a] = hi[a].max(p[a] + study.margin);
                }
            }
            // Faces in one plane span a flat box: give it some depth to build an arch in.
            let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
            if !(longest > 1e-3 && longest < 1e5) {
                return Err("the picked faces span no space to build in".into());
            }
            for a in 0..3 {
                let grow = (0.2 * longest - (hi[a] - lo[a])) / 2.0;
                if grow > 0.0 {
                    lo[a] -= grow;
                    hi[a] += grow;
                }
            }
            let volume = (0..3).map(|a| hi[a] - lo[a]).product::<f64>();
            let h = (volume / study.voxels).cbrt().max(0.05);
            // One more layer all around: behind a picked face it lies in the face's body, and
            // the structure is cut flat on the face there.
            let cells = (0..3).map(|a| ((hi[a] - lo[a]) / h).ceil() + 2.0).product::<f64>();
            if !(cells <= MAX_CELLS as f64) {
                return Err("too many cells; give fewer".into());
            }
            let grid = gv::box_grid(lo.map(|c| c - h), hi.map(|c| c + h), h);
            let within = held.iter().chain(&load).filter_map(|f| f.plane).map(|(pt, n)| ([0, 1, 2].map(|a| pt[a] - OVERLAP * n[a]), n)).collect();
            (grid, None, within)
        }
    };
    let h = grid.h;
    for mesh in &study.obstacles {
        let mask = gv::mesh_mask(&grid, mesh);
        for (inside, blocked) in grid.inside.iter_mut().zip(mask) {
            if blocked {
                *inside = false;
            }
        }
    }
    if boxed {
        // The layer just behind a flat picked face belongs to the structure: it is what the
        // cut on the face's plane leaves flat. And nothing is built on the far side of a
        // picked face's plane.
        let all = Grid { inside: vec![true; grid.len()], ..grid.clone() };
        for f in held.iter().chain(&load).filter(|f| f.plane.is_some()) {
            for v in gv::voxels_near(&all, f.tris, 0.9 * h) {
                grid.inside[v] = true;
            }
        }
        for (point, normal) in &within {
            for v in 0..grid.len() {
                if dot(sub(grid.center(v), *point), *normal) < -0.9 * h {
                    grid.inside[v] = false;
                }
            }
        }
    }
    // A flat picked face is held or loaded through the layer of cells just behind it, inside
    // its body: that layer stands for the body, which spreads the force over the face, and is
    // cut off again at the end. So the structure meets the face only where it needs to. A
    // curved face is held or loaded through the cells in front of it, which stay.
    let mark = |faces: &[Face]| -> (Vec<bool>, Vec<Vec<usize>>) {
        let mut set = vec![false; grid.len()];
        let mut each = Vec::new();
        for f in faces {
            let mut near = gv::voxels_near(&grid, f.tris, 0.9 * h);
            if let Some((point, normal)) = f.plane {
                near.retain(|&v| dot(sub(grid.center(v), point), normal) < 0.0);
            }
            for &v in &near {
                set[v] = true;
            }
            each.push(near);
        }
        (set, each)
    };
    let (fixed, supports) = mark(&held);
    let (loaded, loads) = mark(&load);
    if supports.iter().any(Vec::is_empty) || loads.iter().any(Vec::is_empty) {
        return Err("a picked face has no room to build on; give more cells or a margin".into());
    }
    let both: Vec<bool> = fixed.iter().zip(&loaded).map(|(a, b)| *a || *b).collect();
    let keep = if boxed { both.clone() } else { gv::dilate(&grid, &both, 1.5) };
    let problem = Problem { grid, fixed, load: loaded, keep, supports, force: study.force };
    // With no thinnest member given: 4 % of the design space's longest side, which gives few,
    // sturdy members rather than many hair-thin ones.
    let mut settings = study.settings.clone();
    if !(settings.min_wall > 0.0) {
        let longest = (0..3).map(|a| problem.grid.n[a] as f64 * h).fold(0.0, f64::max);
        settings.min_wall = 0.04 * longest;
    }
    // With no limit given, the loaded faces may move by 1/400 of the longest side: a shape that
    // only just does not break is too soft to be of use.
    if !(settings.max_deflection > 0.0) {
        let longest = (0..3).map(|a| problem.grid.n[a] as f64 * h).fold(0.0, f64::max);
        settings.max_deflection = longest / 400.0;
    }
    let outcome = gv::generate(&problem, &settings, time_left)?;
    let g = &problem.grid;
    let mut field = outcome.field.clone();
    // The layer behind a flat face takes the shape of what stands on it, so the structure
    // runs straight into the cut on the face's plane.
    for f in held.iter().chain(&load) {
        let Some((point, normal)) = f.plane else { continue };
        for v in gv::voxels_near(g, f.tris, 0.9 * h) {
            let c = g.center(v);
            let depth = dot(sub(c, point), normal);
            if depth >= 0.0 {
                continue;
            }
            let front = [0, 1, 2].map(|a| c[a] + (0.5 * h - depth) * normal[a]);
            field[v] = g.at(front).filter(|&w| g.inside[w]).map_or(0.0, |w| outcome.field[w]);
        }
    }
    let up = if g.len() > 400_000 { 1 } else { 2 };
    let mesh = gv::surface(g, &field, up, 10, &[], clip, &within);
    // Loose crumbs carry nothing: only pieces that stand on a held face stay.
    let held_planes: Vec<HalfSpace> = held.iter().filter_map(|f| f.plane).collect();
    let mesh = if held_planes.len() == held.len() { standing(mesh, &held_planes) } else { mesh };
    if mesh.tri.is_empty() {
        return Err("no structure came out; the faces may be too far apart for this margin".into());
    }
    Ok(Report { mesh, outcome, cell: h, cells: g.len() })
}
