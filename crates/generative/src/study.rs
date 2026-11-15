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

/// What acts on loaded faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// A force in newtons, shared evenly by the faces.
    Force(V3),
    /// A moment in newton millimetres about the middle of the faces: it turns them about the
    /// vector, the way a right-handed screw along it turns.
    Moment(V3),
    /// A pressure in MPa on the faces, against their outward normals.
    Pressure(f64),
}

#[derive(Clone, Debug)]
pub struct Study {
    /// Faces that do not move.
    pub held: Vec<FaceTris>,
    /// Faces that are held along their normal only: they slide in their own surface.
    pub slide: Vec<FaceTris>,
    /// Round faces on a pin: held across and along the pin, free to turn about it.
    pub pin: Vec<FaceTris>,
    /// Faces the first load acts on.
    pub load: Vec<FaceTris>,
    /// The first load.
    pub action: Action,
    /// Further loads: faces, what acts on them, and the load case. Case 0 acts at the same
    /// time as the first load; loads with the same other number act together in a case of
    /// their own, and the structure carries every case.
    pub more: Vec<(Vec<FaceTris>, Action, usize)>,
    /// The body to carve the structure out of; `None`: the box around the picked faces.
    pub space: Option<Mesh>,
    /// Closed bodies the structure stays out of. With no `space`, the bodies of the picked
    /// faces belong here.
    pub obstacles: Vec<Mesh>,
    /// Further closed bodies the structure stays out of, and `clearance` mm away from.
    pub avoid: Vec<Mesh>,
    pub clearance: f64,
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

/// A plane: a point on it and its unit normal.
type Plane = (V3, V3);

/// The plane of a face whose triangles all lie in one, facing one way: a point and the normal.
pub fn plane_of(tris: &[[V3; 3]]) -> Option<Plane> {
    let mut plane: Option<Plane> = None;
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

/// The pieces of a mesh that reach one of the planes, and the hollows inside: a closed surface
/// that faces inwards is the wall of a hollow, which a lattice has behind its skin.
fn standing(mesh: Mesh, planes: &[Plane]) -> Mesh {
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
    let mut volume = vec![0.0; mesh.pos.len()];
    for t in &mesh.tri {
        volume[root(&mut parent, t[0])] += dot(mesh.pos[t[0]], cross(mesh.pos[t[1]], mesh.pos[t[2]]));
    }
    let mut new_id = vec![usize::MAX; mesh.pos.len()];
    let mut out = Mesh::default();
    for t in &mesh.tri {
        let piece = root(&mut parent, t[0]);
        if !stands[piece] && volume[piece] >= 0.0 {
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
    plane: Option<Plane>,
}

/// How a face is held.
#[derive(Clone, Copy, PartialEq)]
enum Hold {
    Fixed,
    Slide,
    Pin,
}

/// The outward unit normal of the triangle nearest to `p`.
fn normal_near(tris: &[[V3; 3]], p: V3) -> Option<V3> {
    let nearest = tris.iter().min_by(|a, b| gv::point_triangle_distance(p, a).total_cmp(&gv::point_triangle_distance(p, b)))?;
    unit(cross(sub(nearest[1], nearest[0]), sub(nearest[2], nearest[0])))
}

/// The axis of a round face, as a point on it and its direction: every normal of the face is
/// square to it and points away from it. `None` for a face that is not round.
fn pin_axis(tris: &[[V3; 3]]) -> Option<(V3, V3)> {
    let normals: Vec<(V3, V3, f64)> = tris
        .iter()
        .filter_map(|t| {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let middle = [0, 1, 2].map(|a| (t[0][a] + t[1][a] + t[2][a]) / 3.0);
            Some((middle, unit(n)?, 0.5 * dot(n, n).sqrt()))
        })
        .collect();
    let first = normals.iter().max_by(|a, b| a.2.total_cmp(&b.2))?.1;
    // The normal that turns furthest from the first one spans the section with it.
    let across = |n: &V3| {
        let c = cross(first, *n);
        dot(c, c)
    };
    let second = normals.iter().max_by(|a, b| across(&a.1).total_cmp(&across(&b.1)))?.1;
    if across(&second) < 0.04 {
        return None;
    }
    let axis = unit(cross(first, second))?;
    if normals.iter().any(|(_, n, _)| dot(*n, axis).abs() > 0.05) {
        return None;
    }
    // In the section, the point nearest to the lines along all the normals.
    let (e1, e2) = (first, cross(axis, first));
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (middle, n, area) in &normals {
        let (q1, q2, n1, n2) = (dot(*middle, e1), dot(*middle, e2), dot(*n, e1), dot(*n, e2));
        let (p11, p12, p22) = (1.0 - n1 * n1, -n1 * n2, 1.0 - n2 * n2);
        a11 += area * p11;
        a12 += area * p12;
        a22 += area * p22;
        b1 += area * (p11 * q1 + p12 * q2);
        b2 += area * (p12 * q1 + p22 * q2);
    }
    let det = a11 * a22 - a12 * a12;
    if !(det.abs() > 1e-9 * (a11 + a22).powi(2)) {
        return None;
    }
    let (c1, c2) = ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det);
    Some(([0, 1, 2].map(|a| c1 * e1[a] + c2 * e2[a]), axis))
}

fn faces(list: &[FaceTris], flat: bool) -> Vec<Face<'_>> {
    list.iter().map(|tris| Face { tris, plane: if flat { plane_of(tris) } else { None } }).collect()
}

/// One resolution of a study: the problem on its grid, and how its surface is cut.
struct Level {
    problem: Problem,
    clip: Option<(V3, V3)>,
    within: Vec<HalfSpace>,
    /// For each flat picked face its plane and the voxels behind it, which stand for the
    /// face's body.
    slabs: Vec<(Plane, Vec<usize>)>,
    /// The radius of the fillets into those bodies, in voxels.
    round: f64,
}

/// The box a study builds in, and the volume of its design space.
fn extent(study: &Study, held: &[Face], load: &[Face]) -> Result<(V3, V3, f64), String> {
    if let Some(mesh) = &study.space {
        let volume = gv::mesh_volume(mesh).abs();
        if !(volume > 1e-6) {
            return Err("the design space must be a closed body".into());
        }
        let (lo, hi) = mesh.bbox();
        return Ok((lo, hi, volume));
    }
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in held.iter().chain(load).flat_map(|f| f.tris.iter().flatten()) {
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
        let grow = (0.3 * longest - (hi[a] - lo[a])) / 2.0;
        if grow > 0.0 {
            lo[a] -= grow;
            hi[a] += grow;
        }
    }
    Ok((lo, hi, (0..3).map(|a| hi[a] - lo[a]).product::<f64>()))
}

/// Whether `p` is right in front of another flat picked face than face `me`, no further than
/// `far` from it: the structure stands on that face there.
fn stands_on_other(planes: &[HalfSpace], me: usize, p: V3, far: f64) -> bool {
    planes.iter().enumerate().any(|(j, other)| {
        let depth = other.depth(p);
        j != me && !other.tris.is_empty() && depth > 0.0 && depth < far && other.over(p, 0.0)
    })
}

/// The study's problem on a grid of about `voxels` cells, for members `wall` mm thick. With
/// `band`, a coarser level's grid and the cells around its structure, only the cells there are
/// design space: a finer level works the structure out where the coarser one found it.
/// `load` holds the faces of every load; `more` says which of them each load acts on, and in
/// which load case. `holds` says how each of `held` is held.
fn level(
    study: &Study,
    held: &[Face],
    holds: &[Hold],
    load: &[Face],
    more: &[(std::ops::Range<usize>, Action, usize)],
    voxels: f64,
    wall: f64,
    band: Option<(&Grid, &[bool])>,
) -> Result<Level, String> {
    let boxed = study.space.is_none();
    let (lo, hi, volume) = extent(study, held, load)?;
    let h = (volume / voxels).cbrt().max(0.05);
    // A member runs into a picked face's body with a fillet twice as wide as the thinnest member.
    let round = (2.0 * wall / h).clamp(2.0, 7.0);
    // The design space and its grid.
    let (mut grid, clip, within): (Grid, Option<(V3, V3)>, Vec<HalfSpace>) = match &study.space {
        Some(mesh) => {
            let cells = (0..3).map(|a| ((hi[a] - lo[a]) / h).ceil().max(1.0)).product::<f64>();
            if !(cells <= MAX_CELLS as f64) {
                return Err("the design space is too thin for this many cells; give fewer".into());
            }
            (gv::voxelize(mesh, h), Some((lo, hi)), Vec::new())
        }
        None => {
            // Layers all around the box: behind a picked face they lie in the face's body and
            // stand for it, as deep as a fillet is wide. The structure is cut flat on the face.
            let pad = round.ceil() + 1.0;
            let cells = (0..3).map(|a| ((hi[a] - lo[a]) / h).ceil() + 2.0 * pad).product::<f64>();
            if !(cells <= MAX_CELLS as f64) {
                return Err("too many cells; give fewer".into());
            }
            let mut grid = gv::box_grid(lo.map(|c| c - pad * h), hi.map(|c| c + pad * h), h);
            for v in 0..grid.len() {
                let c = grid.center(v);
                // The box itself, and no cell beyond it: a member along its side is flush
                // with the edge of a picked face there, and does not stand over the edge.
                if (0..3).any(|a| c[a] < lo[a] || c[a] > hi[a]) {
                    grid.inside[v] = false;
                }
            }
            // A flat picked face's plane bounds the structure everywhere, unless another picked
            // face lies behind it (two faces of one body that meet in an edge, a face in the
            // middle of the others): then it bounds the structure around its face only, or
            // it would shut that other face out.
            let within = held
                .iter()
                .chain(load)
                .filter_map(|f| {
                    let (pt, n) = f.plane?;
                    let behind = |p: &V3| dot(sub(*p, pt), n) < -1e-3;
                    let shuts_out = held.iter().chain(load).any(|g| !std::ptr::eq(f, g) && g.tris.iter().flatten().any(behind));
                    let at = [0, 1, 2].map(|a| pt[a] - OVERLAP * n[a]);
                    Some(if shuts_out { HalfSpace::new(at, n, f.tris) } else { HalfSpace::everywhere(at, n) })
                })
                .collect();
            (grid, None, within)
        }
    };
    let mut blocked = vec![false; grid.len()];
    for mesh in &study.obstacles {
        let mask = gv::mesh_mask(&grid, mesh);
        for v in 0..grid.len() {
            if mask[v] {
                grid.inside[v] = false;
                blocked[v] = true;
            }
        }
    }
    for mesh in &study.avoid {
        let mut mask = gv::mesh_mask(&grid, mesh);
        if study.clearance > 0.0 {
            mask = gv::dilate(&grid, &mask, (study.clearance / h).min(8.0));
        }
        for v in 0..grid.len() {
            if mask[v] {
                grid.inside[v] = false;
                blocked[v] = true;
            }
        }
    }
    if let Some((coarse, around)) = band {
        for v in 0..grid.len() {
            if grid.inside[v] && !coarse.at(grid.center(v)).is_some_and(|w| around[w]) {
                grid.inside[v] = false;
            }
        }
    }
    let mut slabs: Vec<(Plane, Vec<usize>)> = Vec::new();
    let mut cover = vec![false; grid.len()];
    // How far beside a flat picked face its plane bounds the structure: two cells further than
    // the fillet around the face.
    let beside_to = (round + 2.0) * h;
    if boxed {
        // Nothing is built on the far side of a picked face's plane. Where the plane bounds the
        // structure around its face only, that is behind the face and beside it as far as the
        // fillet around the face reaches; further away the structure may cross the plane.
        for (i, plane) in within.iter().enumerate() {
            for v in 0..grid.len() {
                let c = grid.center(v);
                if grid.inside[v] && plane.depth(c) < -0.9 * h && gv::plane_holds(&within, i, c, beside_to) {
                    grid.inside[v] = false;
                }
            }
        }
        // The layers behind a flat picked face, in its body, belong to the structure: they are
        // what the cut on the face's plane leaves flat. Beside the face, the layer in front of
        // the plane stays empty: the structure meets the plane on the face only and does not
        // hang over the face's edges. What reaches past an edge further away runs in to the
        // edge at 45 degrees, so a member wider than the face narrows to it and does not end
        // in a step before the plane.
        let deep = (round.ceil() + 0.5) * h;
        let mut on_face = vec![false; grid.len()];
        let mut beside = vec![false; grid.len()];
        for f in held.iter().chain(load) {
            let Some((point, normal)) = f.plane else { continue };
            let area: f64 = f.tris.iter().map(|t| 0.5 * dot(cross(sub(t[1], t[0]), sub(t[2], t[0])), normal).abs()).sum();
            // A small face is covered whole, so the structure grows out of its outline.
            let small = area <= 25.0 * wall * wall;
            // The cover of a small face is a good half member thick, and a cell at least.
            let cover_to = if small { h.max(0.6 * wall) } else { 0.0 };
            let mut behind = Vec::new();
            let mut front = Vec::new();
            let everywhere = within.get(slabs.len()).is_none_or(|plane| plane.tris.is_empty());
            // A face narrower than a cell still gets the cells along it.
            for reach in [0.3 * h, 0.9 * h] {
                behind.clear();
                front.clear();
                for v in 0..grid.len() {
                    let c = grid.center(v);
                    let depth = dot(sub(c, point), normal);
                    if !(depth > -deep && depth < cover_to.max(1.25 * h).max(beside_to)) {
                        continue;
                    }
                    let foot = [0, 1, 2].map(|a| c[a] - depth * normal[a]);
                    let aside = f.tris.iter().map(|t| gv::point_triangle_distance(foot, t)).fold(f64::MAX, f64::min);
                    if aside <= reach && depth < 0.0 {
                        // Not where the structure stands on another face that meets this one
                        // in an edge: there the cells belong to that face's side.
                        if !stands_on_other(&within, slabs.len(), c, 2.0 * beside_to) {
                            behind.push(v);
                        }
                    } else if aside <= reach {
                        if depth < 1.25 * h {
                            on_face[v] = true;
                        }
                        if depth < cover_to {
                            front.push(v);
                        }
                    } else if depth > -0.9 * h
                        && depth < (1.25 * h).max(aside.min(2.0 * beside_to - aside))
                        && (everywhere || aside <= beside_to)
                        && !stands_on_other(&within, slabs.len(), c, 2.0 * beside_to)
                    {
                        beside[v] = true;
                    }
                }
                if !behind.is_empty() {
                    break;
                }
            }
            for v in front {
                cover[v] = !blocked[v];
            }
            slabs.push(((point, normal), behind));
        }
        // A curved picked face gets a collar a good half member thick, and more than a cell:
        // the structure grows out of it.
        for f in held.iter().chain(load).filter(|f| f.plane.is_none()) {
            for v in gv::voxels_near(&grid, f.tris, (1.6 * h).max(0.6 * wall)) {
                cover[v] = !blocked[v];
            }
        }
        for v in 0..grid.len() {
            if beside[v] && !on_face[v] {
                grid.inside[v] = false;
            }
            if cover[v] {
                grid.inside[v] = true;
            }
        }
        for (_, behind) in &slabs {
            for &v in behind {
                grid.inside[v] = true;
            }
        }
    }
    // A flat picked face is held or loaded through the cells behind it, inside its body: they
    // stand for the body, which spreads the force over the face, and are cut off again at the
    // end. So the structure meets the face only where it needs to. A curved face is held or
    // loaded through the cells in front of it, which stay.
    let mut flat = slabs.iter();
    let mut mark = |faces: &[Face]| -> (Vec<bool>, Vec<Vec<usize>>) {
        let mut set = vec![false; grid.len()];
        let mut each = Vec::new();
        for f in faces {
            let near = match f.plane.and_then(|_| flat.next()) {
                Some((_, behind)) => behind.clone(),
                None => gv::voxels_near(&grid, f.tris, 0.9 * h),
            };
            for &v in &near {
                set[v] = true;
            }
            each.push(near);
        }
        (set, each)
    };
    let (fixed, supports) = mark(held);
    let (loaded, loads) = mark(load);
    if supports.iter().any(Vec::is_empty) || loads.iter().any(Vec::is_empty) {
        return Err("a picked face has no room to build on; give more cells or a margin".into());
    }
    let both: Vec<bool> = fixed.iter().zip(&loaded).map(|(a, b)| *a || *b).collect();
    let keep = if boxed { both.iter().zip(&cover).map(|(a, b)| *a || *b).collect() } else { gv::dilate(&grid, &both, 1.5) };
    // A face that slides or turns on a pin is held by springs, in the directions it cannot
    // move in. Where it shares cells with a face that is held outright, those stay held.
    let mut hard = vec![false; grid.len()];
    for (cells, _) in supports.iter().zip(holds).filter(|(_, hold)| **hold == Hold::Fixed) {
        for &v in cells {
            hard[v] = true;
        }
    }
    let mut springs = Vec::new();
    for ((f, hold), cells) in held.iter().zip(holds).zip(&supports) {
        let axis = match hold {
            Hold::Fixed => continue,
            Hold::Slide => None,
            Hold::Pin => Some(pin_axis(f.tris).ok_or("a pin needs a round face, such as the wall of a hole")?),
        };
        for &v in cells.iter().filter(|v| !hard[**v]) {
            let c = grid.center(v);
            let spring = match axis {
                None => f.plane.map(|p| p.1).or_else(|| normal_near(f.tris, c)).map(gv::spring_along),
                Some((point, axis)) => {
                    let from = sub(c, point);
                    let along = dot(from, axis);
                    unit(cross(axis, [0, 1, 2].map(|a| from[a] - along * axis[a]))).map(gv::spring_across)
                }
            };
            if let Some(spring) = spring {
                springs.push((v, spring));
            }
        }
    }
    let mut forces = Vec::new();
    let mut fields = Vec::new();
    for (faces, action, case) in more {
        let mut on = vec![false; grid.len()];
        for &v in loads.get(faces.clone()).into_iter().flatten().flatten() {
            on[v] = true;
        }
        let cells: Vec<usize> = (0..grid.len()).filter(|&v| on[v]).collect();
        let count = cells.len().max(1) as f64;
        match action {
            Action::Force(force) => forces.push((on, *force, *case)),
            Action::Moment(moment) => {
                // Each cell is pushed around the middle of them all, further out harder, so
                // that together the pushes turn with the moment and add up to no force.
                let mut middle = [0.0; 3];
                for &v in &cells {
                    let c = grid.center(v);
                    middle = [0, 1, 2].map(|a| middle[a] + c[a] / count);
                }
                let push = |v: usize| cross(*moment, sub(grid.center(v), middle));
                let sum: f64 = cells.iter().map(|&v| dot(push(v), push(v))).sum();
                if !(sum > 1e-12) {
                    return Err("the faces of a moment are too small to turn; give more cells".into());
                }
                let scale = dot(*moment, *moment) / sum;
                fields.push((cells.iter().map(|&v| (v, push(v).map(|c| c * scale))).collect(), *case));
                forces.push((on, [0.0; 3], *case));
            }
            Action::Pressure(pressure) => {
                let tris: Vec<[V3; 3]> = load.get(faces.clone()).into_iter().flatten().flat_map(|f| f.tris.iter().copied()).collect();
                let area: f64 =
                    tris.iter().map(|t| 0.5 * dot(cross(sub(t[1], t[0]), sub(t[2], t[0])), cross(sub(t[1], t[0]), sub(t[2], t[0]))).sqrt()).sum();
                let share = pressure * area / count;
                let pushes = cells.iter().filter_map(|&v| Some((v, normal_near(&tris, grid.center(v))?.map(|c| -share * c)))).collect();
                fields.push((pushes, *case));
                forces.push((on, [0.0; 3], *case));
            }
        }
    }
    let problem = Problem { grid, fixed, load: loaded, keep, supports, loads, force: [0.0; 3], more: forces, fields, springs };
    Ok(Level { problem, clip, within, slabs, round })
}

/// Runs the study. `time_left` is asked between steps; when it says no, the best shape so far
/// is finished.
pub fn run(study: &Study, time_left: &mut dyn FnMut() -> bool) -> Result<Report, String> {
    let further = study.more.iter().flat_map(|(faces, ..)| faces);
    // Every face that is held, however: what is held outright first.
    let supports: Vec<FaceTris> = study.held.iter().chain(&study.slide).chain(&study.pin).cloned().collect();
    let holds: Vec<Hold> = (0..supports.len())
        .map(|i| match i {
            i if i < study.held.len() => Hold::Fixed,
            i if i < study.held.len() + study.slide.len() => Hold::Slide,
            _ => Hold::Pin,
        })
        .collect();
    if supports.is_empty()
        || study.load.is_empty()
        || supports.iter().chain(&study.load).chain(further).any(Vec::is_empty)
        || study.more.iter().any(|(faces, ..)| faces.is_empty())
    {
        return Err("give at least one held face and one loaded face".into());
    }
    if !(study.voxels >= 1.0 && study.voxels <= MAX_CELLS as f64 && study.margin >= 0.0 && study.margin.is_finite() && study.clearance >= 0.0) {
        return Err("the cell count or the margin is out of range".into());
    }
    let boxed = study.space.is_none();
    let held = faces(&supports, boxed);
    let mut load = faces(&study.load, boxed);
    let mut more = vec![(0..load.len(), study.action, 0)];
    for (on, action, case) in &study.more {
        let from = load.len();
        load.extend(faces(on, boxed));
        more.push((from..load.len(), *action, *case));
    }
    // An organic structure is found on a coarse grid first and worked out on finer ones, each
    // with eight times the cells but only around the structure the one before found: fine
    // members at a fraction of the work a fine grid all over would be.
    let mut sizes = vec![study.voxels];
    while study.settings.style == gv::Style::Organic && sizes.len() < 3 && sizes[sizes.len() - 1] / 8.0 >= 12_000.0 {
        sizes.push(sizes[sizes.len() - 1] / 8.0);
    }
    sizes.reverse();
    let mut settings = study.settings.clone();
    let (lo, hi, volume) = extent(study, &held, &load)?;
    let longest = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
    // With no thinnest member given: two and a half cells of the finest grid, and at least
    // 1.5 % of the design space's longest side. Finer cells give finer members.
    if !(settings.min_wall > 0.0) {
        settings.min_wall = (2.5 * (volume / study.voxels).cbrt().max(0.05)).max(0.015 * longest);
    }
    // With no limit given, the loaded faces may move by 1/300 of the longest side: a shape that
    // only just does not break is too soft to be of use.
    if !(settings.max_deflection > 0.0) {
        settings.max_deflection = longest / 300.0;
    }
    let mut done: Option<(Level, Outcome)> = None;
    let mut space: Option<f64> = None;
    for &voxels in &sizes {
        let around = done.as_ref().map(|(l, o)| {
            let solid: Vec<bool> = o.field.iter().map(|f| *f > 0.5).collect();
            gv::dilate(&l.problem.grid, &solid, 2.5)
        });
        let band = done.as_ref().zip(around.as_deref()).map(|((l, _), a)| (&l.problem.grid, a));
        let next = level(study, &held, &holds, &load, &more, voxels, settings.min_wall, band)?;
        let g = &next.problem.grid;
        // A share of the design space to fill is a share of the whole of it, also on a finer
        // level that only has the cells around the structure.
        if let Some(share) = study.settings.volume {
            let free = (0..g.len()).filter(|&v| g.inside[v] && !next.problem.keep[v]).count() as f64 * g.h.powi(3);
            let whole = *space.get_or_insert(free);
            settings.volume = Some((share * whole / free.max(1e-9)).clamp(0.01, 0.95));
        }
        let outcome = match &done {
            None => gv::generate(&next.problem, &settings, time_left)?,
            Some((before, coarse)) => {
                let from = &before.problem.grid;
                let start: Vec<f64> = (0..g.len()).map(|v| from.at(g.center(v)).map_or(0.0, |w| coarse.field[w])).collect();
                match gv::refine(&next.problem, &settings, &start, time_left) {
                    // What the finer grid cannot carry stays as the coarser one found it.
                    Ok(fine) if fine.passed || !coarse.passed => fine,
                    _ => break,
                }
            }
        };
        done = Some((next, outcome));
        if !time_left() {
            break;
        }
    }
    let (Level { problem, clip, within, slabs, round }, mut outcome) = done.ok_or("no structure came out")?;
    // Supports that only guide leave a part free to slide or turn if there are too few.
    if holds.iter().any(|hold| *hold != Hold::Fixed) && outcome.deflection > longest {
        outcome.note.push("The supports leave the part free to move: hold one more face.".into());
    }
    let g = &problem.grid;
    let h = g.h;
    let mut field = outcome.field.clone();
    if study.settings.style == gv::Style::Organic {
        gv::polish(g, &mut field);
    }
    if !slabs.is_empty() {
        let mut slab = vec![false; g.len()];
        for &v in slabs.iter().flat_map(|(_, behind)| behind) {
            slab[v] = true;
        }
        gv::fillet(g, &mut field, &slab, round);
        // The cells behind a flat face take the shape of what stands on it, so the structure
        // runs straight into the cut on the face's plane.
        for ((point, normal), behind) in &slabs {
            for &v in behind {
                let c = g.center(v);
                let depth = dot(sub(c, *point), *normal);
                let front = [0, 1, 2].map(|a| c[a] + (0.5 * h - depth) * normal[a]);
                field[v] = g.at(front).filter(|&w| g.inside[w] && !slab[w]).map_or(0.0, |w| field[w]);
            }
        }
    }
    // Three times as fine as the cells, so that a member two cells thick comes out round, with
    // about 45 triangles to a cell on the surface; coarser where that would make more
    // triangles than a design is good to work with.
    let skin = (0..g.len()).filter(|&v| field[v] > 0.5 && g.on_surface_of(v, |w| field[w] > 0.5)).count();
    let up = match skin {
        0..=8_000 => 3,
        8_001..=60_000 => 2,
        _ => 1,
    };
    // The same smoothing in mm whatever the fineness. A lattice's struts are two cells thick:
    // they are drawn at a lower level of the field, where they come out as thick as they are,
    // and smoothed less, which would thin them.
    let lattice = study.settings.style == gv::Style::Lattice;
    if lattice {
        field.iter_mut().for_each(|f| *f *= 1.5);
    }
    let mesh = gv::surface(g, &field, up, if lattice { 4 * up * up } else { 15 * up * up }, &[], clip, &within);
    // Loose crumbs carry nothing: only pieces that stand on a held face stay.
    let held_planes: Vec<Plane> = held.iter().filter_map(|f| f.plane).collect();
    let mesh = if held_planes.len() == held.len() { standing(mesh, &held_planes) } else { mesh };
    if mesh.tri.is_empty() {
        return Err("no structure came out; the faces may be too far apart for this margin".into());
    }
    Ok(Report { mesh, outcome, cell: h, cells: g.len() })
}
