//! Strut style: a space truss of round bars between selected faces, sized by the fully stressed
//! design method (each bar ends up as thin as its force allows, unused bars vanish).

use crate::voxel::{Mesh, V3};

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3, s: f64) -> V3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> V3 {
    let n = norm(a).max(1e-300);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// A planar face the truss stands on: a point, the normal pointing away from its body into the
/// free space, and the face's triangles.
#[derive(Clone, Debug)]
pub struct Patch {
    pub point: V3,
    pub normal: V3,
    pub tris: Vec<[V3; 3]>,
}

impl Patch {
    /// The triangles of `mesh` lying in the plane through `point` with outward `normal`.
    pub fn from_mesh(mesh: &Mesh, point: V3, normal: V3) -> Patch {
        let n = unit(normal);
        let tol = 0.05;
        let tris = mesh
            .tri
            .iter()
            .map(|t| t.map(|v| mesh.pos[v]))
            .filter(|t| t.iter().all(|p| dot(sub(*p, point), n).abs() < tol) && dot(unit(cross(sub(t[1], t[0]), sub(t[2], t[0]))), n) > 0.5)
            .collect();
        Patch { point, normal: n, tris }
    }

    fn basis(&self) -> (V3, V3) {
        let helper = if self.normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = unit(cross(self.normal, helper));
        (u, cross(self.normal, u))
    }

    fn contains_uv(&self, u: V3, v: V3, p: [f64; 2]) -> bool {
        self.tris.iter().any(|t| {
            let q = t.map(|x| [dot(sub(x, self.point), u), dot(sub(x, self.point), v)]);
            let s = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            let (d0, d1, d2) = (s(q[0], q[1]), s(q[1], q[2]), s(q[2], q[0]));
            (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
        })
    }

    /// Landing points spread over the face, kept `inset` away from its bounding edges.
    pub fn nodes(&self, inset: f64) -> Vec<V3> {
        let (u, v) = self.basis();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for t in &self.tris {
            for p in t {
                let q = [dot(sub(*p, self.point), u), dot(sub(*p, self.point), v)];
                for a in 0..2 {
                    lo[a] = lo[a].min(q[a]);
                    hi[a] = hi[a].max(q[a]);
                }
            }
        }
        if lo[0] > hi[0] {
            return Vec::new();
        }
        let len = [hi[0] - lo[0], hi[1] - lo[1]];
        let long = len[0].max(len[1]).max(1e-9);
        let count = |l: f64| -> usize { if l < 4.0 * inset { 1 } else { ((3.0 * l / long).round() as usize).clamp(2, 3) } };
        let (nu, nv) = (count(len[0]), count(len[1]));
        let at = |a: usize, i: usize, n: usize| -> f64 {
            let m = inset.min(len[a] / 2.0);
            if n == 1 { (lo[a] + hi[a]) / 2.0 } else { lo[a] + m + (len[a] - 2.0 * m) * i as f64 / (n - 1) as f64 }
        };
        let mut out = Vec::new();
        for j in 0..nv {
            for i in 0..nu {
                let q = [at(0, i, nu), at(1, j, nv)];
                if self.contains_uv(u, v, q) {
                    out.push(add(add(self.point, u, q[0]), v, q[1]));
                }
            }
        }
        if out.is_empty() {
            // An odd-shaped face: its centroid.
            let mut c = [0.0; 3];
            let mut w = 0.0;
            for t in &self.tris {
                let a = norm(cross(sub(t[1], t[0]), sub(t[2], t[0])));
                for p in t {
                    c = add(c, *p, a / 3.0);
                }
                w += a;
            }
            if w > 0.0 {
                out.push([c[0] / w, c[1] / w, c[2] / w]);
            }
        }
        out
    }
}

pub struct Input<'a> {
    /// Faces that are held.
    pub fixed: &'a [Patch],
    /// The face the force acts on.
    pub load: &'a Patch,
    /// Bodies no bar may pass through.
    pub obstacles: &'a [Mesh],
    /// The force, N.
    pub force: V3,
    /// Allowed stress, N/mm².
    pub allowed: f64,
    /// Young's modulus, N/mm².
    pub modulus: f64,
    /// Factor against buckling.
    pub safety: f64,
    /// Thinnest bar radius, mm.
    pub min_radius: f64,
}

#[derive(Clone, Debug)]
pub struct Bar {
    pub a: usize,
    pub b: usize,
    pub radius: f64,
    /// Axial force, N; tension positive.
    pub force: f64,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub pos: V3,
    /// The patch it stands on: 0.. for the held faces, then the load face. `None`: a free joint.
    pub patch: Option<usize>,
}

pub struct Truss {
    pub nodes: Vec<Node>,
    pub bars: Vec<Bar>,
    pub stress: f64,
    pub deflection: f64,
    pub passed: bool,
    pub volume: f64,
}

/// Solves the symmetric positive definite system in place (Cholesky); `false` if it is not.
fn solve(k: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = k[j * n + j];
        for p in 0..j {
            d -= k[j * n + p] * k[j * n + p];
        }
        if !(d > 0.0) || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        k[j * n + j] = d;
        for i in j + 1..n {
            let mut s = k[i * n + j];
            for p in 0..j {
                s -= k[i * n + p] * k[j * n + p];
            }
            k[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for p in 0..i {
            s -= k[i * n + p] * b[p];
        }
        b[i] = s / k[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for p in i + 1..n {
            s -= k[p * n + i] * b[p];
        }
        b[i] = s / k[i * n + i];
    }
    true
}

struct Model {
    /// Degree of freedom of each node's x, or usize::MAX when held.
    dof: Vec<usize>,
    ndof: usize,
    /// (a, b, length, direction a -> b).
    members: Vec<(usize, usize, f64, V3)>,
    load: Vec<f64>,
}

impl Model {
    /// Displacements and member forces for the given areas.
    fn analyse(&self, area: &[f64], e: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        let n = self.ndof;
        let mut k = vec![0.0; n * n];
        for (m, &(a, b, len, d)) in self.members.iter().enumerate() {
            let s = e * area[m] / len;
            if s <= 0.0 {
                continue;
            }
            for (ni, si) in [(a, 1.0), (b, -1.0)] {
                for (nj, sj) in [(a, 1.0), (b, -1.0)] {
                    let (di, dj) = (self.dof[ni], self.dof[nj]);
                    if di == usize::MAX || dj == usize::MAX {
                        continue;
                    }
                    for x in 0..3 {
                        for y in 0..3 {
                            k[(di + x) * n + dj + y] += s * si * sj * d[x] * d[y];
                        }
                    }
                }
            }
        }
        // A whisper of stiffness everywhere: joints that carry nothing stay solvable.
        let top = (0..n).map(|i| k[i * n + i]).fold(0.0, f64::max).max(1e-12);
        for i in 0..n {
            k[i * n + i] += 1e-9 * top;
        }
        let mut u = self.load.clone();
        if !solve(&mut k, &mut u, n) {
            return None;
        }
        let at = |node: usize| -> V3 {
            let d = self.dof[node];
            if d == usize::MAX { [0.0; 3] } else { [u[d], u[d + 1], u[d + 2]] }
        };
        let forces = self.members.iter().enumerate().map(|(m, &(a, b, len, d))| e * area[m] / len * dot(sub(at(b), at(a)), d)).collect();
        Some((u, forces))
    }
}

/// Area a bar needs for force `n` (tension positive) over length `len`.
fn needed(n: f64, len: f64, inp: &Input) -> f64 {
    let strength = n.abs() / inp.allowed;
    if n >= 0.0 {
        return strength;
    }
    // Euler, pinned ends, round solid bar: I = A² / (4π).
    let buckling = (4.0 * n.abs() * inp.safety * len * len / (std::f64::consts::PI * inp.modulus)).sqrt();
    strength.max(buckling)
}

pub fn design(inp: &Input) -> Result<Truss, String> {
    let pi = std::f64::consts::PI;
    let patches: Vec<&Patch> = inp.fixed.iter().chain(std::iter::once(inp.load)).collect();
    let load_patch = patches.len() - 1;
    let inset = (3.0 * inp.min_radius).max(2.0);
    let mut nodes: Vec<Node> = Vec::new();
    for (i, p) in patches.iter().enumerate() {
        let pts = p.nodes(inset);
        if pts.is_empty() {
            return Err(if i == load_patch {
                "the loaded face is not a flat face with an extent".into()
            } else {
                "a held face is not a flat face with an extent".into()
            });
        }
        nodes.extend(pts.into_iter().map(|pos| Node { pos, patch: Some(i) }));
    }
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for n in &nodes {
        for a in 0..3 {
            lo[a] = lo[a].min(n.pos[a]);
            hi[a] = hi[a].max(n.pos[a]);
        }
    }
    let size = sub(hi, lo);
    let diag = norm(size).max(1e-6);
    let free_space = |p: V3, margin: f64| -> bool {
        patches.iter().all(|f| dot(sub(p, f.point), f.normal) > margin) && !inp.obstacles.iter().any(|m| m.contains(p))
    };
    // Free joints on a grid through the space between the faces.
    let long = size[0].max(size[1]).max(size[2]);
    let pitch = long / 4.0;
    let counts = size.map(|s| (((s / pitch).round() as usize) + 1).clamp(1, 5));
    for k in 0..counts[2] {
        for j in 0..counts[1] {
            for i in 0..counts[0] {
                let f = |c: usize, n: usize| if n == 1 { 0.5 } else { c as f64 / (n - 1) as f64 };
                let p = [lo[0] + f(i, counts[0]) * size[0], lo[1] + f(j, counts[1]) * size[1], lo[2] + f(k, counts[2]) * size[2]];
                if free_space(p, 0.2 * pitch) {
                    nodes.push(Node { pos: p, patch: None });
                }
            }
        }
    }
    let real = nodes.len();
    // The loaded body is rigid: an apex behind its face, tied to every landing point.
    let load_nodes: Vec<usize> = (0..real).filter(|&i| nodes[i].patch == Some(load_patch)).collect();
    let mut centre = [0.0; 3];
    for &i in &load_nodes {
        centre = add(centre, nodes[i].pos, 1.0 / load_nodes.len() as f64);
    }
    let apex = add(centre, inp.load.normal, -(0.3 * diag).max(5.0));
    let mut pos: Vec<V3> = nodes.iter().map(|n| n.pos).collect();
    pos.push(apex);
    let mut dof = vec![usize::MAX; pos.len()];
    let mut ndof = 0;
    for i in 0..pos.len() {
        let held = i < real && nodes[i].patch.is_some_and(|p| p != load_patch);
        if !held {
            dof[i] = ndof;
            ndof += 3;
        }
    }
    // Candidate bars.
    let mut members: Vec<(usize, usize, f64, V3)> = Vec::new();
    for a in 0..real {
        for b in a + 1..real {
            if nodes[a].patch.is_some() && nodes[a].patch == nodes[b].patch {
                continue;
            }
            let held = |i: usize| nodes[i].patch.is_some_and(|p| p != load_patch);
            if held(a) && held(b) {
                continue;
            }
            let d = sub(pos[b], pos[a]);
            let len = norm(d);
            if len < 1e-6 || len > 0.85 * diag {
                continue;
            }
            let e = unit(d);
            // Not through a body, not behind a face, and not through another joint.
            let clear = (1..10).all(|s| free_space(add(pos[a], d, s as f64 / 10.0), -1e-6));
            let through = (0..real).any(|c| {
                if c == a || c == b {
                    return false;
                }
                let t = dot(sub(pos[c], pos[a]), e);
                t > 1e-6 && t < len - 1e-6 && norm(sub(sub(pos[c], pos[a]), [e[0] * t, e[1] * t, e[2] * t])) < 1e-6 * diag
            });
            if clear && !through {
                members.push((a, b, len, e));
            }
        }
    }
    let designed = members.len();
    if designed == 0 {
        return Err("there is no free way for a bar between the faces".into());
    }
    // Rigid ties of the loaded face.
    let apex_id = pos.len() - 1;
    for (x, &a) in load_nodes.iter().enumerate() {
        let d = sub(pos[apex_id], pos[a]);
        members.push((a, apex_id, norm(d), unit(d)));
        for &b in &load_nodes[x + 1..] {
            let d = sub(pos[b], pos[a]);
            members.push((a, b, norm(d), unit(d)));
        }
    }
    let mut load = vec![0.0; ndof];
    for &i in &load_nodes {
        for x in 0..3 {
            load[dof[i] + x] = inp.force[x] / load_nodes.len() as f64;
        }
    }
    let model = Model { dof, ndof, members, load };
    let fnorm = norm(inp.force).max(1e-9);
    let start = fnorm / inp.allowed;
    let rigid = 1000.0 * start;
    let min_area = pi * inp.min_radius * inp.min_radius;
    let mut area: Vec<f64> = (0..model.members.len()).map(|m| if m < designed { start } else { rigid }).collect();
    let mut forces = vec![0.0; model.members.len()];
    for _ in 0..90 {
        let (_, f) = model.analyse(&area, inp.modulus).ok_or("the truss could not be solved")?;
        forces = f;
        let top = area[..designed].iter().cloned().fold(0.0, f64::max);
        for m in 0..designed {
            let want = needed(forces[m], model.members[m].2, inp).max(1e-6 * top);
            area[m] = 0.5 * (area[m] + want);
        }
    }
    let _ = forces;
    // Keep the bars that matter, as few as still carry the load.
    let top = area[..designed].iter().cloned().fold(0.0, f64::max);
    let deflection_of = |u: &[f64]| -> f64 {
        let dir = unit(inp.force);
        load_nodes.iter().map(|&i| (0..3).map(|x| u[model.dof[i] + x] * dir[x]).sum::<f64>()).sum::<f64>() / load_nodes.len() as f64
    };
    let mut best: Option<(Vec<f64>, Vec<f64>, f64, f64)> = None;
    for threshold in [0.3, 0.2, 0.12, 0.07, 0.04, 0.02, 0.008, 0.0] {
        let mut a: Vec<f64> = (0..model.members.len())
            .map(|m| {
                if m >= designed {
                    rigid
                } else if area[m] >= threshold * top && area[m] > 2e-6 * top {
                    area[m].max(min_area)
                } else {
                    0.0
                }
            })
            .collect();
        let mut result = None;
        for _ in 0..25 {
            let Some((u, f)) = model.analyse(&a, inp.modulus) else { break };
            let mut worst = 0.0f64;
            for m in 0..designed {
                if a[m] > 0.0 {
                    let want = needed(f[m], model.members[m].2, inp).max(min_area);
                    worst = worst.max(needed(f[m], model.members[m].2, inp) / a[m]);
                    a[m] = 0.5 * (a[m] + want).max(want.min(a[m]));
                }
            }
            result = Some((u, f, worst));
        }
        let Some((u, f, _)) = result else { continue };
        // Final check with the sizes as built.
        let Some((u2, f2)) = model.analyse(&a, inp.modulus) else { continue };
        let _ = (u, f);
        let mut ratio = 0.0f64;
        let mut stress = 0.0f64;
        for m in 0..designed {
            if a[m] > 0.0 {
                ratio = ratio.max(needed(f2[m], model.members[m].2, inp) / a[m]);
                stress = stress.max(f2[m].abs() / a[m]);
            }
        }
        let defl = deflection_of(&u2);
        let ok = ratio <= 1.05 && defl.is_finite() && defl.abs() < 0.2 * diag;
        if ok {
            best = Some((a, f2, stress, defl));
            break;
        }
        if best.is_none() && threshold == 0.0 {
            best = Some((a, f2, stress, defl));
        }
    }
    let (a, f, stress, deflection) = best.ok_or("no arrangement of bars carries the load")?;
    let mut bars = Vec::new();
    let mut volume = 0.0;
    for m in 0..designed {
        if a[m] > 0.0 {
            let (na, nb, len, _) = model.members[m];
            bars.push(Bar { a: na, b: nb, radius: (a[m] / pi).sqrt(), force: f[m] });
            volume += a[m] * len;
        }
    }
    if bars.is_empty() {
        return Err("no arrangement of bars carries the load".into());
    }
    let ratio_ok = bars.iter().all(|b| {
        let len = norm(sub(pos[b.b], pos[b.a]));
        needed(b.force, len, inp) <= 1.05 * pi * b.radius * b.radius
    });
    Ok(Truss { nodes, bars, stress, deflection, passed: ratio_ok && deflection.is_finite() && deflection.abs() < 0.2 * diag, volume })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::box_mesh;

    fn plates() -> (Mesh, Mesh) {
        (box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 40.0]), box_mesh([120.0, 10.0, 0.0], [125.0, 50.0, 40.0]))
    }

    #[test]
    fn bars_between_two_plates_carry_the_load() {
        let (wall, plate) = plates();
        let fixed = [Patch::from_mesh(&wall, [5.0, 30.0, 20.0], [1.0, 0.0, 0.0])];
        let load = Patch::from_mesh(&plate, [120.0, 30.0, 20.0], [-1.0, 0.0, 0.0]);
        assert_eq!(fixed[0].tris.len(), 2);
        let obstacles = [wall.clone(), plate.clone()];
        let inp = Input {
            fixed: &fixed,
            load: &load,
            obstacles: &obstacles,
            force: [0.0, 0.0, -200.0],
            allowed: 12.25,
            modulus: 1900.0,
            safety: 2.0,
            min_radius: 1.5,
        };
        let t = design(&inp).unwrap();
        assert!(t.passed);
        assert!(t.bars.len() >= 3);
        assert!(t.stress <= 12.25 * 1.05);
        assert!(t.deflection > 0.0 && t.deflection < 10.0);
    }
}
