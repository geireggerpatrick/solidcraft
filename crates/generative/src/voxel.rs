//! Generative design on a voxel grid: the user's body is the design space, selected faces are
//! supports and the load, and one of three styles fills the space with as little material as
//! the force allows. The result is a smooth triangle mesh.
//!
//! - Organic: SIMP topology optimisation (3D linear elastic FEM, 8-node bricks, optimality
//!   criteria update, sensitivity filter), the volume lowered step by step while the stress check
//!   passes.
//! - Tree: shortest paths through the inside of the design space from the load to every support,
//!   merged into branches; branch thickness grows with the supports it carries and is sized by FEM.
//! - Lattice: a closed skin with a cubic lattice inside, skin and cell size picked by FEM.
//!
//! Everything here is pure computation; `lib.rs` talks to SolidCraft.

use crate::Material;

pub type V3 = [f64; 3];

/// Element matrix-vector products done so far (a deterministic measure of compute spent).
static WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn work() -> u64 {
    WORK.load(std::sync::atomic::Ordering::Relaxed)
}

fn add_work(n: usize) {
    WORK.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
}

#[inline(always)]
fn dot24(row: &[f64; 24], u: &[f64; 24]) -> f64 {
    let mut a = [0.0; 4];
    let mut i = 0;
    while i < 24 {
        a[0] += row[i] * u[i];
        a[1] += row[i + 1] * u[i + 1];
        a[2] += row[i + 2] * u[i + 2];
        a[3] += row[i + 3] * u[i + 3];
        i += 4;
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub pos: Vec<V3>,
    pub tri: Vec<[usize; 3]>,
}

impl Mesh {
    pub fn bbox(&self) -> (V3, V3) {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for p in &self.pos {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }

    /// Heights where a vertical ray through (x, y) crosses the surface, sorted.
    fn crossings(&self, x: f64, y: f64) -> Vec<f64> {
        let mut zs = Vec::new();
        for t in &self.tri {
            let (a, b, c) = (self.pos[t[0]], self.pos[t[1]], self.pos[t[2]]);
            if let Some(z) = ray_z(a, b, c, x, y) {
                zs.push(z);
            }
        }
        zs.sort_by(f64::total_cmp);
        zs
    }

    /// Whether a point is inside the closed mesh (vertical ray parity).
    pub fn contains(&self, p: V3) -> bool {
        let zs = self.crossings(p[0] + 1.37e-5, p[1] + 2.71e-5);
        zs.iter().filter(|z| **z > p[2]).count() % 2 == 1
    }
}

/// A closed box mesh, outward facing.
pub fn box_mesh(lo: V3, hi: V3) -> Mesh {
    let p = |i: usize| [if i & 1 == 1 { hi[0] } else { lo[0] }, if i & 2 == 2 { hi[1] } else { lo[1] }, if i & 4 == 4 { hi[2] } else { lo[2] }];
    let pos = (0..8).map(p).collect();
    let tri =
        vec![[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]];
    Mesh { pos, tri }
}

/// Height where the vertical line through (x, y) meets triangle abc, if it does.
fn ray_z(a: V3, b: V3, c: V3, x: f64, y: f64) -> Option<f64> {
    let d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if d.abs() < 1e-14 {
        return None;
    }
    let l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d;
    let l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d;
    let l3 = 1.0 - l1 - l2;
    if l1 < 0.0 || l2 < 0.0 || l3 < 0.0 {
        return None;
    }
    Some(l1 * a[2] + l2 * b[2] + l3 * c[2])
}

/// A regular voxel grid over the design space.
#[derive(Clone, Debug)]
pub struct Grid {
    pub n: [usize; 3],
    pub h: f64,
    pub origin: V3,
    pub inside: Vec<bool>,
}

impl Grid {
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn id(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.n[0] * (j + self.n[1] * k)
    }
    pub fn ijk(&self, id: usize) -> [usize; 3] {
        let i = id % self.n[0];
        let j = (id / self.n[0]) % self.n[1];
        [i, j, id / (self.n[0] * self.n[1])]
    }
    pub fn center(&self, id: usize) -> V3 {
        let [i, j, k] = self.ijk(id);
        [self.origin[0] + (i as f64 + 0.5) * self.h, self.origin[1] + (j as f64 + 0.5) * self.h, self.origin[2] + (k as f64 + 0.5) * self.h]
    }
    /// Neighbours sharing a face, edge or corner (26), with their distance in voxels.
    pub fn neighbours(&self, id: usize, out: &mut Vec<(usize, f64)>) {
        out.clear();
        let [i, j, k] = self.ijk(id);
        for dk in -1i64..=1 {
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    if di == 0 && dj == 0 && dk == 0 {
                        continue;
                    }
                    let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                    if a < 0 || b < 0 || c < 0 || a >= self.n[0] as i64 || b >= self.n[1] as i64 || c >= self.n[2] as i64 {
                        continue;
                    }
                    out.push((self.id(a as usize, b as usize, c as usize), ((di * di + dj * dj + dk * dk) as f64).sqrt()));
                }
            }
        }
    }
    /// A voxel inside with a face neighbour outside (or at the grid's edge).
    pub fn on_surface(&self, id: usize) -> bool {
        if !self.inside[id] {
            return false;
        }
        let [i, j, k] = self.ijk(id);
        let p = [i, j, k];
        for a in 0..3 {
            for s in [-1i64, 1] {
                let v = p[a] as i64 + s;
                if v < 0 || v >= self.n[a] as i64 {
                    return true;
                }
                let mut q = p;
                q[a] = v as usize;
                if !self.inside[self.id(q[0], q[1], q[2])] {
                    return true;
                }
            }
        }
        false
    }
    /// The voxel a point lies in.
    pub fn at(&self, p: V3) -> Option<usize> {
        let c = [0, 1, 2].map(|a| ((p[a] - self.origin[a]) / self.h).floor());
        (0..3).all(|a| c[a] >= 0.0 && c[a] < self.n[a] as f64).then(|| self.id(c[0] as usize, c[1] as usize, c[2] as usize))
    }

    pub fn nearest(&self, p: V3, accept: impl Fn(usize) -> bool) -> Option<usize> {
        (0..self.len()).filter(|&v| accept(v)).min_by(|&a, &b| norm(sub(self.center(a), p)).total_cmp(&norm(sub(self.center(b), p))))
    }
}

/// Voxelizes a closed mesh with cubes of edge `h`.
pub fn voxelize(mesh: &Mesh, h: f64) -> Grid {
    let (lo, hi) = mesh.bbox();
    let n = [0, 1, 2].map(|a| (((hi[a] - lo[a]) / h).ceil() as usize).max(1));
    // Centre the grid on the bounding box.
    let origin = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0 - n[a] as f64 * h / 2.0);
    let mut grid = Grid { n, h, origin, inside: Vec::new() };
    grid.inside = mesh_mask(&grid, mesh);
    grid
}

/// A grid of cubes of edge `h` over the box `lo..hi`, every voxel in the design space.
pub fn box_grid(lo: V3, hi: V3, h: f64) -> Grid {
    let n = [0, 1, 2].map(|a| (((hi[a] - lo[a]) / h).ceil() as usize).max(1));
    let origin = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0 - n[a] as f64 * h / 2.0);
    Grid { n, h, origin, inside: vec![true; n[0] * n[1] * n[2]] }
}

/// For each voxel of the grid, whether its centre is inside the closed mesh.
pub fn mesh_mask(g: &Grid, mesh: &Mesh) -> Vec<bool> {
    let (n, h, origin) = (g.n, g.h, g.origin);
    let mut inside = vec![false; n[0] * n[1] * n[2]];
    // Bin triangles by column so each column tests only nearby triangles.
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); n[0] * n[1]];
    for (t, tri) in mesh.tri.iter().enumerate() {
        let ps = tri.map(|v| mesh.pos[v]);
        let x0 = ps.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
        let x1 = ps.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
        let y0 = ps.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
        let y1 = ps.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        // A triangle wholly beside the grid crosses none of its columns.
        if x1 < origin[0] || y1 < origin[1] || x0 > origin[0] + n[0] as f64 * h || y0 > origin[1] + n[1] as f64 * h {
            continue;
        }
        let ci = |x: f64, a: usize| (((x - origin[a]) / h - 0.5).floor().max(0.0) as usize).min(n[a] - 1);
        let cj = |x: f64, a: usize| (((x - origin[a]) / h - 0.5).ceil().max(0.0) as usize).min(n[a] - 1);
        for j in ci(y0, 1)..=cj(y1, 1) {
            for i in ci(x0, 0)..=cj(x1, 0) {
                bins[i + n[0] * j].push(t);
            }
        }
    }
    for j in 0..n[1] {
        for i in 0..n[0] {
            let x = origin[0] + (i as f64 + 0.5) * h + 1.37e-5 * h;
            let y = origin[1] + (j as f64 + 0.5) * h + 2.71e-5 * h;
            let mut zs: Vec<f64> = bins[i + n[0] * j]
                .iter()
                .filter_map(|&t| {
                    let tri = mesh.tri[t];
                    ray_z(mesh.pos[tri[0]], mesh.pos[tri[1]], mesh.pos[tri[2]], x, y)
                })
                .collect();
            if zs.is_empty() {
                continue;
            }
            zs.sort_by(f64::total_cmp);
            for k in 0..n[2] {
                let z = origin[2] + (k as f64 + 0.5) * h;
                let below = zs.iter().filter(|v| **v < z).count();
                if below % 2 == 1 {
                    inside[g.id(i, j, k)] = true;
                }
            }
        }
    }
    inside
}

/// The geometry of a selected face.
#[derive(Clone, Debug)]
pub enum Surface {
    Plane { point: V3, normal: V3 },
    Cylinder { origin: V3, axis: V3, radius: f64 },
}

#[derive(Clone, Debug)]
pub struct Face {
    pub surface: Surface,
    /// A point on the face.
    pub point: V3,
}

fn axis_dist(p: V3, origin: V3, axis: V3) -> f64 {
    let d = sub(p, origin);
    norm(cross(d, unit(axis)))
}

/// Voxels that belong to a face: connected voxels near its surface, starting at its point.
pub fn face_voxels(g: &Grid, f: &Face) -> Vec<usize> {
    let h = g.h;
    let near = |v: usize| -> bool {
        if !g.inside[v] {
            return false;
        }
        let c = g.center(v);
        match f.surface {
            Surface::Plane { point, normal } => g.on_surface(v) && dot(sub(c, point), unit(normal)).abs() <= 0.9 * h,
            Surface::Cylinder { origin, axis, radius } => {
                (axis_dist(c, origin, axis) - radius).abs() <= 1.0 * h || axis_dist(c, origin, axis) <= radius + h
            }
        }
    };
    let Some(seed) = g.nearest(f.point, near) else { return Vec::new() };
    if norm(sub(g.center(seed), f.point)) > 3.0 * h {
        return Vec::new();
    }
    let mut seen = vec![false; g.len()];
    let mut stack = vec![seed];
    seen[seed] = true;
    let mut out = Vec::new();
    let mut nb = Vec::new();
    while let Some(v) = stack.pop() {
        out.push(v);
        g.neighbours(v, &mut nb);
        for &(w, _) in &nb {
            if !seen[w] && near(w) {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    out
}

/// Grows a set by `r` voxels (Euclidean) inside the design space.
pub fn dilate(g: &Grid, set: &[bool], r: f64) -> Vec<bool> {
    let mut out = set.to_vec();
    let ri = r.ceil() as i64;
    for v in 0..g.len() {
        if !set[v] {
            continue;
        }
        let [i, j, k] = g.ijk(v);
        for dk in -ri..=ri {
            for dj in -ri..=ri {
                for di in -ri..=ri {
                    if ((di * di + dj * dj + dk * dk) as f64) > r * r + 1e-9 {
                        continue;
                    }
                    let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                    if a < 0 || b < 0 || c < 0 || a >= g.n[0] as i64 || b >= g.n[1] as i64 || c >= g.n[2] as i64 {
                        continue;
                    }
                    let w = g.id(a as usize, b as usize, c as usize);
                    if g.inside[w] {
                        out[w] = true;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- FEM

/// Element stiffness of a unit cube, E = 1, nu = 0.3, and the strain-displacement matrix at its
/// centre.
struct Brick {
    ke: [[f64; 24]; 24],
    b0: [[f64; 24]; 6],
    d: [[f64; 6]; 6],
}

const CORNERS: [[f64; 3]; 8] =
    [[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]];

fn bmat(xi: f64, eta: f64, zeta: f64) -> [[f64; 24]; 6] {
    let mut b = [[0.0; 24]; 6];
    for (n, c) in CORNERS.iter().enumerate() {
        // dN/dx = 2 dN/dxi for a unit cube.
        let dx = 2.0 * 0.125 * c[0] * (1.0 + eta * c[1]) * (1.0 + zeta * c[2]);
        let dy = 2.0 * 0.125 * c[1] * (1.0 + xi * c[0]) * (1.0 + zeta * c[2]);
        let dz = 2.0 * 0.125 * c[2] * (1.0 + xi * c[0]) * (1.0 + eta * c[1]);
        let (u, v, w) = (3 * n, 3 * n + 1, 3 * n + 2);
        b[0][u] = dx;
        b[1][v] = dy;
        b[2][w] = dz;
        b[3][u] = dy;
        b[3][v] = dx;
        b[4][v] = dz;
        b[4][w] = dy;
        b[5][u] = dz;
        b[5][w] = dx;
    }
    b
}

impl Brick {
    fn new() -> Brick {
        let nu = 0.3;
        let lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = 1.0 / (2.0 * (1.0 + nu));
        let mut d = [[0.0; 6]; 6];
        for a in 0..3 {
            for b in 0..3 {
                d[a][b] = lam;
            }
            d[a][a] = lam + 2.0 * mu;
            d[a + 3][a + 3] = mu;
        }
        let g = 1.0 / 3f64.sqrt();
        let mut ke = [[0.0; 24]; 24];
        for gx in [-g, g] {
            for gy in [-g, g] {
                for gz in [-g, g] {
                    let b = bmat(gx, gy, gz);
                    let mut db = [[0.0; 24]; 6];
                    for r in 0..6 {
                        for c in 0..24 {
                            db[r][c] = (0..6).map(|s| d[r][s] * b[s][c]).sum();
                        }
                    }
                    for r in 0..24 {
                        for c in 0..24 {
                            ke[r][c] += (0..6).map(|s| b[s][r] * db[s][c]).sum::<f64>() * 0.125;
                        }
                    }
                }
            }
        }
        Brick { ke, b0: bmat(0.0, 0.0, 0.0), d }
    }
}

/// A linear elastic model over some voxels of the grid.
/// The stiffness of a unit voxel of unit modulus, as a [`GridJob`] carries it.
pub fn unit_stiffness() -> [[f64; 24]; 24] {
    Brick::new().ke
}

/// One linear problem on the whole voxel grid, for a solver outside this crate (a GPU): find the
/// displacements `d` with `K d = rhs`, where `K` is the grid's stiffness with the held nodes
/// taken out. Nodes are numbered `i + (n[0] + 1) * (j + (n[1] + 1) * k)`, voxels as in [`Grid`].
pub struct GridJob<'a> {
    /// Voxels along x, y and z.
    pub n: [usize; 3],
    /// The stiffness of a unit voxel of unit modulus: 8 corners with 3 directions each, the
    /// corners in the order (0,0,0) (1,0,0) (1,1,0) (0,1,0) (0,0,1) (1,0,1) (1,1,1) (0,1,1).
    pub ke: &'a [[f64; 24]; 24],
    /// What each voxel's stiffness is multiplied with; 0 for a voxel that is not there.
    pub stiff: &'a [f32],
    /// Per node: held in all three directions.
    pub fixed: &'a [bool],
    /// Three values per node; 0 at the held nodes.
    pub rhs: &'a [f32],
    /// Stop when the residual is this small against `rhs`.
    pub tol: f64,
    pub max_iter: usize,
}

/// A solver for [`GridJob`]s.
pub trait GridSolver: Send {
    /// The displacements, three per node, and the iterations it took.
    fn solve(&mut self, job: &GridJob) -> Result<(Vec<f32>, usize), String>;
    /// What computes, for the notes of a result.
    fn name(&self) -> String;
}

static GRID_SOLVER: std::sync::Mutex<Option<Box<dyn GridSolver>>> = std::sync::Mutex::new(None);

/// Installs (or with `None` removes) the solver that designs with [`Settings::accelerate`] set
/// use for their linear problems. Without one, or when it fails, the built-in solver computes.
pub fn set_grid_solver(solver: Option<Box<dyn GridSolver>>) {
    if let Ok(mut slot) = GRID_SOLVER.lock() {
        *slot = solver;
    }
}

/// What opens the solver when a design first asks for one.
type SolverSource = Box<dyn FnOnce() -> Option<Box<dyn GridSolver>> + Send>;

static GRID_SOLVER_SOURCE: std::sync::Mutex<Option<SolverSource>> = std::sync::Mutex::new(None);

/// Registers what opens the solver, to be called once, when a design with
/// [`Settings::accelerate`] first needs it: a program that never computes a design opens none.
pub fn set_grid_solver_source(source: impl FnOnce() -> Option<Box<dyn GridSolver>> + Send + 'static) {
    if let Ok(mut slot) = GRID_SOLVER_SOURCE.lock() {
        *slot = Some(Box::new(source));
    }
}

/// The installed solver's name; opens the solver if a source for one is registered.
pub fn grid_solver_name() -> Option<String> {
    let source = GRID_SOLVER_SOURCE.lock().ok()?.take();
    if let Some(open) = source
        && let Some(solver) = open()
    {
        set_grid_solver(Some(solver));
    }
    GRID_SOLVER.lock().ok()?.as_ref().map(|s| s.name())
}

struct Fem<'a> {
    /// For each node of the model, its node on the whole grid.
    grid_node: Vec<u32>,
    /// Whether to hand the linear problems to the installed [`GridSolver`].
    accelerate: bool,
    g: &'a Grid,
    brick: &'a Brick,
    elems: Vec<usize>,
    dofs: Vec<[u32; 24]>,
    ndof: usize,
    fixed: Vec<bool>,
    force: Vec<f64>,
    /// The built-in solver's levels, built when it first solves.
    mg: std::sync::OnceLock<Multigrid<'a>>,
    coords: Vec<[usize; 3]>,
}

const NONE: u32 = u32::MAX;

impl<'a> Fem<'a> {
    /// `elems`: the voxels in the model; `fixed`/`load`: voxels whose nodes are held / loaded.
    fn new(g: &'a Grid, brick: &'a Brick, elems: Vec<usize>, fixed: &[bool], load: &[bool], f: V3, accelerate: bool) -> Fem<'a> {
        let nn = (g.n[0] + 1) * (g.n[1] + 1) * (g.n[2] + 1);
        let node = |i: usize, j: usize, k: usize| i + (g.n[0] + 1) * (j + (g.n[1] + 1) * k);
        let corners = |v: usize| -> [usize; 8] {
            let [i, j, k] = g.ijk(v);
            [
                node(i, j, k),
                node(i + 1, j, k),
                node(i + 1, j + 1, k),
                node(i, j + 1, k),
                node(i, j, k + 1),
                node(i + 1, j, k + 1),
                node(i + 1, j + 1, k + 1),
                node(i, j + 1, k + 1),
            ]
        };
        let mut map = vec![NONE; nn];
        let mut count = 0u32;
        for &v in &elems {
            for n in corners(v) {
                if map[n] == NONE {
                    map[n] = count;
                    count += 1;
                }
            }
        }
        let ndof = 3 * count as usize;
        let dofs: Vec<[u32; 24]> = elems
            .iter()
            .map(|&v| {
                let c = corners(v);
                let mut d = [0u32; 24];
                for (a, n) in c.iter().enumerate() {
                    for x in 0..3 {
                        d[3 * a + x] = 3 * map[*n] + x as u32;
                    }
                }
                d
            })
            .collect();
        let mut fixed_dof = vec![false; ndof];
        let mut loaded = vec![false; count as usize];
        for (e, &v) in elems.iter().enumerate() {
            if fixed[v] {
                for d in dofs[e] {
                    fixed_dof[d as usize] = true;
                }
            }
            if load[v] {
                for a in 0..8 {
                    loaded[(dofs[e][3 * a] / 3) as usize] = true;
                }
            }
        }
        let free_loaded: Vec<usize> = (0..count as usize).filter(|&n| loaded[n] && !fixed_dof[3 * n]).collect();
        let mut force = vec![0.0; ndof];
        if !free_loaded.is_empty() {
            let share = 1.0 / free_loaded.len() as f64;
            for n in free_loaded {
                for x in 0..3 {
                    force[3 * n + x] = f[x] * share;
                }
            }
        }
        let coords: Vec<[usize; 3]> = elems.iter().map(|&v| g.ijk(v)).collect();
        let mut grid_node = vec![0u32; count as usize];
        for (n, &m) in map.iter().enumerate() {
            if m != NONE {
                grid_node[m as usize] = n as u32;
            }
        }
        let accelerate = accelerate && grid_solver_name().is_some();
        Fem { grid_node, accelerate, g, brick, elems, dofs, ndof, fixed: fixed_dof, force, mg: std::sync::OnceLock::new(), coords }
    }

    fn matvec(&self, stiff: &[f64], x: &[f64], y: &mut [f64]) {
        apply_elements(&self.brick.ke, &self.dofs, stiff, x, y);
        for i in 0..self.ndof {
            if self.fixed[i] {
                y[i] = x[i];
            }
        }
    }

    /// Solves K u = f by CG preconditioned with a geometric multigrid V-cycle, starting from `u`.
    /// Returns the iterations.
    fn solve(&self, stiff: &[f64], u: &mut Vec<f64>, tol: f64, max_iter: usize) -> usize {
        if self.accelerate
            && let Some(iterations) = self.solve_outside(stiff, u, tol, max_iter)
        {
            return iterations;
        }
        self.solve_here(stiff, u, tol, max_iter)
    }

    /// Solves with the installed [`GridSolver`], which computes in single precision: the
    /// residual is taken here in double precision and the solver finds the correction to it,
    /// until the residual is small enough. `None` when there is no solver or it failed.
    fn solve_outside(&self, stiff: &[f64], u: &mut Vec<f64>, tol: f64, max_iter: usize) -> Option<usize> {
        let n = self.ndof;
        if u.len() != n {
            *u = vec![0.0; n];
        }
        let g = self.g;
        let nodes = (g.n[0] + 1) * (g.n[1] + 1) * (g.n[2] + 1);
        let mut stiff_grid = vec![0.0f32; g.len()];
        for (&v, &s) in self.elems.iter().zip(stiff) {
            stiff_grid[v] = s as f32;
        }
        let mut fixed_grid = vec![false; nodes];
        for (m, &node) in self.grid_node.iter().enumerate() {
            fixed_grid[node as usize] = self.fixed[3 * m];
        }
        let bnorm = self.force.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let mut r = vec![0.0; n];
        let mut total = 0;
        for round in 0..6 {
            self.matvec(stiff, u, &mut r);
            for i in 0..n {
                r[i] = if self.fixed[i] { 0.0 } else { self.force[i] - r[i] };
            }
            let rnorm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rnorm / bnorm < tol || total >= max_iter {
                return Some(total);
            }
            // Scaled to about one: single precision keeps its digits for the correction.
            let scale = r.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-300);
            let mut rhs = vec![0.0f32; 3 * nodes];
            for (m, &node) in self.grid_node.iter().enumerate() {
                for x in 0..3 {
                    rhs[3 * node as usize + x] = (r[3 * m + x] / scale) as f32;
                }
            }
            let inner = (0.5 * tol * bnorm / rnorm).clamp(1e-3, 0.5);
            let job =
                GridJob { n: g.n, ke: &self.brick.ke, stiff: &stiff_grid, fixed: &fixed_grid, rhs: &rhs, tol: inner, max_iter: max_iter - total };
            let (d, iterations) = {
                let mut slot = GRID_SOLVER.lock().ok()?;
                let solved = slot.as_mut()?.solve(&job);
                match solved {
                    Ok(x) if x.0.len() == 3 * nodes && x.0.iter().all(|v| v.is_finite()) => x,
                    // A solver that fails is not asked again: the built-in one takes over.
                    _ => {
                        *slot = None;
                        return if round == 0 { None } else { Some(total) };
                    }
                }
            };
            total += iterations.max(1);
            for (m, &node) in self.grid_node.iter().enumerate() {
                for x in 0..3 {
                    if !self.fixed[3 * m + x] {
                        u[3 * m + x] += scale * d[3 * node as usize + x] as f64;
                    }
                }
            }
        }
        Some(total)
    }

    fn levels(&self) -> &Multigrid<'a> {
        self.mg.get_or_init(|| Multigrid::new(self.brick, self.g.n, &self.coords, &self.dofs, self.ndof, &self.fixed))
    }

    fn solve_here(&self, stiff: &[f64], u: &mut Vec<f64>, tol: f64, max_iter: usize) -> usize {
        let n = self.ndof;
        if u.len() != n {
            *u = vec![0.0; n];
        }
        let mg = self.levels().prepare(stiff);
        let b: Vec<f64> = (0..n).map(|i| if self.fixed[i] { 0.0 } else { self.force[i] }).collect();
        for i in 0..n {
            if self.fixed[i] {
                u[i] = 0.0;
            }
        }
        let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let mut r = vec![0.0; n];
        self.matvec(stiff, u, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        let mut z = mg.vcycle(0, &r);
        let mut p = z.clone();
        let mut rz: f64 = (0..n).map(|i| r[i] * z[i]).sum();
        let mut q = vec![0.0; n];
        for it in 0..max_iter {
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn / bnorm < tol {
                return it;
            }
            self.matvec(stiff, &p, &mut q);
            let pq: f64 = (0..n).map(|i| p[i] * q[i]).sum();
            if pq.abs() < 1e-300 {
                return it;
            }
            let alpha = rz / pq;
            let r_old = r.clone();
            for i in 0..n {
                u[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }
            z = mg.vcycle(0, &r);
            // Flexible (Polak-Ribiere) beta: the preconditioner is not exactly fixed.
            let rz_new: f64 = (0..n).map(|i| r[i] * z[i]).sum();
            let rz_mixed: f64 = (0..n).map(|i| r_old[i] * z[i]).sum();
            let beta = ((rz_new - rz_mixed) / rz).max(0.0);
            rz = rz_new;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
        max_iter
    }

    /// Strain energy density measure u_e^T KE u_e per element (unit stiffness).
    fn energies(&self, u: &[f64]) -> Vec<f64> {
        add_work(self.dofs.len());
        let ke = &self.brick.ke;
        self.dofs
            .iter()
            .map(|d| {
                let mut ue = [0.0; 24];
                for a in 0..24 {
                    ue[a] = u[d[a] as usize];
                }
                let mut s = 0.0;
                for a in 0..24 {
                    let mut acc = 0.0;
                    for b in 0..24 {
                        acc += ke[a][b] * ue[b];
                    }
                    s += ue[a] * acc;
                }
                s
            })
            .collect()
    }

    /// Von Mises stress at each element's centre, for modulus `e` (MPa) and voxel size h (mm).
    fn von_mises(&self, u: &[f64], e: f64) -> Vec<f64> {
        let h = self.g.h;
        let (b0, d) = (&self.brick.b0, &self.brick.d);
        self.dofs
            .iter()
            .map(|dd| {
                let mut eps = [0.0; 6];
                for r in 0..6 {
                    eps[r] = (0..24).map(|c| b0[r][c] * u[dd[c] as usize]).sum::<f64>() / h;
                }
                let mut s = [0.0; 6];
                for r in 0..6 {
                    s[r] = e * (0..6).map(|c| d[r][c] * eps[c]).sum::<f64>();
                }
                (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
                    .sqrt()
            })
            .collect()
    }

    /// Mean displacement of the loaded nodes along the force.
    fn load_displacement(&self, u: &[f64]) -> f64 {
        let fnorm: f64 = self.force.iter().map(|v| v * v).sum::<f64>().sqrt();
        if fnorm == 0.0 {
            return 0.0;
        }
        let work: f64 = self.force.iter().zip(u).map(|(f, u)| f * u).sum();
        // force entries sum to F over the loaded nodes; work / F is the weighted displacement.
        let total: f64 = (0..self.ndof / 3).map(|n| norm([self.force[3 * n], self.force[3 * n + 1], self.force[3 * n + 2]])).sum();
        work / total.max(1e-300)
    }
}

// ---------------------------------------------------------------- multigrid

/// One level of the geometric multigrid hierarchy.
struct Level {
    dofs: Vec<[u32; 24]>,
    ndof: usize,
    fixed: Vec<bool>,
    /// Child elements (indices into the finer level) of each element; empty on the finest.
    children: Vec<Vec<u32>>,
    /// For each node of the finer level: the coarse nodes (of this level) and weights that
    /// interpolate it. Empty on the finest level.
    prolong: Vec<Vec<(u32, f64)>>,
}

struct Multigrid<'a> {
    brick: &'a Brick,
    levels: Vec<Level>,
}

struct Prepared<'m, 'a> {
    mg: &'m Multigrid<'a>,
    stiff: Vec<Vec<f64>>,
    inv_diag: Vec<Vec<f64>>,
}

/// Elements per piece of [`apply_elements`]. The pieces, not the threads, fix the order of the
/// sums, so the result is the same on every machine.
const PIECE: usize = 8192;

/// One piece's share of `K x`: the first dof it touches and its sums from there on.
fn apply_piece(ke: &[[f64; 24]; 24], dofs: &[[u32; 24]], stiff: &[f64], x: &[f64]) -> (usize, Vec<f64>) {
    let (mut lo, mut hi) = (usize::MAX, 0usize);
    for d in dofs {
        for &a in d {
            lo = lo.min(a as usize);
            hi = hi.max(a as usize);
        }
    }
    if lo > hi {
        return (0, Vec::new());
    }
    let mut y = vec![0.0; hi - lo + 1];
    for (d, &s) in dofs.iter().zip(stiff) {
        if s == 0.0 {
            continue;
        }
        let mut ue = [0.0; 24];
        for (u, &da) in ue.iter_mut().zip(d.iter()) {
            *u = x[da as usize];
        }
        for (row, &da) in ke.iter().zip(d.iter()) {
            y[da as usize - lo] += s * dot24(row, &ue);
        }
    }
    (lo, y)
}

/// How many threads to compute on.
fn threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(32)
    }
}

/// `y = K x` without the supports, element by element, on all cores.
fn apply_elements(ke: &[[f64; 24]; 24], dofs: &[[u32; 24]], stiff: &[f64], x: &[f64], y: &mut [f64]) {
    add_work(dofs.len());
    y.iter_mut().for_each(|v| *v = 0.0);
    let pieces: Vec<(&[[u32; 24]], &[f64])> = dofs.chunks(PIECE).zip(stiff.chunks(PIECE)).collect();
    let workers = threads().min(pieces.len());
    let mut done: Vec<Option<(usize, Vec<f64>)>> = vec![None; pieces.len()];
    if workers <= 1 {
        for (slot, (d, s)) in done.iter_mut().zip(&pieces) {
            *slot = Some(apply_piece(ke, d, s, x));
        }
    } else {
        let pieces = &pieces;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|w| {
                    scope.spawn(move || {
                        pieces.iter().enumerate().skip(w).step_by(workers).map(|(i, (d, s))| (i, apply_piece(ke, d, s, x))).collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                // A worker only adds numbers; if one died all the same, its pieces are done here.
                for (i, r) in h.join().unwrap_or_default() {
                    done[i] = Some(r);
                }
            }
        });
        for (slot, (d, s)) in done.iter_mut().zip(pieces) {
            if slot.is_none() {
                *slot = Some(apply_piece(ke, d, s, x));
            }
        }
    }
    for (lo, part) in done.into_iter().flatten() {
        for (t, v) in y[lo..].iter_mut().zip(part) {
            *t += v;
        }
    }
}

fn level_matvec(brick: &Brick, l: &Level, stiff: &[f64], x: &[f64], y: &mut [f64]) {
    apply_elements(&brick.ke, &l.dofs, stiff, x, y);
    for i in 0..l.ndof {
        if l.fixed[i] {
            y[i] = x[i];
        }
    }
}

impl<'a> Multigrid<'a> {
    fn new(brick: &'a Brick, n0: [usize; 3], coords0: &[[usize; 3]], dofs0: &[[u32; 24]], ndof0: usize, fixed0: &[bool]) -> Multigrid<'a> {
        let mut levels = vec![Level { dofs: dofs0.to_vec(), ndof: ndof0, fixed: fixed0.to_vec(), children: Vec::new(), prolong: Vec::new() }];
        let mut n = n0;
        let mut coords: Vec<[usize; 3]> = coords0.to_vec();
        // Node grid id -> compact node of the current level.
        let node_id = |n: [usize; 3], i: usize, j: usize, k: usize| i + (n[0] + 1) * (j + (n[1] + 1) * k);
        let mut node_map: Vec<u32> = {
            let mut m = vec![NONE; (n[0] + 1) * (n[1] + 1) * (n[2] + 1)];
            for (e, c) in coords.iter().enumerate() {
                for (a, corner) in CORNERS.iter().enumerate() {
                    let (i, j, k) = (c[0] + (corner[0] > 0.0) as usize, c[1] + (corner[1] > 0.0) as usize, c[2] + (corner[2] > 0.0) as usize);
                    m[node_id(n, i, j, k)] = dofs0[e][3 * a] / 3;
                }
            }
            m
        };
        while levels.last().map(|l| l.ndof).unwrap_or(0) > 600 && n.iter().all(|&v| v >= 2) {
            let nc = n.map(|v| v.div_ceil(2));
            // Coarse elements and their children.
            let mut cmap = std::collections::HashMap::new();
            let mut ccoords: Vec<[usize; 3]> = Vec::new();
            let mut children: Vec<Vec<u32>> = Vec::new();
            for (e, c) in coords.iter().enumerate() {
                let key = [c[0] / 2, c[1] / 2, c[2] / 2];
                let idx = *cmap.entry(key).or_insert_with(|| {
                    ccoords.push(key);
                    children.push(Vec::new());
                    ccoords.len() - 1
                });
                children[idx].push(e as u32);
            }
            let mut cnode = vec![NONE; (nc[0] + 1) * (nc[1] + 1) * (nc[2] + 1)];
            let mut count = 0u32;
            let mut cdofs = Vec::with_capacity(ccoords.len());
            for c in &ccoords {
                let mut d = [0u32; 24];
                for (a, corner) in CORNERS.iter().enumerate() {
                    let id = node_id(nc, c[0] + (corner[0] > 0.0) as usize, c[1] + (corner[1] > 0.0) as usize, c[2] + (corner[2] > 0.0) as usize);
                    if cnode[id] == NONE {
                        cnode[id] = count;
                        count += 1;
                    }
                    for x in 0..3 {
                        d[3 * a + x] = 3 * cnode[id] + x as u32;
                    }
                }
                cdofs.push(d);
            }
            let fine = levels.last().map(|l| (l.ndof, l.fixed.clone())).unwrap_or_default();
            let fine_nodes = fine.0 / 3;
            let mut prolong: Vec<Vec<(u32, f64)>> = vec![Vec::new(); fine_nodes];
            let mut cfixed = vec![false; 3 * count as usize];
            for k in 0..=n[2] {
                for j in 0..=n[1] {
                    for i in 0..=n[0] {
                        let f = node_map[node_id(n, i, j, k)];
                        if f == NONE {
                            continue;
                        }
                        let mut list = Vec::new();
                        let opts = |v: usize| -> Vec<(usize, f64)> {
                            if v.is_multiple_of(2) { vec![(v / 2, 1.0)] } else { vec![(v / 2, 0.5), (v / 2 + 1, 0.5)] }
                        };
                        for (ci, wi) in opts(i) {
                            for (cj, wj) in opts(j) {
                                for (ck, wk) in opts(k) {
                                    let cn = cnode[node_id(nc, ci, cj, ck)];
                                    if cn != NONE {
                                        list.push((cn, wi * wj * wk));
                                    }
                                }
                            }
                        }
                        let total: f64 = list.iter().map(|x| x.1).sum();
                        if total > 0.0 {
                            for x in &mut list {
                                x.1 /= total;
                            }
                        }
                        // A coarse node next to a held fine node is held too. Taking only the
                        // coinciding (even) nodes can leave a coarse level with no support at all
                        // when the held face falls on odd nodes, and then it cannot help.
                        for &(cn, _) in &list {
                            for x in 0..3 {
                                if fine.1[3 * f as usize + x] {
                                    cfixed[3 * cn as usize + x] = true;
                                }
                            }
                        }
                        prolong[f as usize] = list;
                    }
                }
            }
            levels.push(Level { dofs: cdofs, ndof: 3 * count as usize, fixed: cfixed, children, prolong });
            n = nc;
            coords = ccoords;
            node_map = cnode;
            if levels.len() > 8 {
                break;
            }
        }
        Multigrid { brick, levels }
    }

    fn prepare(&self, stiff0: &[f64]) -> Prepared<'_, 'a> {
        let mut stiff = vec![stiff0.to_vec()];
        for l in &self.levels[1..] {
            let fine = stiff.last().cloned().unwrap_or_default();
            stiff.push(l.children.iter().map(|ch| ch.iter().map(|&c| fine[c as usize]).sum::<f64>() / 4.0).collect());
        }
        let inv_diag = self
            .levels
            .iter()
            .zip(&stiff)
            .map(|(l, s)| {
                let mut d = vec![0.0; l.ndof];
                for (e, dd) in l.dofs.iter().enumerate() {
                    for a in 0..24 {
                        d[dd[a] as usize] += s[e] * self.brick.ke[a][a];
                    }
                }
                d.iter().zip(&l.fixed).map(|(v, f)| if *f || *v <= 0.0 { 1.0 } else { 1.0 / v }).collect()
            })
            .collect();
        Prepared { mg: self, stiff, inv_diag }
    }
}

impl Prepared<'_, '_> {
    fn smooth(&self, l: usize, r: &[f64], z: &mut [f64], sweeps: usize) {
        let lv = &self.mg.levels[l];
        let mut az = vec![0.0; lv.ndof];
        for _ in 0..sweeps {
            level_matvec(self.mg.brick, lv, &self.stiff[l], z, &mut az);
            for i in 0..lv.ndof {
                z[i] += 0.6 * self.inv_diag[l][i] * (r[i] - az[i]);
                if lv.fixed[i] {
                    z[i] = 0.0;
                }
            }
        }
    }

    fn vcycle(&self, l: usize, r: &[f64]) -> Vec<f64> {
        let lv = &self.mg.levels[l];
        let mut z = vec![0.0; lv.ndof];
        if l + 1 == self.mg.levels.len() {
            // Coarsest: plenty of Jacobi-preconditioned CG.
            self.coarse_solve(l, r, &mut z);
            return z;
        }
        self.smooth(l, r, &mut z, 1);
        let mut az = vec![0.0; lv.ndof];
        level_matvec(self.mg.brick, lv, &self.stiff[l], &z, &mut az);
        let coarse = &self.mg.levels[l + 1];
        let mut rc = vec![0.0; coarse.ndof];
        for (f, list) in coarse.prolong.iter().enumerate() {
            for x in 0..3 {
                let res = if lv.fixed[3 * f + x] { 0.0 } else { r[3 * f + x] - az[3 * f + x] };
                for &(c, w) in list {
                    rc[3 * c as usize + x] += w * res;
                }
            }
        }
        for i in 0..coarse.ndof {
            if coarse.fixed[i] {
                rc[i] = 0.0;
            }
        }
        let zc = self.vcycle(l + 1, &rc);
        for (f, list) in coarse.prolong.iter().enumerate() {
            for x in 0..3 {
                if lv.fixed[3 * f + x] {
                    continue;
                }
                z[3 * f + x] += list.iter().map(|&(c, w)| w * zc[3 * c as usize + x]).sum::<f64>();
            }
        }
        self.smooth(l, r, &mut z, 1);
        z
    }

    fn coarse_solve(&self, l: usize, b: &[f64], x: &mut [f64]) {
        let lv = &self.mg.levels[l];
        let n = lv.ndof;
        let d = &self.inv_diag[l];
        let mut r: Vec<f64> = (0..n).map(|i| if lv.fixed[i] { 0.0 } else { b[i] }).collect();
        let bn = r.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let mut z: Vec<f64> = (0..n).map(|i| r[i] * d[i]).collect();
        let mut p = z.clone();
        let mut rz: f64 = (0..n).map(|i| r[i] * z[i]).sum();
        let mut q = vec![0.0; n];
        for _ in 0..60 {
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() / bn < 1e-4 {
                break;
            }
            level_matvec(self.mg.brick, lv, &self.stiff[l], &p, &mut q);
            let pq: f64 = (0..n).map(|i| p[i] * q[i]).sum();
            if pq.abs() < 1e-300 {
                break;
            }
            let a = rz / pq;
            for i in 0..n {
                x[i] += a * p[i];
                r[i] -= a * q[i];
                z[i] = r[i] * d[i];
            }
            let rz2: f64 = (0..n).map(|i| r[i] * z[i]).sum();
            let beta = rz2 / rz;
            rz = rz2;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
    }
}

// ---------------------------------------------------------------- problem and styles

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Organic,
    Tree,
    Lattice,
}

pub const STYLE_ORGANIC: &str = "organic";
pub const STYLE_TREE: &str = "tree";
pub const STYLE_LATTICE: &str = "lattice";

#[derive(Clone, Debug)]
pub struct Problem {
    pub grid: Grid,
    /// Voxels whose nodes are held.
    pub fixed: Vec<bool>,
    /// Voxels whose nodes carry the force.
    pub load: Vec<bool>,
    /// Voxels that always stay (around supports, load and holes).
    pub keep: Vec<bool>,
    /// Each support's voxels, for the tree.
    pub supports: Vec<Vec<usize>>,
    pub force: V3,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub style: Style,
    pub material: Material,
    pub safety: f64,
    /// Fraction of the design space to keep; None = as little as the stress allows.
    pub volume: Option<f64>,
    pub min_wall: f64,
    /// The most the loaded faces may move, in mm; 0 for no limit.
    pub max_deflection: f64,
    /// Compute with the installed [`GridSolver`] (a GPU) when there is one.
    pub accelerate: bool,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    /// Smoothed material field per voxel, 0..1 (0.5 is the surface).
    pub field: Vec<f64>,
    pub solid_voxels: usize,
    pub design_voxels: usize,
    pub stress: f64,
    pub allowed: f64,
    pub deflection: f64,
    pub passed: bool,
    pub iterations: usize,
    pub note: Vec<String>,
}

/// Strength allowed: printed parts are weaker across layers, and the print direction of a
/// generated shape is not known yet, so 70 % of the strength along the lines.
pub fn allowed_stress(s: &Settings) -> f64 {
    0.7 * s.material.strength / s.safety
}

struct Check {
    stress: f64,
    deflection: f64,
    connected: bool,
}

/// Keeps only the solid voxels connected to a support.
fn connected_part(g: &Grid, mask: &[bool], fixed: &[bool]) -> Vec<bool> {
    let mut out = vec![false; g.len()];
    let mut stack: Vec<usize> = (0..g.len()).filter(|&v| mask[v] && fixed[v]).collect();
    for &v in &stack {
        out[v] = true;
    }
    let mut nb = Vec::new();
    while let Some(v) = stack.pop() {
        g.neighbours(v, &mut nb);
        for &(w, d) in &nb {
            // Face neighbours only: edge contact carries no load in a brick model.
            if d < 1.01 && mask[w] && !out[w] {
                out[w] = true;
                stack.push(w);
            }
        }
    }
    out
}

fn check_binary(p: &Problem, brick: &Brick, mask: &[bool], s: &Settings, u: &mut Vec<f64>) -> Check {
    let g = &p.grid;
    let part = connected_part(g, mask, &p.fixed);
    let connected = (0..g.len()).any(|v| part[v] && p.load[v]);
    if !connected {
        return Check { stress: f64::INFINITY, deflection: f64::INFINITY, connected: false };
    }
    let elems: Vec<usize> = (0..g.len()).filter(|&v| part[v]).collect();
    let fem = Fem::new(g, brick, elems, &p.fixed, &p.load, p.force, s.accelerate);
    let e = s.material.modulus;
    let stiff = vec![e * g.h; fem.elems.len()];
    u.clear();
    fem.solve(&stiff, u, 1e-4, 120);
    let vm = fem.von_mises(u, e);
    // Peaks right at the supports and the load are artefacts of holding whole voxels.
    let near = dilate(g, &p.fixed.iter().zip(&p.load).map(|(a, b)| *a || *b).collect::<Vec<_>>(), 1.0);
    let stress = fem.elems.iter().zip(&vm).filter(|(v, _)| !near[**v]).map(|(_, s)| *s).fold(0.0, f64::max);
    Check { stress, deflection: fem.load_displacement(u), connected }
}

/// Box blur of a 0/1 mask, kept inside the design space, as a smooth field.
fn smooth(g: &Grid, x: &[f64], keep: &[bool]) -> Vec<f64> {
    let mut out = vec![0.0; g.len()];
    let mut nb = Vec::new();
    for v in 0..g.len() {
        if !g.inside[v] {
            continue;
        }
        g.neighbours(v, &mut nb);
        let mut sum = 2.0 * x[v];
        let mut w = 2.0;
        for &(n, d) in &nb {
            if d < 1.5 {
                let wt = if d < 1.01 { 1.0 } else { 0.5 };
                sum += wt * if g.inside[n] { x[n] } else { 0.0 };
                w += wt;
            }
        }
        // Blur for smooth surfaces, but never across 0.5: thin members stay, holes stay open.
        let b = sum / w;
        out[v] = if keep[v] {
            1.0
        } else if x[v] > 0.5 {
            0.6 + 0.4 * b
        } else {
            0.45 * b
        };
    }
    out
}

/// Fills voxels so that no 2x2x2 block has solid (or empty) voxels touching only at an edge
/// or a corner: such blocks make the surface non-manifold.
pub fn fix_diagonals(g: &Grid, mask: &mut [bool]) {
    for _ in 0..4 {
        let mut changed = false;
        for k in 0..g.n[2].saturating_sub(1) {
            for j in 0..g.n[1].saturating_sub(1) {
                for i in 0..g.n[0].saturating_sub(1) {
                    let ids: [usize; 8] = std::array::from_fn(|c| g.id(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)));
                    let solid: u8 = (0..8).filter(|&c| mask[ids[c]]).fold(0, |a, c| a | (1 << c));
                    if solid == 0 || solid == 255 {
                        continue;
                    }
                    if face_connected(solid) && face_connected(!solid) {
                        continue;
                    }
                    for &v in &ids {
                        if g.inside[v] && !mask[v] {
                            mask[v] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Whether the corners in `set` (bits of a 2x2x2 block) are connected through shared faces.
fn face_connected(set: u8) -> bool {
    if set == 0 {
        return true;
    }
    let start = set.trailing_zeros() as u8;
    let mut seen = 1u8 << start;
    let mut stack = vec![start];
    while let Some(c) = stack.pop() {
        for bit in [1u8, 2, 4] {
            let n = c ^ bit;
            if set & (1 << n) != 0 && seen & (1 << n) == 0 {
                seen |= 1 << n;
                stack.push(n);
            }
        }
    }
    seen == set
}

fn result(p: &Problem, s: &Settings, mask: &[bool], chk: &Check, iterations: usize, note: Vec<String>) -> Outcome {
    let g = &p.grid;
    let mut mask = mask.to_vec();
    fix_diagonals(g, &mut mask);
    let mask = &mask[..];
    let x: Vec<f64> = mask.iter().map(|m| if *m { 1.0 } else { 0.0 }).collect();
    Outcome {
        field: smooth(g, &x, &p.keep),
        solid_voxels: mask.iter().filter(|m| **m).count(),
        design_voxels: g.inside.iter().filter(|m| **m).count(),
        stress: chk.stress,
        allowed: allowed_stress(s),
        deflection: chk.deflection,
        passed: chk.connected && chk.stress <= allowed_stress(s),
        iterations,
        note,
    }
}

/// Runs the chosen style. `time_left` returns false when the time budget is spent.
pub fn generate(p: &Problem, s: &Settings, time_left: &mut dyn FnMut() -> bool) -> Result<Outcome, String> {
    if !p.fixed.iter().any(|v| *v) {
        return Err("nothing is held: no held face lies in the design space".into());
    }
    if !p.load.iter().any(|v| *v) {
        return Err("the loaded face does not lie in the design space".into());
    }
    let brick = Brick::new();
    match s.style {
        Style::Organic => organic(p, s, &brick, time_left),
        Style::Tree => tree(p, s, &brick, time_left),
        Style::Lattice => lattice(p, s, &brick, time_left),
    }
}

fn organic(p: &Problem, s: &Settings, brick: &Brick, time_left: &mut dyn FnMut() -> bool) -> Result<Outcome, String> {
    let g = &p.grid;
    let elems: Vec<usize> = (0..g.len()).filter(|&v| g.inside[v]).collect();
    let ne = elems.len();
    let fem = Fem::new(g, brick, elems.clone(), &p.fixed, &p.load, p.force, s.accelerate);
    let e0 = s.material.modulus;
    let emin = e0 * 1e-4;
    let pen = 3.0;
    let keep: Vec<bool> = elems.iter().map(|&v| p.keep[v]).collect();
    // Filter neighbourhood: a cone as wide as the thinnest member wanted, at least 1.5 voxels.
    // It is what keeps the members from getting thinner than that.
    let radius = (s.min_wall / g.h).clamp(1.5, 6.0);
    let mut local = vec![u32::MAX; g.len()];
    for (e, &v) in elems.iter().enumerate() {
        local[v] = e as u32;
    }
    let reach = radius.ceil() as i64;
    let mut stencil: Vec<([i64; 3], f64)> = Vec::new();
    for dk in -reach..=reach {
        for dj in -reach..=reach {
            for di in -reach..=reach {
                let d = ((di * di + dj * dj + dk * dk) as f64).sqrt();
                if d < radius {
                    stencil.push(([di, dj, dk], radius - d));
                }
            }
        }
    }
    let cells: Vec<[i64; 3]> = elems.iter().map(|&v| g.ijk(v).map(|c| c as i64)).collect();
    let n = g.n.map(|c| c as i64);
    // The filtered sensitivities, piece by piece on all cores.
    let filter = |x: &[f64], dc: &[f64]| -> Vec<f64> {
        let piece = |range: std::ops::Range<usize>| -> Vec<f64> {
            range
                .map(|e| {
                    let c = cells[e];
                    let (mut num, mut den) = (0.0, 0.0);
                    for &(o, w) in &stencil {
                        let q = [c[0] + o[0], c[1] + o[1], c[2] + o[2]];
                        if q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= n[0] || q[1] >= n[1] || q[2] >= n[2] {
                            continue;
                        }
                        let f = local[g.id(q[0] as usize, q[1] as usize, q[2] as usize)];
                        if f != u32::MAX {
                            num += w * x[f as usize] * dc[f as usize];
                            den += w;
                        }
                    }
                    num / (x[e].max(1e-3) * den)
                })
                .collect()
        };
        let ranges: Vec<std::ops::Range<usize>> = (0..ne).step_by(PIECE).map(|a| a..(a + PIECE).min(ne)).collect();
        let workers = threads().min(ranges.len());
        if workers <= 1 {
            return ranges.into_iter().flat_map(piece).collect();
        }
        let (ranges, piece) = (&ranges, &piece);
        let mut done: Vec<Vec<f64>> = vec![Vec::new(); ranges.len()];
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|w| {
                    scope.spawn(move || ranges.iter().enumerate().skip(w).step_by(workers).map(|(i, r)| (i, piece(r.clone()))).collect::<Vec<_>>())
                })
                .collect();
            for h in handles {
                for (i, r) in h.join().unwrap_or_default() {
                    done[i] = r;
                }
            }
        });
        for (slot, r) in done.iter_mut().zip(ranges) {
            if slot.len() != r.len() {
                *slot = piece(r.clone());
            }
        }
        done.into_iter().flatten().collect()
    };

    let schedule: Vec<f64> = match s.volume {
        Some(v) => vec![v.clamp(0.05, 1.0)],
        None => vec![0.5, 0.4, 0.32, 0.26, 0.21, 0.17, 0.14, 0.11, 0.09, 0.075, 0.062, 0.051, 0.042, 0.035, 0.029, 0.024, 0.02],
    };
    let mut x = vec![schedule[0]; ne];
    for e in 0..ne {
        if keep[e] {
            x[e] = 1.0;
        }
    }
    let mut u = Vec::new();
    let mut ub = Vec::new();
    let mut best: Option<(Vec<bool>, Check, f64)> = None;
    let mut iterations = 0;
    let mut notes = Vec::new();
    let allowed = allowed_stress(s);
    let mut out_of_time = false;
    for (step, &vf) in schedule.iter().enumerate() {
        if out_of_time {
            break;
        }
        let n_iter = if step == 0 { if s.volume.is_some() { 36 } else { 14 } } else { 6 };
        for _ in 0..n_iter {
            if !time_left() {
                notes.push("Stopped early: this is the last shape that was checked.".into());
                out_of_time = true;
                break;
            }
            iterations += 1;
            let stiff: Vec<f64> = x.iter().map(|xe| (emin + xe.powf(pen) * (e0 - emin)) * g.h).collect();
            fem.solve(&stiff, &mut u, 5e-3, 40);
            let en = fem.energies(&u);
            let dc: Vec<f64> = (0..ne).map(|e| -pen * x[e].powf(pen - 1.0) * (e0 - emin) * g.h * en[e]).collect();
            // Sensitivity filter.
            let dcf = filter(&x, &dc);
            // Optimality criteria update.
            let target = vf * ne as f64;
            let (mut l1, mut l2) = (0.0f64, 1e12f64);
            let mv = 0.2;
            let mut xn = x.clone();
            for _ in 0..80 {
                let lm = 0.5 * (l1 + l2);
                for e in 0..ne {
                    xn[e] = if keep[e] {
                        1.0
                    } else {
                        let be = (-dcf[e] / lm).max(0.0).sqrt();
                        (x[e] * be).clamp((x[e] - mv).max(1e-3), (x[e] + mv).min(1.0))
                    };
                }
                if xn.iter().sum::<f64>() > target {
                    l1 = lm;
                } else {
                    l2 = lm;
                }
                if (l2 - l1) / (l1 + l2).max(1e-300) < 1e-4 {
                    break;
                }
            }
            x = xn;
        }
        // Stress check of the 0/1 design this step reached.
        let mut mask = vec![false; g.len()];
        for (e, &v) in elems.iter().enumerate() {
            mask[v] = x[e] > 0.5 || keep[e];
        }
        let chk = check_binary(p, brick, &mask, s, &mut ub);
        let ok = chk.connected && chk.stress <= allowed && (!(s.max_deflection > 0.0) || chk.deflection <= s.max_deflection);
        if ok || best.is_none() || s.volume.is_some() {
            best = Some((connected_part(g, &mask, &p.fixed), chk, vf));
        }
        if !ok && s.volume.is_none() {
            if step == 0 {
                notes.push("The stress is too high even with half the design space filled: more space, a stronger material or less force.".into());
            }
            break;
        }
    }
    let (mask, chk, vf) = best.ok_or_else(|| "no shape carries the load".to_string())?;
    notes.push(format!("Material left: {:.0} % of the design space.", vf * 100.0));
    Ok(result(p, s, &mask, &chk, iterations, notes))
}

/// Distance of every inside voxel to the outside, in voxels (city-block BFS).
fn depth(g: &Grid) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; g.len()];
    let mut queue = std::collections::VecDeque::new();
    for v in 0..g.len() {
        if g.on_surface(v) {
            d[v] = 1.0;
            queue.push_back(v);
        }
    }
    let mut nb = Vec::new();
    while let Some(v) = queue.pop_front() {
        g.neighbours(v, &mut nb);
        for &(w, dist) in &nb {
            if dist < 1.01 && g.inside[w] && d[w] > d[v] + 1.0 {
                d[w] = d[v] + 1.0;
                queue.push_back(w);
            }
        }
    }
    d
}

fn centroid_voxel(g: &Grid, set: &[usize]) -> Option<usize> {
    if set.is_empty() {
        return None;
    }
    let mut c = [0.0; 3];
    for &v in set {
        let p = g.center(v);
        for a in 0..3 {
            c[a] += p[a] / set.len() as f64;
        }
    }
    set.iter().copied().min_by(|&a, &b| norm(sub(g.center(a), c)).total_cmp(&norm(sub(g.center(b), c))))
}

fn tree(p: &Problem, s: &Settings, brick: &Brick, time_left: &mut dyn FnMut() -> bool) -> Result<Outcome, String> {
    let g = &p.grid;
    let load: Vec<usize> = (0..g.len()).filter(|&v| p.load[v]).collect();
    let start = centroid_voxel(g, &load).ok_or("the loaded face does not lie in the design space")?;
    let dep = depth(g);
    // Dijkstra through the inside, preferring to stay away from the skin.
    let mut dist = vec![f64::INFINITY; g.len()];
    let mut prev = vec![usize::MAX; g.len()];
    let mut heap = std::collections::BinaryHeap::new();
    dist[start] = 0.0;
    heap.push((std::cmp::Reverse(0u64), start));
    let mut nb = Vec::new();
    while let Some((std::cmp::Reverse(dk), v)) = heap.pop() {
        let d = dk as f64 / 1e6;
        if d > dist[v] + 1e-9 {
            continue;
        }
        g.neighbours(v, &mut nb);
        for &(w, len) in &nb {
            if !g.inside[w] {
                continue;
            }
            let nd = d + len * (1.0 + 1.5 / dep[w].max(1.0));
            if nd < dist[w] {
                dist[w] = nd;
                prev[w] = v;
                heap.push((std::cmp::Reverse((nd * 1e6) as u64), w));
            }
        }
    }
    // Paths to each support; count how many supports each voxel carries.
    let mut carry = vec![0u32; g.len()];
    let mut reached = 0;
    for sup in &p.supports {
        let Some(t) = centroid_voxel(g, sup) else { continue };
        if !dist[t].is_finite() {
            continue;
        }
        reached += 1;
        let mut v = t;
        let mut guard = 0;
        while v != usize::MAX && guard < g.len() {
            carry[v] += 1;
            v = prev[v];
            guard += 1;
        }
    }
    if reached == 0 {
        return Err("Zwischen Last und Befestigung gibt es im Bauraum keinen Weg.".into());
    }
    let path: Vec<usize> = (0..g.len()).filter(|&v| carry[v] > 0).collect();
    let min_r = (s.min_wall / g.h / 2.0).max(0.6);
    let paint = |scale: f64| -> Vec<bool> {
        let mut mask = p.keep.clone();
        for &v in &path {
            let r = (scale * (carry[v] as f64).sqrt()).max(min_r);
            let ri = r.ceil() as i64;
            let [i, j, k] = g.ijk(v);
            for dk in -ri..=ri {
                for dj in -ri..=ri {
                    for di in -ri..=ri {
                        if ((di * di + dj * dj + dk * dk) as f64) > r * r {
                            continue;
                        }
                        let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                        if a < 0 || b < 0 || c < 0 || a >= g.n[0] as i64 || b >= g.n[1] as i64 || c >= g.n[2] as i64 {
                            continue;
                        }
                        let w = g.id(a as usize, b as usize, c as usize);
                        if g.inside[w] {
                            mask[w] = true;
                        }
                    }
                }
            }
        }
        mask
    };
    let allowed = allowed_stress(s);
    let mut u = Vec::new();
    // Bisection on the branch thickness.
    let (mut lo, mut hi) = (min_r, (g.n.iter().copied().max().unwrap_or(1) as f64) / 2.0);
    let mut best: Option<(Vec<bool>, Check)> = None;
    let mut iterations = 0;
    let hi_mask = paint(hi);
    let chk = check_binary(p, brick, &hi_mask, s, &mut u);
    iterations += 1;
    if !(chk.connected && chk.stress <= allowed) {
        let notes = vec!["The stress is too high even with the thickest branches: more space, a stronger material or less force.".to_owned()];
        return Ok(result(p, s, &hi_mask, &chk, iterations, notes));
    }
    best = best.or(Some((hi_mask, chk)));
    for _ in 0..7 {
        if !time_left() {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let mask = paint(mid);
        let chk = check_binary(p, brick, &mask, s, &mut u);
        iterations += 1;
        if chk.connected && chk.stress <= allowed {
            hi = mid;
            best = Some((mask, chk));
        } else {
            lo = mid;
        }
        if hi - lo < 0.15 {
            break;
        }
    }
    let (mask, chk) = best.ok_or("no shape carries the load")?;
    let notes = vec![format!("{} branches from the load to {} held faces.", path.len().min(1) * reached, reached)];
    Ok(result(p, s, &mask, &chk, iterations, notes))
}

fn lattice(p: &Problem, s: &Settings, brick: &Brick, time_left: &mut dyn FnMut() -> bool) -> Result<Outcome, String> {
    let g = &p.grid;
    let dep = depth(g);
    let allowed = allowed_stress(s);
    let mut u = Vec::new();
    // From lightest to heaviest: (skin voxels, cell size in voxels).
    let options = [(1.0, 6), (1.0, 5), (1.0, 4), (2.0, 5), (2.0, 4), (2.0, 3), (3.0, 3)];
    let mut last = None;
    let mut iterations = 0;
    for (skin, cell) in options {
        if !time_left() {
            break;
        }
        let mut mask = p.keep.clone();
        for v in 0..g.len() {
            if !g.inside[v] {
                continue;
            }
            let [i, j, k] = g.ijk(v);
            let (a, b, c) = (i % cell == 0, j % cell == 0, k % cell == 0);
            if dep[v] <= skin || (a && b) || (b && c) || (a && c) {
                mask[v] = true;
            }
        }
        let chk = check_binary(p, brick, &mask, s, &mut u);
        iterations += 1;
        let ok = chk.connected && chk.stress <= allowed;
        let note = vec![format!("Shell {:.1} mm, lattice cell {:.1} mm.", skin * g.h, cell as f64 * g.h)];
        if ok {
            return Ok(result(p, s, &mask, &chk, iterations, note));
        }
        last = Some((mask, chk, note));
    }
    let (mask, chk, mut note) = last.ok_or("no shape carries the load")?;
    note.push("Auch das dichteste Gitter erreicht die Sicherheit nicht.".into());
    Ok(result(p, s, &mask, &chk, iterations, note))
}

// ---------------------------------------------------------------- surface

/// A closed triangle surface of field = 0.5, sampled `up` times finer than the voxels
/// (trilinear), by surface nets, then Taubin-smoothed.
/// Exact cuts the surface keeps: hole cylinders (origin, unit axis, radius).
pub type Cut = (V3, V3, f64);

/// A half-space the surface stays in: a point on its plane and the normal pointing into it.
pub type HalfSpace = (V3, V3);

/// Distance from `p` to triangle `t`.
pub fn point_triangle_distance(p: V3, t: &[V3; 3]) -> f64 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let at = |q: V3| norm(sub(p, q));
    let along = |o: V3, d: V3, t: f64| [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
    if d1 <= 0.0 && d2 <= 0.0 {
        return at(a);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return at(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return at(along(a, ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return at(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return at(along(a, ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return at(along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let denom = va + vb + vc;
    if !(denom.abs() > 0.0) {
        return at(a);
    }
    let (v, w) = (vb / denom, vc / denom);
    at([a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w])
}

/// The design voxels whose centre lies within `reach` of the triangles of a face.
pub fn voxels_near(g: &Grid, tris: &[[V3; 3]], reach: f64) -> Vec<usize> {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for t in tris {
        for p in t {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a] - reach);
                hi[a] = hi[a].max(p[a] + reach);
            }
        }
    }
    (0..g.len())
        .filter(|&v| {
            if !g.inside[v] {
                return false;
            }
            let c = g.center(v);
            (0..3).all(|a| c[a] >= lo[a] && c[a] <= hi[a]) && tris.iter().any(|t| point_triangle_distance(c, t) <= reach)
        })
        .collect()
}

pub fn surface(g: &Grid, field: &[f64], up: usize, smooth_iters: usize, cuts: &[Cut], clip: Option<(V3, V3)>, within: &[HalfSpace]) -> Mesh {
    let up = up.max(1);
    // Padded sample grid: voxel centres with one empty layer around, refined `up` times.
    // Samples sit a quarter step off the voxel centres and faces, so a 0/1 field never puts one
    // exactly on the surface.
    let pn = [0, 1, 2].map(|a| (g.n[a] + 1) * up);
    let val = |i: i64, j: i64, k: i64| -> f64 {
        if i < 0 || j < 0 || k < 0 || i >= g.n[0] as i64 || j >= g.n[1] as i64 || k >= g.n[2] as i64 {
            0.0
        } else {
            field[g.id(i as usize, j as usize, k as usize)]
        }
    };
    let mut s = vec![0.0f64; pn[0] * pn[1] * pn[2]];
    let sid = |i: usize, j: usize, k: usize| i + pn[0] * (j + pn[1] * k);
    for k in 0..pn[2] {
        for j in 0..pn[1] {
            for i in 0..pn[0] {
                // Position in voxel-centre units, padded by one.
                let f = [i, j, k].map(|c| (c as f64 + 0.5) / up as f64 - 1.0);
                let b = f.map(|c| c.floor());
                let t = [f[0] - b[0], f[1] - b[1], f[2] - b[2]];
                let (x0, y0, z0) = (b[0] as i64, b[1] as i64, b[2] as i64);
                let mut acc = 0.0;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let w = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                                * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                                * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                            if w > 0.0 {
                                acc += w * val(x0 + dx, y0 + dy, z0 + dz);
                            }
                        }
                    }
                }
                let mut v = acc - 0.5;
                if !cuts.is_empty() {
                    let p = [0, 1, 2].map(|a| g.origin[a] + (f[a] + 0.5) * g.h);
                    for (o, ax, r) in cuts {
                        v = v.min((axis_dist(p, *o, *ax) - r) / (2.0 * g.h));
                    }
                }
                if !within.is_empty() {
                    let p = [0, 1, 2].map(|a| g.origin[a] + (f[a] + 0.5) * g.h);
                    for (point, normal) in within {
                        v = v.min(dot(sub(p, *point), unit(*normal)) / (2.0 * g.h));
                    }
                }
                if let Some((lo, hi)) = clip {
                    let p = [0, 1, 2].map(|a| g.origin[a] + (f[a] + 0.5) * g.h);
                    for a in 0..3 {
                        v = v.min((p[a] - lo[a]) / (2.0 * g.h)).min((hi[a] - p[a]) / (2.0 * g.h));
                    }
                }
                // Never exactly on the surface: a zero sample would put the vertices of
                // neighbouring cells on the same point.
                s[sid(i, j, k)] = if v.abs() < 1e-7 { 1e-7 } else { v };
            }
        }
    }
    let step = g.h / up as f64;
    let world = |p: [f64; 3]| -> V3 { [0, 1, 2].map(|a| g.origin[a] + ((p[a] + 0.5) / up as f64 - 0.5) * g.h) };
    // Marching tetrahedra: every sample cube is split into six tetrahedra around its main
    // diagonal, the same way in every cube, so neighbouring cubes share their face diagonals.
    // Inside a tetrahedron the field is linear: no ambiguous cases, so the result is always a
    // closed 2-manifold. One vertex per sample edge with a sign change, shared by every
    // triangle that uses that edge.
    let mut mesh = Mesh::default();
    let mut edge_vertex: std::collections::HashMap<(usize, usize), u32> = std::collections::HashMap::new();
    let corner_of = |c: usize| [c & 1, (c >> 1) & 1, (c >> 2) & 1];
    const TETS: [[usize; 4]; 6] = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];
    for k in 0..pn[2] - 1 {
        for j in 0..pn[1] - 1 {
            for i in 0..pn[0] - 1 {
                let ids: [usize; 8] = std::array::from_fn(|c| {
                    let o = corner_of(c);
                    sid(i + o[0], j + o[1], k + o[2])
                });
                let inside = ids.iter().filter(|&&x| s[x] > 0.0).count();
                if inside == 0 || inside == 8 {
                    continue;
                }
                let pos_of = |c: usize| -> [f64; 3] {
                    let o = corner_of(c);
                    [(i + o[0]) as f64, (j + o[1]) as f64, (k + o[2]) as f64]
                };
                for tet in TETS {
                    let ins: Vec<usize> = tet.iter().copied().filter(|&c| s[ids[c]] > 0.0).collect();
                    let outs: Vec<usize> = tet.iter().copied().filter(|&c| s[ids[c]] <= 0.0).collect();
                    if ins.is_empty() || outs.is_empty() {
                        continue;
                    }
                    let mut vert = |a: usize, b: usize| -> u32 {
                        let (ia, ib) = (ids[a], ids[b]);
                        let key = if ia < ib { (ia, ib) } else { (ib, ia) };
                        if let Some(&v) = edge_vertex.get(&key) {
                            return v;
                        }
                        let (va, vb) = (s[ia], s[ib]);
                        // Kept off the samples, so two vertices never land on the same point.
                        let t = (va / (va - vb)).clamp(0.05, 0.95);
                        let (pa, pb) = (pos_of(a), pos_of(b));
                        let id = mesh.pos.len() as u32;
                        mesh.pos.push(world([0, 1, 2].map(|x| pa[x] + t * (pb[x] - pa[x]))));
                        edge_vertex.insert(key, id);
                        id
                    };
                    let polys: Vec<[u32; 3]> = match (ins.len(), outs.len()) {
                        (1, 3) => vec![[vert(ins[0], outs[0]), vert(ins[0], outs[1]), vert(ins[0], outs[2])]],
                        (3, 1) => vec![[vert(ins[0], outs[0]), vert(ins[1], outs[0]), vert(ins[2], outs[0])]],
                        _ => {
                            let (a, b, c, d) = (ins[0], ins[1], outs[0], outs[1]);
                            let (ac, ad, bd, bc) = (vert(a, c), vert(a, d), vert(b, d), vert(b, c));
                            vec![[ac, ad, bd], [ac, bd, bc]]
                        }
                    };
                    // Outward: from the inside corners towards the outside corners.
                    let centre = |cs: &[usize]| -> [f64; 3] {
                        let mut m = [0.0; 3];
                        for &c in cs {
                            let p = pos_of(c);
                            for x in 0..3 {
                                m[x] += p[x] / cs.len() as f64;
                            }
                        }
                        m
                    };
                    let out_dir = sub(centre(&outs), centre(&ins));
                    for t in polys {
                        let [p0, p1, p2] = t.map(|v| mesh.pos[v as usize]);
                        let n = cross(sub(p1, p0), sub(p2, p0));
                        let t = if dot(n, out_dir) >= 0.0 { t } else { [t[0], t[2], t[1]] };
                        mesh.tri.push(t.map(|v| v as usize));
                    }
                }
            }
        }
    }
    taubin(&mut mesh, smooth_iters);
    // Holes back to their exact radius, and nothing outside the design space's box.
    for p in &mut mesh.pos {
        for (o, ax, r) in cuts {
            let a = unit(*ax);
            let d = sub(*p, *o);
            let t = dot(d, a);
            let radial = [d[0] - t * a[0], d[1] - t * a[1], d[2] - t * a[2]];
            let dist = norm(radial);
            if dist > 1e-9 && (dist - r).abs() < 0.6 * step {
                for x in 0..3 {
                    p[x] = o[x] + t * a[x] + radial[x] / dist * r;
                }
            }
        }
    }
    mesh
}

/// Taubin smoothing: shrink-free Laplacian smoothing.
fn taubin(m: &mut Mesh, iters: usize) {
    let n = m.pos.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for t in &m.tri {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if !adj[a].contains(&b) {
                adj[a].push(b);
            }
            if !adj[b].contains(&a) {
                adj[b].push(a);
            }
        }
    }
    for _ in 0..iters {
        for f in [0.5, -0.53] {
            let old = m.pos.clone();
            for v in 0..n {
                if adj[v].is_empty() {
                    continue;
                }
                let mut c = [0.0; 3];
                for &w in &adj[v] {
                    for a in 0..3 {
                        c[a] += old[w][a] / adj[v].len() as f64;
                    }
                }
                for a in 0..3 {
                    m.pos[v][a] = old[v][a] + f * (c[a] - old[v][a]);
                }
            }
        }
    }
}

pub fn mesh_volume(m: &Mesh) -> f64 {
    m.tri.iter().map(|t| dot(m.pos[t[0]], cross(m.pos[t[1]], m.pos[t[2]]))).sum::<f64>() / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(h: f64) -> Problem {
        // A 100 x 40 x 30 block: held at x = 0 face, loaded downward at the x = 100 face.
        let mesh = box_mesh([0.0, 0.0, 0.0], [100.0, 40.0, 30.0]);
        let grid = voxelize(&mesh, h);
        let fix = Face { surface: Surface::Plane { point: [0.0, 20.0, 15.0], normal: [-1.0, 0.0, 0.0] }, point: [0.0, 20.0, 15.0] };
        let ld = Face { surface: Surface::Plane { point: [100.0, 20.0, 15.0], normal: [1.0, 0.0, 0.0] }, point: [100.0, 20.0, 15.0] };
        let fv = face_voxels(&grid, &fix);
        let lv = face_voxels(&grid, &ld);
        let mut fixed = vec![false; grid.len()];
        let mut load = vec![false; grid.len()];
        fv.iter().for_each(|&v| fixed[v] = true);
        lv.iter().for_each(|&v| load[v] = true);
        let both: Vec<bool> = fixed.iter().zip(&load).map(|(a, b)| *a || *b).collect();
        let keep = dilate(&grid, &both, 1.0);
        Problem { grid, fixed, load, keep, supports: vec![fv], force: [0.0, 0.0, -300.0] }
    }

    fn settings(style: Style) -> Settings {
        Settings {
            style,
            material: Material { strength: 35.0, modulus: 1900.0, density: 1.27 },
            safety: 2.5,
            volume: None,
            min_wall: 2.0,
            max_deflection: 0.0,
            accelerate: false,
        }
    }

    #[test]
    fn voxelize_box_counts() {
        let g = voxelize(&box_mesh([0.0; 3], [10.0, 10.0, 10.0]), 1.0);
        assert_eq!(g.inside.iter().filter(|v| **v).count(), 1000);
    }

    #[test]
    fn cantilever_tip_deflection_matches_beam_theory() {
        // Solid bar 100 x 40 x 30 (h=30 in z), F=300 N down: delta = F L^3 / (3 E I).
        let p = problem(5.0);
        let s = settings(Style::Organic);
        let brick = Brick::new();
        let mut u = Vec::new();
        let chk = check_binary(&p, &brick, &p.grid.inside.clone(), &s, &mut u);
        let i = 40.0 * 30f64.powi(3) / 12.0;
        let theory = 300.0 * 100f64.powi(3) / (3.0 * 1900.0 * i);
        // Coarse linear bricks are stiffer than theory; within a factor of two is right.
        assert!(chk.deflection > 0.4 * theory && chk.deflection < 1.2 * theory, "{} vs {theory}", chk.deflection);
    }

    fn run(style: Style, h: f64) -> Outcome {
        let p = problem(h);
        let s = settings(style);
        let t = std::time::Instant::now();
        let out = generate(&p, &s, &mut || t.elapsed().as_secs_f64() < 60.0).unwrap();
        let m = surface(&p.grid, &out.field, 2, 6, &[], None, &[]);
        println!(
            "{style:?} h={h}: {} voxels, {} of {} solid, stress {:.1}/{:.1}, defl {:.3}, passed {}, {} it, {:.2}s, mesh {} tris vol {:.0} {:?}",
            p.grid.inside.iter().filter(|v| **v).count(),
            out.solid_voxels,
            out.design_voxels,
            out.stress,
            out.allowed,
            out.deflection,
            out.passed,
            out.iterations,
            t.elapsed().as_secs_f64(),
            m.tri.len(),
            mesh_volume(&m),
            out.note
        );
        out
    }

    #[test]
    fn organic_saves_material() {
        let out = run(Style::Organic, std::env::var("H").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0));
        assert!(out.passed);
        assert!(out.solid_voxels < out.design_voxels / 2);
    }

    #[test]
    fn tree_passes() {
        let out = run(Style::Tree, 4.0);
        assert!(out.passed);
        assert!(out.solid_voxels < out.design_voxels);
    }

    #[test]
    fn lattice_surface_is_manifold() {
        let p = problem(2.36);
        let s = settings(Style::Lattice);
        let out = generate(&p, &s, &mut || true).unwrap();
        let m = surface(&p.grid, &out.field, 3, 0, &[], None, &[]);
        // Weld by rounded position, as SolidCraft does.
        let mut ids = std::collections::HashMap::new();
        let key = |p: V3| p.map(|v| (v * 100.0).round() as i64);
        let remap: Vec<usize> = m
            .pos
            .iter()
            .map(|p| {
                let n = ids.len();
                *ids.entry(key(*p)).or_insert(n)
            })
            .collect();
        let mut edges = std::collections::HashMap::new();
        for t in &m.tri {
            let t = t.map(|v| remap[v]);
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_insert(0) += 1;
            }
        }
        let bad = edges.iter().filter(|&(&(a, b), &c)| c != 1 || edges.get(&(b, a)) != Some(&1)).count();
        assert_eq!(bad, 0, "{} of {} edges", bad, edges.len());
    }

    #[test]
    fn lattice_passes() {
        let out = run(Style::Lattice, 4.0);
        assert!(out.passed);
    }

    #[test]
    fn surface_of_box_is_closed() {
        let p = problem(5.0);
        let field: Vec<f64> = p.grid.inside.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect();
        let m = surface(&p.grid, &field, 2, 0, &[], None, &[]);
        // Every edge used exactly twice, in opposite directions.
        let mut edges = std::collections::HashMap::new();
        for t in &m.tri {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_insert(0) += 1;
            }
        }
        let bad: Vec<_> = edges
            .iter()
            .filter(|&(&(a, b), &c)| c != 1 || edges.get(&(b, a)) != Some(&1))
            .map(|(&(a, b), &c)| (m.pos[a], m.pos[b], c))
            .take(6)
            .collect();
        assert!(bad.is_empty(), "{bad:?}");
        assert!(mesh_volume(&m) > 0.0);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    #[test]
    fn bench_matvec() {
        let g = voxelize(&tests_box(), 2.5);
        let brick = Brick::new();
        let elems: Vec<usize> = (0..g.len()).filter(|&v| g.inside[v]).collect();
        let none = vec![false; g.len()];
        let mut fixed = vec![false; g.len()];
        fixed[0] = true;
        let mut load = none.clone();
        load[g.len() - 1] = true;
        let t = std::time::Instant::now();
        let fem = Fem::new(&g, &brick, elems, &fixed, &load, [0.0, 0.0, -1.0], false);
        println!(
            "build {:?} levels {} ndof {:?}",
            t.elapsed(),
            fem.levels().levels.len(),
            fem.levels().levels.iter().map(|l| l.ndof).collect::<Vec<_>>()
        );
        let stiff = vec![1.0; fem.elems.len()];
        let x = vec![1.0; fem.ndof];
        let mut y = vec![0.0; fem.ndof];
        let t = std::time::Instant::now();
        for _ in 0..20 {
            fem.matvec(&stiff, &x, &mut y);
        }
        println!("matvec {:?}", t.elapsed() / 20);
        let t = std::time::Instant::now();
        let p = fem.levels().prepare(&stiff);
        println!("prepare {:?}", t.elapsed());
        let t = std::time::Instant::now();
        for _ in 0..5 {
            let _ = p.vcycle(0, &x);
        }
        println!("vcycle {:?}", t.elapsed() / 5);
    }
    fn tests_box() -> Mesh {
        let (lo, hi) = ([0.0, 0.0, 0.0], [100.0, 40.0, 30.0]);
        let p = |i: usize| [if i & 1 == 1 { hi[0] } else { lo[0] }, if i & 2 == 2 { hi[1] } else { lo[1] }, if i & 4 == 4 { hi[2] } else { lo[2] }];
        Mesh {
            pos: (0..8).map(p).collect(),
            tri: vec![
                [0, 2, 1],
                [1, 2, 3],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [1, 3, 5],
                [3, 7, 5],
            ],
        }
    }
}
