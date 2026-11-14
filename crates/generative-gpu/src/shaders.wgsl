// The voxel grid's linear algebra. Every entry point works on one multigrid level, whose nodes
// are numbered i + (nx + 1) * (j + (ny + 1) * k) and whose cells i + nx * (j + ny * k).
// A vector holds one vec4 per node (xyz used). `info` holds per node: bit 0, the node belongs to
// a cell of the model; bit 1, the node is held.

struct Level {
    // Cells along x, y, z of this level, and its node count.
    cells: vec4<u32>,
    // Cells along x, y, z of the next finer level (restrict, prolong, coarsen), and its node count.
    fine: vec4<u32>,
}

// One dispatch row holds this many invocations; longer arrays wrap into further rows.
const ROW: u32 = 262144u;

const CORNER = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u), vec3<u32>(1u, 1u, 0u), vec3<u32>(0u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u), vec3<u32>(1u, 1u, 1u), vec3<u32>(0u, 1u, 1u),
);

fn node_at(cells: vec3<u32>, p: vec3<u32>) -> u32 {
    return p.x + (cells.x + 1u) * (p.y + (cells.y + 1u) * p.z);
}

fn node_pos(cells: vec3<u32>, idx: u32) -> vec3<u32> {
    let nx = cells.x + 1u;
    let ny = cells.y + 1u;
    return vec3<u32>(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}

fn cell_at(cells: vec3<u32>, e: vec3<u32>) -> u32 {
    return e.x + cells.x * (e.y + cells.y * e.z);
}

// ---- y = K x ----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> mv_level: Level;
@group(0) @binding(1) var<storage, read> mv_ke: array<f32>;
@group(0) @binding(2) var<storage, read> mv_stiff: array<f32>;
@group(0) @binding(3) var<storage, read> mv_info: array<u32>;
@group(0) @binding(4) var<storage, read> mv_x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> mv_y: array<vec4<f32>>;

@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= mv_level.cells.w {
        return;
    }
    let info = mv_info[idx];
    if (info & 2u) != 0u {
        mv_y[idx] = mv_x[idx];
        return;
    }
    if (info & 1u) == 0u {
        mv_y[idx] = vec4<f32>(0.0);
        return;
    }
    let cells = mv_level.cells.xyz;
    let p = node_pos(cells, idx);
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let c = CORNER[a];
        if p.x < c.x || p.y < c.y || p.z < c.z {
            continue;
        }
        let e = p - c;
        if e.x >= cells.x || e.y >= cells.y || e.z >= cells.z {
            continue;
        }
        let s = mv_stiff[cell_at(cells, e)];
        if s == 0.0 {
            continue;
        }
        var sum = vec3<f32>(0.0);
        for (var b = 0u; b < 8u; b++) {
            let u = mv_x[node_at(cells, e + CORNER[b])].xyz;
            let at = (3u * a) * 24u + 3u * b;
            sum.x += mv_ke[at] * u.x + mv_ke[at + 1u] * u.y + mv_ke[at + 2u] * u.z;
            sum.y += mv_ke[at + 24u] * u.x + mv_ke[at + 25u] * u.y + mv_ke[at + 26u] * u.z;
            sum.z += mv_ke[at + 48u] * u.x + mv_ke[at + 49u] * u.y + mv_ke[at + 50u] * u.z;
        }
        acc += s * sum;
    }
    mv_y[idx] = vec4<f32>(acc, 0.0);
}

// ---- 1 / diagonal of K ------------------------------------------------------------------

@group(0) @binding(0) var<uniform> dg_level: Level;
@group(0) @binding(1) var<storage, read> dg_ke: array<f32>;
@group(0) @binding(2) var<storage, read> dg_stiff: array<f32>;
@group(0) @binding(3) var<storage, read> dg_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> dg_inv: array<vec4<f32>>;

@compute @workgroup_size(64)
fn diag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= dg_level.cells.w {
        return;
    }
    let info = dg_info[idx];
    if info != 1u {
        dg_inv[idx] = vec4<f32>(1.0);
        return;
    }
    let cells = dg_level.cells.xyz;
    let p = node_pos(cells, idx);
    var acc = vec3<f32>(0.0);
    for (var a = 0u; a < 8u; a++) {
        let c = CORNER[a];
        if p.x < c.x || p.y < c.y || p.z < c.z {
            continue;
        }
        let e = p - c;
        if e.x >= cells.x || e.y >= cells.y || e.z >= cells.z {
            continue;
        }
        let s = dg_stiff[cell_at(cells, e)];
        let at = 3u * a;
        acc += s * vec3<f32>(dg_ke[at * 25u], dg_ke[(at + 1u) * 25u], dg_ke[(at + 2u) * 25u]);
    }
    var inv = vec3<f32>(1.0);
    if acc.x > 0.0 { inv.x = 1.0 / acc.x; }
    if acc.y > 0.0 { inv.y = 1.0 / acc.y; }
    if acc.z > 0.0 { inv.z = 1.0 / acc.z; }
    dg_inv[idx] = vec4<f32>(inv, 0.0);
}

// ---- z += 0.6 D^-1 (r - K z) --------------------------------------------------------------

@group(0) @binding(0) var<uniform> jc_level: Level;
@group(0) @binding(1) var<storage, read> jc_info: array<u32>;
@group(0) @binding(2) var<storage, read> jc_inv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> jc_r: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> jc_az: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> jc_z: array<vec4<f32>>;

fn jacobi_at(idx: u32, damping: f32) {
    if idx >= jc_level.cells.w {
        return;
    }
    if jc_info[idx] != 1u {
        jc_z[idx] = vec4<f32>(0.0);
        return;
    }
    jc_z[idx] = jc_z[idx] + damping * jc_inv[idx] * (jc_r[idx] - jc_az[idx]);
}

@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    jacobi_at(gid.x + gid.y * ROW, 0.6);
}

// The same with less damping: safe to repeat many times, as the coarsest level's solve.
@compute @workgroup_size(64)
fn jacobi_gentle(@builtin(global_invocation_id) gid: vec3<u32>) {
    jacobi_at(gid.x + gid.y * ROW, 0.3);
}

// ---- coarse residual: P^T (r - K z) -------------------------------------------------------

@group(0) @binding(0) var<uniform> rs_level: Level;
@group(0) @binding(1) var<storage, read> rs_info: array<u32>;
@group(0) @binding(2) var<storage, read> rs_fine_info: array<u32>;
@group(0) @binding(3) var<storage, read> rs_weight: array<f32>;
@group(0) @binding(4) var<storage, read> rs_r: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> rs_az: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> rs_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn coarse_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= rs_level.cells.w {
        return;
    }
    if rs_info[idx] != 1u {
        rs_out[idx] = vec4<f32>(0.0);
        return;
    }
    let fine = rs_level.fine.xyz;
    let c = vec3<i32>(node_pos(rs_level.cells.xyz, idx)) * 2;
    var acc = vec4<f32>(0.0);
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let f = c + vec3<i32>(dx, dy, dz);
                if f.x < 0 || f.y < 0 || f.z < 0 || f.x > i32(fine.x) || f.y > i32(fine.y) || f.z > i32(fine.z) {
                    continue;
                }
                let at = node_at(fine, vec3<u32>(f));
                if rs_fine_info[at] != 1u {
                    continue;
                }
                var w = rs_weight[at];
                if dx != 0 { w *= 0.5; }
                if dy != 0 { w *= 0.5; }
                if dz != 0 { w *= 0.5; }
                acc += w * (rs_r[at] - rs_az[at]);
            }
        }
    }
    rs_out[idx] = acc;
}

// ---- z_fine += P z_coarse -----------------------------------------------------------------

@group(0) @binding(0) var<uniform> pr_level: Level;
@group(0) @binding(1) var<storage, read> pr_info: array<u32>;
@group(0) @binding(2) var<storage, read> pr_fine_info: array<u32>;
@group(0) @binding(3) var<storage, read> pr_weight: array<f32>;
@group(0) @binding(4) var<storage, read> pr_zc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> pr_z: array<vec4<f32>>;

@compute @workgroup_size(64)
fn prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= pr_level.fine.w {
        return;
    }
    if pr_fine_info[idx] != 1u {
        return;
    }
    let cells = pr_level.cells.xyz;
    let p = node_pos(pr_level.fine.xyz, idx);
    let base = p / 2u;
    let odd = p % 2u;
    var acc = vec4<f32>(0.0);
    for (var dz = 0u; dz <= odd.z; dz++) {
        for (var dy = 0u; dy <= odd.y; dy++) {
            for (var dx = 0u; dx <= odd.x; dx++) {
                let at = node_at(cells, base + vec3<u32>(dx, dy, dz));
                if (pr_info[at] & 1u) == 0u {
                    continue;
                }
                var w = 1.0;
                if odd.x == 1u { w *= 0.5; }
                if odd.y == 1u { w *= 0.5; }
                if odd.z == 1u { w *= 0.5; }
                acc += w * pr_zc[at];
            }
        }
    }
    pr_z[idx] = pr_z[idx] + pr_weight[idx] * acc;
}

// ---- a coarse cell's stiffness from its eight children ------------------------------------

@group(0) @binding(0) var<uniform> cs_level: Level;
@group(0) @binding(1) var<storage, read> cs_fine: array<f32>;
@group(0) @binding(2) var<storage, read_write> cs_out: array<f32>;

@compute @workgroup_size(64)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    let cells = cs_level.cells.xyz;
    if idx >= cells.x * cells.y * cells.z {
        return;
    }
    let fine = cs_level.fine.xyz;
    let e = vec3<u32>(idx % cells.x, (idx / cells.x) % cells.y, idx / (cells.x * cells.y)) * 2u;
    var acc = 0.0;
    for (var a = 0u; a < 8u; a++) {
        let f = e + CORNER[a];
        if f.x < fine.x && f.y < fine.y && f.z < fine.z {
            acc += cs_fine[cell_at(fine, f)];
        }
    }
    cs_out[idx] = acc / 4.0;
}

// ---- y = (ca * S[ia]) x + (cb * S[ib]) y ---------------------------------------------------

struct Mix {
    ia: u32,
    ib: u32,
    ca: f32,
    cb: f32,
}

@group(0) @binding(0) var<uniform> ax_level: Level;
@group(0) @binding(1) var<uniform> ax_mix: Mix;
@group(0) @binding(2) var<storage, read> ax_s: array<f32>;
@group(0) @binding(3) var<storage, read> ax_x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> ax_y: array<vec4<f32>>;

@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= ax_level.cells.w {
        return;
    }
    ax_y[idx] = (ax_mix.ca * ax_s[ax_mix.ia]) * ax_x[idx] + (ax_mix.cb * ax_s[ax_mix.ib]) * ax_y[idx];
}

// ---- out[i] = x[i] . y[i] -----------------------------------------------------------------

@group(0) @binding(0) var<uniform> pd_level: Level;
@group(0) @binding(1) var<storage, read> pd_x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pd_y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pd_out: array<f32>;

@compute @workgroup_size(64)
fn product(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    if idx >= pd_level.cells.w {
        return;
    }
    pd_out[idx] = dot(pd_x[idx].xyz, pd_y[idx].xyz);
}

// ---- out[g] = sum of in[64 g .. 64 g + 64] -------------------------------------------------

struct Count {
    // How many values `in` holds.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> rd_count: Count;
@group(0) @binding(1) var<storage, read> rd_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> rd_out: array<f32>;

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ROW;
    let first = idx * 64u;
    if first >= rd_count.count {
        return;
    }
    let last = min(first + 64u, rd_count.count);
    var acc = 0.0;
    for (var i = first; i < last; i++) {
        acc += rd_in[i];
    }
    rd_out[idx] = acc;
}

// ---- the scalars of the conjugate gradients -------------------------------------------------

struct Scalar {
    dst: u32,
    a: u32,
    b: u32,
    c: u32,
    // 0: S[a] / S[b]; 1: max(0, (S[a] - S[b]) / S[c]); 2: S[a].
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> sc_op: Scalar;
@group(0) @binding(1) var<storage, read_write> sc_s: array<f32>;

fn ratio(a: f32, b: f32) -> f32 {
    if abs(b) < 1e-30 {
        return 0.0;
    }
    return a / b;
}

@compute @workgroup_size(1)
fn scalars() {
    var v = sc_s[sc_op.a];
    if sc_op.kind == 0u {
        v = ratio(sc_s[sc_op.a], sc_s[sc_op.b]);
    } else if sc_op.kind == 1u {
        v = max(0.0, ratio(sc_s[sc_op.a] - sc_s[sc_op.b], sc_s[sc_op.c]));
    }
    sc_s[sc_op.dst] = v;
}
