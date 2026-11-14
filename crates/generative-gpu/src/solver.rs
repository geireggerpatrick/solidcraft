//! The solver: its plan of buffers and bind groups for one grid size, and the recorded passes.

use solidcraft_generative::voxel::{GridJob, GridSolver};
use wgpu::util::DeviceExt;

const SHADERS: &str = include_str!("shaders.wgsl");
/// Invocations per workgroup, and per dispatch row (`ROW` in the shaders).
const GROUP: u32 = 64;
const ROW_GROUPS: u32 = 4096;
/// Jacobi sweeps that stand in for a solve on the coarsest level.
const COARSEST_SWEEPS: usize = 30;
/// Iterations recorded between two looks at the residual.
const BLOCK: usize = 4;

// The scalars of the conjugate gradients, as slots of one small buffer.
const ONE: u32 = 0;
const RZ: u32 = 1;
const PQ: u32 = 2;
const ALPHA: u32 = 3;
const RZ_NEW: u32 = 4;
const RZ_MIXED: u32 = 5;
const BETA: u32 = 6;
const RR: u32 = 7;
const SCALARS: usize = 8;

pub struct GpuSolver {
    name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Pipes,
    plan: Option<Plan>,
    /// What the device reported outside a call's own result (a validation error, memory).
    fault: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

struct Pipes {
    matvec: wgpu::ComputePipeline,
    diag: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    jacobi_gentle: wgpu::ComputePipeline,
    restrict: wgpu::ComputePipeline,
    prolong: wgpu::ComputePipeline,
    coarsen: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    product: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    scalar: wgpu::ComputePipeline,
}

/// A recorded pass: the pipeline's bind group and how many invocations it covers.
struct Step {
    bind: wgpu::BindGroup,
    count: u32,
}

struct Level {
    cells: [usize; 3],
    nodes: usize,
    stiff: wgpu::Buffer,
    info: wgpu::Buffer,
    /// Per node: one over the sum of its weights to the coarser level's nodes.
    weight: wgpu::Buffer,
    z: wgpu::Buffer,
    az: wgpu::Buffer,
    /// `az = K z`.
    apply: Step,
    diag: Step,
    smooth: Step,
    /// From the finer level to this one; `None` on the finest.
    coarsen: Option<Step>,
    restrict: Option<Step>,
    prolong: Option<Step>,
    /// The right-hand side; `None` on the finest level, where it is the conjugate gradients' `r`.
    rhs: Option<wgpu::Buffer>,
}

/// A dot product: the products per node, the sums of 64 at a time down to one value, and the
/// buffer that value ends in.
struct Dot {
    product: Step,
}

/// Everything for one grid size.
struct Plan {
    n: [usize; 3],
    levels: Vec<Level>,
    scalars: wgpu::Buffer,
    scalars_read: wgpu::Buffer,
    u: wgpu::Buffer,
    u_read: wgpu::Buffer,
    r: wgpu::Buffer,
    r_old: wgpu::Buffer,
    p: wgpu::Buffer,
    /// `q = K p`.
    apply_p: Step,
    add_p_to_u: Step,
    take_q_from_r: Step,
    next_p: Step,
    dot_rz: Dot,
    dot_pq: Dot,
    dot_old_z: Dot,
    dot_rr: Dot,
    /// The sums after a product, and the buffer whose first value is the dot product.
    reduce: Vec<Step>,
    sum: wgpu::Buffer,
    alpha: Step,
    beta: Step,
    keep_rz: Step,
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn nodes_of(cells: [usize; 3]) -> usize {
    (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1)
}

/// The level's uniform block: its cells and node count, and the finer level's.
fn level_block(cells: [usize; 3], fine: Option<[usize; 3]>) -> Vec<u8> {
    let f = fine.unwrap_or([0; 3]);
    bytes_u32(&[
        cells[0] as u32,
        cells[1] as u32,
        cells[2] as u32,
        nodes_of(cells) as u32,
        f[0] as u32,
        f[1] as u32,
        f[2] as u32,
        if fine.is_some() { nodes_of(f) as u32 } else { 0 },
    ])
}

impl GpuSolver {
    /// Opens a device of its own on the adapter `policy` picks.
    pub fn open(policy: solidcraft_viewport_gpu::select::Policy) -> Result<Self, String> {
        let (adapter, _) = solidcraft_viewport_gpu::select::pick(policy).ok_or("no GPU adapter")?;
        let info = adapter.get_info();
        // The adapter's own limits: a fine grid's vectors are larger than the default buffer sizes.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("generative design"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        // An error on the device is kept and ends the solve it happened in; wgpu's own handler
        // would panic.
        let fault = std::sync::Arc::new(std::sync::Mutex::new(None));
        let seen = fault.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = seen.lock() {
                slot.get_or_insert_with(|| e.to_string());
            }
        }));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("generative design"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let pipe = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipes {
            matvec: pipe("matvec"),
            diag: pipe("diag"),
            jacobi: pipe("jacobi"),
            jacobi_gentle: pipe("jacobi_gentle"),
            restrict: pipe("coarse_residual"),
            prolong: pipe("prolong"),
            coarsen: pipe("coarsen"),
            axpy: pipe("axpy"),
            product: pipe("product"),
            reduce: pipe("reduce"),
            scalar: pipe("scalars"),
        };
        if let Some(e) = fault.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(e);
        }
        Ok(GpuSolver { name: format!("{} ({:?})", info.name, info.backend), device, queue, pipes, plan: None, fault })
    }

    fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::UNIFORM })
    }

    fn readable(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn step(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer], count: usize) -> Step {
        let entries: Vec<wgpu::BindGroupEntry> =
            buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries });
        Step { bind, count: count as u32 }
    }

    /// The buffers and bind groups for a grid of `n` cells.
    fn plan(&self, n: [usize; 3], ke: &[[f64; 24]; 24]) -> Result<Plan, String> {
        let limits = self.device.limits();
        let vec_bytes = |nodes: usize| 16 * nodes as u64;
        let finest = nodes_of(n);
        if vec_bytes(finest) > limits.max_storage_buffer_binding_size
            || vec_bytes(finest) > limits.max_buffer_size
            || finest > (u32::MAX / 4) as usize
        {
            return Err("the grid is too large for this GPU".into());
        }
        let ke = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ke"),
            contents: &bytes_f32(&ke.iter().flatten().map(|v| *v as f32).collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let p = &self.pipes;
        let r = self.storage("r", vec_bytes(finest));
        let z0 = self.storage("z", vec_bytes(finest));
        // The grids, each half the one before.
        let mut sizes = vec![n];
        while let Some(&c) = sizes.last() {
            if sizes.len() >= 10 || c.iter().any(|&v| v < 2) || nodes_of(c) <= 64 {
                break;
            }
            sizes.push(c.map(|v| v.div_ceil(2)));
        }
        let mut levels: Vec<Level> = Vec::new();
        for (l, &cells) in sizes.iter().enumerate() {
            let nodes = nodes_of(cells);
            let count = cells[0] * cells[1] * cells[2];
            let fine = l.checked_sub(1).and_then(|f| sizes.get(f).copied());
            let block = self.uniform("level", &level_block(cells, fine));
            let stiff = self.storage("stiff", 4 * count as u64);
            let info = self.storage("info", 4 * nodes as u64);
            let weight = self.storage("weight", 4 * nodes as u64);
            let inv = self.storage("inv", vec_bytes(nodes));
            let az = self.storage("az", vec_bytes(nodes));
            // The finest level's right-hand side and result are the conjugate gradients' r and z.
            let (rhs, z) =
                if l == 0 { (r.clone(), z0.clone()) } else { (self.storage("rhs", vec_bytes(nodes)), self.storage("z", vec_bytes(nodes))) };
            let apply = self.step(&p.matvec, &[&block, &ke, &stiff, &info, &z, &az], nodes);
            let diag = self.step(&p.diag, &[&block, &ke, &stiff, &info, &inv], nodes);
            // The coarsest level smooths many times over, as its solve: with the gentle pipeline.
            let smoother = if l + 1 == sizes.len() { &p.jacobi_gentle } else { &p.jacobi };
            let smooth = self.step(smoother, &[&block, &info, &inv, &rhs, &az, &z], nodes);
            let (coarsen, restrict, prolong) = match levels.last() {
                Some(f) => {
                    let fine_rhs = f.rhs_or(&r);
                    (
                        Some(self.step(&p.coarsen, &[&block, &f.stiff, &stiff], count)),
                        Some(self.step(&p.restrict, &[&block, &info, &f.info, &f.weight, fine_rhs, &f.az, &rhs], nodes)),
                        Some(self.step(&p.prolong, &[&block, &info, &f.info, &f.weight, &z, &f.z], f.nodes)),
                    )
                }
                None => (None, None, None),
            };
            levels.push(Level {
                cells,
                nodes,
                stiff,
                info,
                weight,
                z,
                az,
                apply,
                diag,
                smooth,
                coarsen,
                restrict,
                prolong,
                rhs: (l > 0).then_some(rhs),
            });
        }
        let first = levels.first().ok_or("no grid")?;
        let block = self.uniform("level", &level_block(n, None));
        let scalars = self.storage("scalars", 4 * SCALARS as u64);
        let u = self.storage("u", vec_bytes(finest));
        let r_old = self.storage("r_old", vec_bytes(finest));
        let pv = self.storage("p", vec_bytes(finest));
        let q = self.storage("q", vec_bytes(finest));
        let apply_p = self.step(&p.matvec, &[&block, &ke, &first.stiff, &first.info, &pv, &q], finest);
        let mix = |ia: u32, ib: u32, ca: f32, cb: f32| {
            let mut b = bytes_u32(&[ia, ib]);
            b.extend(bytes_f32(&[ca, cb]));
            self.uniform("mix", &b)
        };
        let add_p_to_u = self.step(&p.axpy, &[&block, &mix(ALPHA, ONE, 1.0, 1.0), &scalars, &pv, &u], finest);
        let take_q_from_r = self.step(&p.axpy, &[&block, &mix(ALPHA, ONE, -1.0, 1.0), &scalars, &q, &r], finest);
        let next_p = self.step(&p.axpy, &[&block, &mix(ONE, BETA, 1.0, 1.0), &scalars, &z0, &pv], finest);
        // The sums: the products in `a`, then 64 at a time back and forth between `a` and `b`.
        let a = self.storage("products", 4 * finest as u64);
        let b = self.storage("sums", 4 * finest.div_ceil(64) as u64);
        let dot = |x: &wgpu::Buffer, y: &wgpu::Buffer| Dot { product: self.step(&p.product, &[&block, x, y, &a], finest) };
        let mut reduce = Vec::new();
        let (mut len, mut from_a) = (finest, true);
        while len > 1 {
            let count = self.uniform("count", &bytes_u32(&[len as u32, 0, 0, 0]));
            let (src, dst) = if from_a { (&a, &b) } else { (&b, &a) };
            len = len.div_ceil(64);
            reduce.push(self.step(&p.reduce, &[&count, src, dst], len));
            from_a = !from_a;
        }
        let sum = if from_a { a.clone() } else { b.clone() };
        let op = |dst: u32, x: u32, y: u32, z: u32, kind: u32| {
            let block = self.uniform("scalar", &bytes_u32(&[dst, x, y, z, kind, 0, 0, 0]));
            self.step(&p.scalar, &[&block, &scalars], 1)
        };
        Ok(Plan {
            n,
            scalars_read: self.readable("scalars read", 4 * SCALARS as u64),
            u_read: self.readable("u read", vec_bytes(finest)),
            dot_rz: dot(&r, &z0),
            dot_pq: dot(&pv, &q),
            dot_old_z: dot(&r_old, &z0),
            dot_rr: dot(&r, &r),
            alpha: op(ALPHA, RZ, PQ, 0, 0),
            beta: op(BETA, RZ_NEW, RZ_MIXED, RZ, 1),
            keep_rz: op(RZ, RZ_NEW, 0, 0, 2),
            levels,
            scalars,
            u,
            r,
            r_old,
            p: pv,
            apply_p,
            add_p_to_u,
            take_q_from_r,
            next_p,
            reduce,
            sum,
        })
    }
}

impl Level {
    /// This level's right-hand side, which on the finest level is the conjugate gradients' `r`.
    fn rhs_or<'a>(&'a self, r: &'a wgpu::Buffer) -> &'a wgpu::Buffer {
        self.rhs.as_ref().unwrap_or(r)
    }
}

/// What the levels hold for one job, worked out here: which nodes belong to the model, which
/// are held, and the weights between the levels.
struct Masks {
    info: Vec<u32>,
    weight: Vec<f32>,
}

fn masks(sizes: &[[usize; 3]], stiff: &[f32], fixed: &[bool]) -> Vec<Masks> {
    let mut out: Vec<Masks> = Vec::new();
    let mut inside: Vec<bool> = stiff.iter().map(|s| *s != 0.0).collect();
    let mut held: Vec<bool> = fixed.to_vec();
    for (l, &c) in sizes.iter().enumerate() {
        let node = |i: usize, j: usize, k: usize| i + (c[0] + 1) * (j + (c[1] + 1) * k);
        let mut info = vec![0u32; nodes_of(c)];
        for k in 0..c[2] {
            for j in 0..c[1] {
                for i in 0..c[0] {
                    if inside[i + c[0] * (j + c[1] * k)] {
                        for d in 0..8 {
                            info[node(i + (d & 1), j + ((d >> 1) & 1), k + (d >> 2))] |= 1;
                        }
                    }
                }
            }
        }
        for (v, h) in info.iter_mut().zip(&held) {
            if *h && *v != 0 {
                *v |= 2;
            }
        }
        let mut weight = vec![0.0f32; info.len()];
        if let Some(&cc) = sizes.get(l + 1) {
            // The coarser level: a cell is there when one of its children is, and a node is held
            // when a held node of this level leans on it.
            let mut coarse_inside = vec![false; cc[0] * cc[1] * cc[2]];
            for k in 0..c[2] {
                for j in 0..c[1] {
                    for i in 0..c[0] {
                        if inside[i + c[0] * (j + c[1] * k)] {
                            coarse_inside[i / 2 + cc[0] * (j / 2 + cc[1] * (k / 2))] = true;
                        }
                    }
                }
            }
            let cnode = |i: usize, j: usize, k: usize| i + (cc[0] + 1) * (j + (cc[1] + 1) * k);
            let mut coarse_active = vec![false; nodes_of(cc)];
            for k in 0..cc[2] {
                for j in 0..cc[1] {
                    for i in 0..cc[0] {
                        if coarse_inside[i + cc[0] * (j + cc[1] * k)] {
                            for d in 0..8 {
                                coarse_active[cnode(i + (d & 1), j + ((d >> 1) & 1), k + (d >> 2))] = true;
                            }
                        }
                    }
                }
            }
            let mut coarse_held = vec![false; nodes_of(cc)];
            for k in 0..=c[2] {
                for j in 0..=c[1] {
                    for i in 0..=c[0] {
                        let f = node(i, j, k);
                        if info[f] & 1 == 0 {
                            continue;
                        }
                        let mut total = 0.0f32;
                        for dk in 0..=(k & 1) {
                            for dj in 0..=(j & 1) {
                                for di in 0..=(i & 1) {
                                    let at = cnode(i / 2 + di, j / 2 + dj, k / 2 + dk);
                                    if coarse_active[at] {
                                        total += [1.0, 0.5][i & 1] * [1.0, 0.5][j & 1] * [1.0, 0.5][k & 1];
                                        if info[f] & 2 != 0 {
                                            coarse_held[at] = true;
                                        }
                                    }
                                }
                            }
                        }
                        if total > 0.0 {
                            weight[f] = 1.0 / total;
                        }
                    }
                }
            }
            inside = coarse_inside;
            held = coarse_held;
        }
        out.push(Masks { info, weight });
    }
    out
}

impl Plan {
    fn run(&self, enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, step: &Step) {
        let groups = step.count.div_ceil(GROUP).max(1);
        let x = groups.min(ROW_GROUPS);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &step.bind, &[]);
        pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
    }

    /// `z = M r` on level `l`, where `M` is one V-cycle.
    fn vcycle(&self, enc: &mut wgpu::CommandEncoder, p: &Pipes, l: usize) {
        let Some(level) = self.levels.get(l) else { return };
        enc.clear_buffer(&level.z, 0, None);
        enc.clear_buffer(&level.az, 0, None);
        let Some(coarse) = self.levels.get(l + 1) else {
            self.run(enc, &p.jacobi_gentle, &level.smooth);
            for _ in 1..COARSEST_SWEEPS {
                self.run(enc, &p.matvec, &level.apply);
                self.run(enc, &p.jacobi_gentle, &level.smooth);
            }
            return;
        };
        self.run(enc, &p.jacobi, &level.smooth);
        self.run(enc, &p.matvec, &level.apply);
        if let Some(step) = &coarse.restrict {
            self.run(enc, &p.restrict, step);
        }
        self.vcycle(enc, p, l + 1);
        if let Some(step) = &coarse.prolong {
            self.run(enc, &p.prolong, step);
        }
        self.run(enc, &p.matvec, &level.apply);
        self.run(enc, &p.jacobi, &level.smooth);
    }

    /// The dot product into the scalar `slot`.
    fn dot(&self, enc: &mut wgpu::CommandEncoder, p: &Pipes, dot: &Dot, slot: u32) {
        self.run(enc, &p.product, &dot.product);
        for step in &self.reduce {
            self.run(enc, &p.reduce, step);
        }
        enc.copy_buffer_to_buffer(&self.sum, 0, &self.scalars, 4 * slot as u64, 4);
    }
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<f32>, String> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let bytes = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let values = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    drop(bytes);
    buffer.unmap();
    Ok(values)
}

impl GpuSolver {
    /// The error the device reported since the last look, if any.
    fn fault(&self) -> Result<(), String> {
        match self.fault.lock().map(|mut slot| slot.take()) {
            Ok(None) => Ok(()),
            Ok(Some(e)) => Err(e),
            Err(_) => Err("the GPU's error state is lost".into()),
        }
    }
}

impl GridSolver for GpuSolver {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn solve(&mut self, job: &GridJob) -> Result<(Vec<f32>, usize), String> {
        let n = job.n;
        let nodes = nodes_of(n);
        if job.stiff.len() != n[0] * n[1] * n[2] || job.fixed.len() != nodes || job.rhs.len() != 3 * nodes || n.contains(&0) {
            return Err("the job's arrays do not fit its grid".into());
        }
        if self.plan.as_ref().is_none_or(|p| p.n != n) {
            self.plan = None;
            self.plan = Some(self.plan(n, job.ke)?);
            self.fault()?;
        }
        let plan = self.plan.as_ref().ok_or("no plan")?;
        let (p, queue) = (&self.pipes, &self.queue);
        // This job's model: stiffness, masks, right-hand side.
        let sizes: Vec<[usize; 3]> = plan.levels.iter().map(|l| l.cells).collect();
        let first = plan.levels.first().ok_or("no grid")?;
        queue.write_buffer(&first.stiff, 0, &bytes_f32(job.stiff));
        for (level, m) in plan.levels.iter().zip(masks(&sizes, job.stiff, job.fixed)) {
            queue.write_buffer(&level.info, 0, &bytes_u32(&m.info));
            queue.write_buffer(&level.weight, 0, &bytes_f32(&m.weight));
        }
        let mut rhs = vec![0.0f32; 4 * nodes];
        let mut norm = 0.0f64;
        for (node, held) in job.fixed.iter().enumerate() {
            if !held {
                for x in 0..3 {
                    let v = job.rhs[3 * node + x];
                    rhs[4 * node + x] = v;
                    norm += f64::from(v) * f64::from(v);
                }
            }
        }
        let norm = norm.sqrt();
        if !(norm > 0.0) {
            return Ok((vec![0.0; 3 * nodes], 0));
        }
        queue.write_buffer(&plan.r, 0, &bytes_f32(&rhs));
        let mut scalars = [0.0f32; SCALARS];
        scalars[ONE as usize] = 1.0;
        queue.write_buffer(&plan.scalars, 0, &bytes_f32(&scalars));
        let encoder = || self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // The levels' stiffness and diagonals, then the first direction: p = z = M r.
        let mut enc = encoder();
        enc.clear_buffer(&plan.u, 0, None);
        for level in &plan.levels {
            if let Some(step) = &level.coarsen {
                plan.run(&mut enc, &p.coarsen, step);
            }
            plan.run(&mut enc, &p.diag, &level.diag);
        }
        plan.vcycle(&mut enc, p, 0);
        enc.copy_buffer_to_buffer(&first.z, 0, &plan.p, 0, None);
        plan.dot(&mut enc, p, &plan.dot_rz, RZ);
        queue.submit([enc.finish()]);
        let mut iterations = 0;
        while iterations < job.max_iter {
            let mut enc = encoder();
            for _ in 0..BLOCK {
                plan.run(&mut enc, &p.matvec, &plan.apply_p);
                plan.dot(&mut enc, p, &plan.dot_pq, PQ);
                plan.run(&mut enc, &p.scalar, &plan.alpha);
                enc.copy_buffer_to_buffer(&plan.r, 0, &plan.r_old, 0, None);
                plan.run(&mut enc, &p.axpy, &plan.add_p_to_u);
                plan.run(&mut enc, &p.axpy, &plan.take_q_from_r);
                plan.vcycle(&mut enc, p, 0);
                // Flexible (Polak-Ribiere) beta: the V-cycle is not exactly the same each time.
                plan.dot(&mut enc, p, &plan.dot_rz, RZ_NEW);
                plan.dot(&mut enc, p, &plan.dot_old_z, RZ_MIXED);
                plan.run(&mut enc, &p.scalar, &plan.beta);
                plan.run(&mut enc, &p.scalar, &plan.keep_rz);
                plan.run(&mut enc, &p.axpy, &plan.next_p);
            }
            plan.dot(&mut enc, p, &plan.dot_rr, RR);
            enc.copy_buffer_to_buffer(&plan.scalars, 0, &plan.scalars_read, 0, None);
            queue.submit([enc.finish()]);
            iterations += BLOCK;
            let s = read(&self.device, &plan.scalars_read)?;
            self.fault()?;
            let rr = f64::from(s.get(RR as usize).copied().unwrap_or(f32::NAN));
            if !rr.is_finite() {
                return Err("the GPU solve did not stay finite".into());
            }
            if rr.sqrt() / norm < job.tol {
                break;
            }
        }
        let mut enc = encoder();
        enc.copy_buffer_to_buffer(&plan.u, 0, &plan.u_read, 0, None);
        queue.submit([enc.finish()]);
        let u = read(&self.device, &plan.u_read)?;
        self.fault()?;
        let d: Vec<f32> = u.as_chunks::<4>().0.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
        Ok((d, iterations))
    }
}

#[cfg(test)]
#[path = "solver/tests.rs"]
mod tests;
