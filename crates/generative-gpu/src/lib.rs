//! Generative design on the GPU: the solver the generative crate hands its linear problems to
//! ([`solidcraft_generative::voxel::GridSolver`]). Conjugate gradients preconditioned by a
//! geometric multigrid V-cycle, all of it in compute shaders on the whole voxel grid, in single
//! precision; the generative crate keeps the residual in double precision and asks only for
//! corrections.
//!
//! [`install`] registers the solver; the first design opens a device on the best hardware
//! adapter. Without one (and on the web) designs compute on the CPU as before.

#[cfg(not(target_arch = "wasm32"))]
mod solver;

#[cfg(not(target_arch = "wasm32"))]
pub use solver::GpuSolver;

/// Registers the GPU solver for generative designs: the first design that computes opens a
/// device on this machine's best hardware adapter, and without one computes on the CPU.
#[cfg(not(target_arch = "wasm32"))]
pub fn install() {
    solidcraft_generative::voxel::set_grid_solver_source(|| {
        let solver = GpuSolver::open(solidcraft_viewport_gpu::select::Policy::Hardware).ok()?;
        Some(Box::new(solver))
    });
}

/// The web build computes on the CPU.
#[cfg(target_arch = "wasm32")]
pub fn install() {}
