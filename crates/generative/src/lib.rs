//! Generative design: from held faces, a loaded face and a force to a light shape that carries it.
//!
//! - [`voxel`]: a voxel finite-element model with a multigrid solver, topology optimisation
//!   (organic), branching trees, shell-and-lattice infill, and the surface of the result.
//! - [`truss`]: a space truss of round bars between planar faces, sized bar by bar.
//!
//! Pure computation: no document, no kernel. The engine command turns bodies and faces into the
//! inputs here and the results back into bodies.

// Finite-element loops index several arrays by the same element or degree of freedom.
#![allow(clippy::needless_range_loop)]

pub mod truss;
pub mod voxel;

/// What the part is made of, as far as its strength goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Tensile strength, MPa.
    pub strength: f64,
    /// Young's modulus, MPa.
    pub modulus: f64,
    /// Density, g/cm³.
    pub density: f64,
}
