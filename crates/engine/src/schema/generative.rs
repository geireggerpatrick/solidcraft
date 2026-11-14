//! Contracts for the commands in `cmd/generative.rs`. Generative Design replies with the plain
//! feature reply, [`super::features::FeatureResult`].

use schemars::JsonSchema;
use serde::Deserialize;

use super::types::{Point3, StringList};

/// A face, by its body and a point on it.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GenerativeFace {
    /// The body the face belongs to, by name.
    pub body: String,
    /// A point on the face, in mm.
    pub point: Point3,
}

/// Grow a light structure that carries a force from the loaded faces to the held faces, as a new
/// mesh body. The bodies the faces belong to stay as they are: the structure is built in the
/// space between the faces, around those bodies, and ends flat on the picked faces.
///
/// With `space`, the structure is carved out of that body instead and the faces are faces of it.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GenerativeDesignParams {
    /// The faces that are held (screwed or glued to something that does not move). At least one.
    pub held: Vec<GenerativeFace>,
    /// The faces the force acts on. At least one.
    pub load: Vec<GenerativeFace>,
    /// The force in newtons, as a vector `[x, y, z]`; 9.81 N carry one kilogram.
    pub force: Point3,
    /// `organic` (default): material only where the force flows. `tree`: branches from the load
    /// to the held faces. `lattice`: a regular lattice, denser where the stress is higher.
    #[serde(default)]
    pub style: Option<String>,
    /// A body to carve the structure out of. Default: the box around the picked faces.
    #[serde(default)]
    pub space: Option<String>,
    /// Further bodies the structure must stay clear of, by name.
    #[serde(default)]
    pub avoid: Option<StringList>,
    /// How far the design box reaches beyond the picked faces, in mm. Default: 0.
    #[serde(default)]
    pub margin: Option<f64>,
    /// The material's tensile strength in MPa. Default: 35 (PETG, printed).
    #[serde(default)]
    pub strength: Option<f64>,
    /// The material's stiffness (Young's modulus) in MPa. Default: 1900.
    #[serde(default)]
    pub modulus: Option<f64>,
    /// The safety factor, from 1 to 20. Default: 2.
    #[serde(default)]
    pub safety: Option<f64>,
    /// How many cells the design space is divided into, from 500 to 2000000. More cells give a
    /// finer shape and take longer. Default: 40000.
    #[serde(default)]
    pub voxels: Option<f64>,
    /// The thinnest wall or strut to make, in mm, from 0.2 to 100. Default: 4 % of the design
    /// space's longest side.
    #[serde(default)]
    pub wall: Option<f64>,
    /// The feature's name in the timeline.
    #[serde(default)]
    pub name: Option<String>,
    /// The new body's name. Default: `Generative`.
    #[serde(default)]
    pub body_name: Option<String>,
}
