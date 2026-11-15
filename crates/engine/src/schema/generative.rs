//! Contracts for the commands in `cmd/generative.rs`. Generative Design replies with the plain
//! feature reply, [`super::features::FeatureResult`].

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::features::FeatureResult;
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

/// A further load: faces, what acts on them (one of `force`, `moment` and `pressure`) and the
/// load case it belongs to.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GenerativeLoad {
    /// The faces the load acts on. At least one.
    pub faces: Vec<GenerativeFace>,
    /// A force in newtons, as a vector `[x, y, z]`, shared evenly by the faces.
    #[serde(default)]
    pub force: Option<Point3>,
    /// A moment in newton metres, as a vector `[x, y, z]`: it turns the faces about that
    /// vector through their middle, the way a right-handed screw along it turns.
    #[serde(default)]
    pub moment: Option<Point3>,
    /// A pressure in MPa that presses on the faces, against their outward normals; a negative
    /// one pulls.
    #[serde(default)]
    pub pressure: Option<f64>,
    /// The load case, from 1 to 8. Loads of case 1 act at the same time as `load`; the loads of
    /// each other case act together on their own. Default: 1.
    #[serde(default)]
    pub case: Option<u32>,
}

/// Grow a light structure that carries a force from the loaded faces to the held faces, as a new
/// mesh body. The bodies the faces belong to stay as they are: the structure is built in the
/// space between the faces, around those bodies, and ends flat on the picked faces.
///
/// With `space`, the structure is carved out of that body instead and the faces are faces of it.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GenerativeDesignParams {
    /// The faces that are held (screwed or glued to something that does not move). At least
    /// one of `held`, `slide` and `pin` has a face.
    #[serde(default)]
    pub held: Vec<GenerativeFace>,
    /// Faces that rest on something they can slide on: held along their normal only.
    /// Default: none.
    #[serde(default)]
    pub slide: Option<Vec<GenerativeFace>>,
    /// Round faces a pin goes through, such as the wall of a hole: held across and along the
    /// pin, free to turn about it. Default: none.
    #[serde(default)]
    pub pin: Option<Vec<GenerativeFace>>,
    /// The faces the first load acts on. Without them, `loads` gives every load.
    #[serde(default)]
    pub load: Vec<GenerativeFace>,
    /// The first load as a force in newtons, as a vector `[x, y, z]`; 9.81 N carry one
    /// kilogram. Give one of `force`, `moment` and `pressure` with `load`.
    #[serde(default)]
    pub force: Option<Point3>,
    /// The first load as a moment in newton metres, as a vector `[x, y, z]`: it turns the
    /// `load` faces about that vector through their middle.
    #[serde(default)]
    pub moment: Option<Point3>,
    /// The first load as a pressure in MPa on the `load` faces, against their outward normals.
    #[serde(default)]
    pub pressure: Option<f64>,
    /// Further loads, each with its own faces and a force, a moment or a pressure. Those of
    /// load case 1 act at the same time as `load`; each other case acts on its own, and the
    /// structure carries every case. At most 8. Default: none.
    #[serde(default)]
    pub loads: Option<Vec<GenerativeLoad>>,
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
    /// The thinnest wall or strut to make, in mm, from 0.2 to 100. Default: two and a half
    /// cells, and at least 1.5 % of the design space's longest side.
    #[serde(default)]
    pub wall: Option<f64>,
    /// The most the loaded faces may move under the force, in mm, from 0.001 to 1000. Default:
    /// 1/300 of the design space's longest side.
    #[serde(default)]
    pub deflection: Option<f64>,
    /// `lightest` (default): as little material as carries the force with the safety factor
    /// and within the deflection. `stiffest`: the stiffest structure made of `keep` percent of
    /// the design space. Organic style only.
    #[serde(default)]
    pub goal: Option<String>,
    /// For the `stiffest` goal: the share of the design space to fill, in percent, from 2 to 90.
    /// Default: 20.
    #[serde(default)]
    pub keep: Option<f64>,
    /// Axes to mirror the structure in, any of `x`, `y`, `z` in one word such as `x` or `xy`:
    /// the structure is its own mirror image in the plane through the middle of the design
    /// space that is normal to each. Organic style only. Default: none.
    #[serde(default)]
    pub symmetry: Option<String>,
    /// `x`, `y` or `z`: the structure has the same section everywhere along that axis, so it
    /// can be cut out of a plate. Organic style only. Default: a free shape.
    #[serde(default)]
    pub extrude: Option<String>,
    /// The direction a 3D printer builds the structure up in: `+x`, `-x`, `+y`, `-y`, `+z` or
    /// `-z`. The structure then stands without supports: nothing overhangs by more than 45
    /// degrees, counted from the picked faces' bodies and the lowest layer of the design space.
    /// Organic style only. Default: any shape.
    #[serde(default)]
    pub print: Option<String>,
    /// How far the structure stays away from the `avoid` bodies, in mm, from 0 to 100. Default: 0.
    #[serde(default)]
    pub clearance: Option<f64>,
    /// The longest the computation may take, in seconds, from 10 to 7200; the best shape found
    /// by then is finished. Default: 900.
    #[serde(default)]
    pub time: Option<f64>,
    /// The feature's name in the timeline.
    #[serde(default)]
    pub name: Option<String>,
    /// The new body's name. Default: `Generative`.
    #[serde(default)]
    pub body_name: Option<String>,
}

/// What a generative design returns: the feature it added and how the structure fared.
#[derive(Debug, Serialize, JsonSchema)]
pub struct GenerativeDesignResult {
    #[serde(flatten)]
    pub feature: FeatureResult,
    /// How the structure was computed and what it carries.
    pub design: GenerativeDesignReport,
}

/// How a generative design was computed and what the structure carries.
#[derive(Debug, Serialize, JsonSchema)]
pub struct GenerativeDesignReport {
    /// What solved the equations: the graphics card's name, or `CPU`.
    pub solver: String,
    /// How long the computation took, in seconds.
    pub seconds: f64,
    /// How many of those seconds went into solving the equations, which is the part a
    /// graphics card computes.
    pub solving: f64,
    /// The cells of the grid the structure was computed on.
    pub cells: u64,
    /// A cell's size in mm.
    pub cell: f64,
    /// The highest stress in the structure, in MPa.
    pub stress: f64,
    /// The stress the material and the safety factor allow, in MPa.
    pub allowed: f64,
    /// How far the loaded faces move under the force, in mm.
    pub deflection: f64,
    /// Whether the structure carries the force within the stress allowed.
    pub passed: bool,
    /// What else there is to say about the run, such as that it was stopped at the time limit.
    pub notes: Vec<String>,
}
