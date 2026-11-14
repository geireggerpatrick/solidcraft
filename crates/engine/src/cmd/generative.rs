//! Generative design: a light structure between held and loaded faces, as a mesh body.

use serde_json::Value;
use solidcore_record::Vec3;
use solidcraft_doc::{FeatureKind, MeshData, ModelState};
use solidcraft_generative::Material;
use solidcraft_generative::study::{self, Report, Study};
use solidcraft_generative::voxel::{self as gv, Settings, Style, V3};

use super::{CommandSpec, face::face_at_state, features::add_feature};
use crate::schema::generative::{GenerativeDesignParams, GenerativeDesignReport, GenerativeDesignResult, GenerativeFace};
use crate::schema::{TypedContract, output_schema_for};
use crate::{Result, Session, params::bad};

const CMD: &str = "generative.design";
const DESIGN_CONTRACT: TypedContract<GenerativeDesignParams> = TypedContract::new();

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("generative.design", "Generative Design", crate::checked!("generative.design", design))
        .icon("import")
        .contract(&DESIGN_CONTRACT)
        .output(output_schema_for::<GenerativeDesignResult>)
        .params("held: [{body, point: [x,y,z]}] (faces that are held); load: [{body, point: [x,y,z]}] (faces the force acts on); force: [x,y,z] in newtons; style?: organic|tree|lattice (default organic); space?: body to carve the structure out of (default: the box around the picked faces, around the bodies they belong to, which stay as they are); avoid?: [body] (further bodies to stay clear of); margin?: mm the design box reaches beyond the faces; strength?: MPa (default 35); modulus?: MPa (default 1900); safety?: 1…20 (default 2); voxels?: 500…2000000 cells (default 40000); wall?: thinnest strut in mm (default: 4 % of the design space's longest side); deflection?: the most the loaded faces may move, in mm (default: 1/400 of the design space's longest side); name?; body_name? (default Generative): a new mesh body that carries the force from the loaded to the held faces with as little material as the strength allows"),
];

fn v3(p: Vec3) -> V3 {
    [p.x, p.y, p.z]
}

/// A picked face: its body and its triangles.
struct Picked {
    body: String,
    tris: Vec<[V3; 3]>,
}

fn pick(s: &Session, st: &ModelState, f: &GenerativeFace, key: &str) -> Result<Picked> {
    let body = s.own_body_name(&f.body).ok_or_else(|| bad(CMD, format!("`{key}`: no body `{}`", f.body)))?;
    let found = face_at_state(st, f.point.0, Some(&body)).map_err(|_| bad(CMD, format!("`{key}`: the point is on faces of several bodies")))?;
    let Some((mesh, face, dist, _)) = found else { return Err(bad(CMD, format!("`{key}`: body `{}` has no faces", f.body))) };
    if !(dist <= 1.0) {
        return Err(bad(CMD, format!("`{key}`: the point is {dist:.2} mm away from body `{}`; give a point on the face", f.body)));
    }
    let tris: Vec<[V3; 3]> =
        mesh.triangles.iter().zip(&mesh.tri_face).filter(|(_, tf)| **tf == face).filter_map(|(t, _)| mesh.tri(t)).map(|t| t.map(v3)).collect();
    if tris.is_empty() {
        return Err(bad(CMD, format!("`{key}`: no face at the point")));
    }
    Ok(Picked { body, tris })
}

/// A body's triangles as a closed mesh for the voxel grid.
fn body_mesh(st: &ModelState, name: &str) -> Option<gv::Mesh> {
    let m = st.body(name)?.mesh();
    let mut out = gv::Mesh::default();
    for t in m.triangles.iter() {
        let Some(tri) = m.tri(t) else { continue };
        let i = out.pos.len();
        out.pos.extend(tri.map(v3));
        out.tri.push([i, i + 1, i + 2]);
    }
    Some(out)
}

fn in_range(v: Option<f64>, default: f64, lo: f64, hi: f64, key: &str) -> Result<f64> {
    let v = v.unwrap_or(default);
    if v.is_finite() && v >= lo && v <= hi { Ok(v) } else { Err(bad(CMD, format!("`{key}` must be from {lo} to {hi}"))) }
}

fn design(s: &mut Session, p: &Value) -> Result<Value> {
    let q: GenerativeDesignParams = serde_json::from_value(p.clone()).map_err(|e| bad(CMD, e.to_string()))?;
    if q.held.is_empty() || q.load.is_empty() {
        return Err(bad(CMD, "give at least one `held` face and one `load` face"));
    }
    if q.held.len() > 64 || q.load.len() > 64 {
        return Err(bad(CMD, "at most 64 `held` and 64 `load` faces"));
    }
    let force = v3(q.force.0);
    let newtons = force.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !(newtons > 1e-6 && newtons < 1e7) {
        return Err(bad(CMD, "`force` must be a vector in newtons, not zero"));
    }
    let style = match q.style.as_deref().unwrap_or(gv::STYLE_ORGANIC) {
        gv::STYLE_ORGANIC => Style::Organic,
        gv::STYLE_TREE => Style::Tree,
        gv::STYLE_LATTICE => Style::Lattice,
        _ => return Err(bad(CMD, "`style` must be organic, tree or lattice")),
    };
    let strength = in_range(q.strength, 35.0, 1.0, 5000.0, "strength")?;
    let modulus = in_range(q.modulus, 1900.0, 10.0, 1e6, "modulus")?;
    let safety = in_range(q.safety, 2.0, 1.0, 20.0, "safety")?;
    let voxels = in_range(q.voxels, 40_000.0, 500.0, 2_000_000.0, "voxels")?;
    let wall = match q.wall {
        None => 0.0,
        Some(_) => in_range(q.wall, 1.0, 0.2, 100.0, "wall")?,
    };
    let deflection = match q.deflection {
        None => 0.0,
        Some(_) => in_range(q.deflection, 1.0, 0.001, 1000.0, "deflection")?,
    };
    let margin = in_range(q.margin, 0.0, 0.0, 1000.0, "margin")?;

    let st = s.model.state();
    let held: Vec<Picked> = q.held.iter().map(|f| pick(s, &st, f, "held")).collect::<Result<_>>()?;
    let load: Vec<Picked> = q.load.iter().map(|f| pick(s, &st, f, "load")).collect::<Result<_>>()?;
    let space = match q.space.as_deref().filter(|n| !n.trim().is_empty()) {
        Some(n) => Some(s.own_body_name(n).ok_or_else(|| bad(CMD, format!("`space`: no body `{n}`")))?),
        None => None,
    };
    // The bodies to build around: those of the picked faces (unless the structure is carved
    // out of one body) and the ones to avoid.
    let mut obstacles: Vec<String> = Vec::new();
    if space.is_none() {
        obstacles.extend(held.iter().chain(&load).map(|f| f.body.clone()));
    }
    for n in q.avoid.iter().flat_map(|l| &l.0) {
        obstacles.push(s.own_body_name(n).ok_or_else(|| bad(CMD, format!("`avoid`: no body `{n}`")))?);
    }
    obstacles.sort();
    obstacles.dedup();
    obstacles.retain(|n| space.as_deref() != Some(n.as_str()));

    let space = match &space {
        Some(name) => Some(body_mesh(&st, name).ok_or_else(|| bad(CMD, format!("`space`: no body `{name}`")))?),
        None => None,
    };
    let study = Study {
        held: held.into_iter().map(|f| f.tris).collect(),
        load: load.into_iter().map(|f| f.tris).collect(),
        force,
        space,
        obstacles: obstacles.iter().filter_map(|n| body_mesh(&st, n)).collect(),
        margin,
        voxels,
        settings: Settings {
            style,
            material: Material { strength, modulus, density: 1.27 },
            safety,
            volume: None,
            min_wall: wall,
            max_deflection: deflection,
            accelerate: true,
        },
    };
    let began = web_time::Instant::now();
    let Report { mesh, outcome, cell, cells } = study::run(&study, &mut || true).map_err(|e| bad(CMD, e))?;
    // A graphics card that failed on the way was dropped, and the CPU finished the work.
    let report = GenerativeDesignReport {
        solver: gv::grid_solver_name().unwrap_or_else(|| "CPU".into()),
        seconds: began.elapsed().as_secs_f64(),
        cells: cells as u64,
        cell,
        stress: outcome.stress,
        allowed: outcome.allowed,
        deflection: outcome.deflection,
        passed: outcome.passed,
    };
    if mesh.pos.len() > u32::MAX as usize {
        return Err(bad(CMD, "the result has too many triangles; give fewer `voxels`"));
    }
    let name = q.body_name.as_deref().map(str::trim).filter(|n| !n.is_empty()).unwrap_or("Generative").to_string();
    let mut data = MeshData {
        name,
        positions: mesh.pos.iter().flatten().copied().collect(),
        triangles: mesh.tri.iter().flatten().map(|v| *v as u32).collect(),
        corner_normals: Default::default(),
        color: None,
    };
    // Shaded as the round shape it is, not as its triangles: only the flat cuts on the picked
    // faces keep their edge.
    data.generate_crease_normals(50.0).map_err(|e| bad(CMD, e.to_string()))?;
    let mut reply = add_feature(s, p, FeatureKind::MeshImport { file: "generative design".into(), meshes: vec![data] })?;
    if let Some(o) = reply.as_object_mut() {
        if !outcome.passed {
            let text = format!(
                "the structure reaches {:.1} MPa where {:.1} MPa are allowed; give it more room, a stronger material or a smaller force",
                outcome.stress, outcome.allowed
            );
            o.insert("warning".into(), Value::String(text));
        }
        o.insert("design".into(), serde_json::to_value(&report).map_err(|e| bad(CMD, e.to_string()))?);
    }
    Ok(reply)
}
