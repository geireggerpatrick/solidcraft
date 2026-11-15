//! Generative design: a light structure between held and loaded faces, as a mesh body.

use serde_json::Value;
use solidcore_record::Vec3;
use solidcraft_doc::{FeatureKind, MeshData, ModelState};
use solidcraft_generative::Material;
use solidcraft_generative::study::{self, Action, Report, Study};
use solidcraft_generative::voxel::{self as gv, Settings, Style, V3};

use super::{CommandSpec, face::face_at_state, features::add_feature};
use crate::schema::generative::{GenerativeDesignParams, GenerativeDesignReport, GenerativeDesignResult, GenerativeFace};
use crate::schema::{Point3, TypedContract, output_schema_for};
use crate::{Result, Session, params::bad};

const CMD: &str = "generative.design";
const DESIGN_CONTRACT: TypedContract<GenerativeDesignParams> = TypedContract::new();

pub static COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("generative.design", "Generative Design", crate::checked!("generative.design", design))
        .icon("import")
        .contract(&DESIGN_CONTRACT)
        .output(output_schema_for::<GenerativeDesignResult>)
        .params("held: [{body, point: [x,y,z]}] (faces that are held); slide?: [{body, point}] (faces held along their normal only, which slide in their surface); pin?: [{body, point}] (round faces on a pin: free to turn about it); load: [{body, point: [x,y,z]}] (faces the first load acts on); force: [x,y,z] in newtons, or moment: [x,y,z] in newton metres about the middle of the faces, or pressure: MPa onto the faces; loads?: [{faces: [{body, point}], force | moment | pressure, case?: 1…8}] (further loads; those of case 1, the default, act at the same time as `load`, each other case acts on its own and the structure carries every case); style?: organic|tree|lattice (default organic); space?: body to carve the structure out of (default: the box around the picked faces, around the bodies they belong to, which stay as they are); avoid?: [body] (further bodies to stay clear of); clearance?: mm to stay away from them; margin?: mm the design box reaches beyond the faces; strength?: MPa (default 35); modulus?: MPa (default 1900); safety?: 1…20 (default 2); voxels?: 500…2000000 cells (default 40000); wall?: thinnest strut in mm (default: two and a half cells, at least 1.5 % of the design space's longest side); deflection?: the most the loaded faces may move, in mm (default: 1/300 of the design space's longest side); goal?: lightest|stiffest; keep?: percent of the design space to fill for stiffest (default 20); symmetry?: axes to mirror in, such as x or xy; extrude?: x|y|z (one section along that axis); print?: +x|-x|+y|-y|+z|-z (stands without supports when printed that way up); time?: seconds the computation may take (default 900); name?; body_name? (default Generative): a new mesh body that carries the force from the loaded to the held faces with as little material as the strength allows"),
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

/// Evens the shading normals out along each smooth stretch of the surface, `rounds` times.
/// The triangles a grid gives repeat in a pattern, which would show as faint rings around a
/// round member. Corners at one vertex that share a normal are one point of the surface; a
/// crease has two, which do not mix.
fn even_normals(triangles: &[u32], normals: &mut [f64], rounds: usize) {
    if normals.len() != triangles.len() * 3 {
        return;
    }
    let mut ids = std::collections::HashMap::new();
    let mut points: Vec<V3> = Vec::new();
    let of: Vec<usize> = triangles
        .iter()
        .zip(normals.as_chunks::<3>().0)
        .map(|(vertex, n)| {
            let next = points.len();
            let id = *ids.entry((*vertex, n.map(|c| (c * 1e6).round() as i64))).or_insert(next);
            if id == next {
                points.push(*n);
            }
            id
        })
        .collect();
    for _ in 0..rounds {
        // A neighbour along the surface is in two triangles of a point, so it counts once.
        let mut sum = points.clone();
        for t in of.as_chunks::<3>().0 {
            for (a, b) in [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)] {
                for x in 0..3 {
                    sum[t[a]][x] += 0.5 * points[t[b]][x];
                }
            }
        }
        for (point, sum) in points.iter_mut().zip(sum) {
            let length = sum.iter().map(|c| c * c).sum::<f64>().sqrt();
            if length > 1e-9 {
                *point = sum.map(|c| c / length);
            }
        }
    }
    for (corner, n) in of.iter().zip(normals.as_chunks_mut::<3>().0) {
        *n = points[*corner];
    }
}

/// Seconds since the first design, for the study to time its equation solver with.
fn clock() -> f64 {
    static START: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(web_time::Instant::now).elapsed().as_secs_f64()
}

fn in_range(v: Option<f64>, default: f64, lo: f64, hi: f64, key: &str) -> Result<f64> {
    let v = v.unwrap_or(default);
    if v.is_finite() && v >= lo && v <= hi { Ok(v) } else { Err(bad(CMD, format!("`{key}` must be from {lo} to {hi}"))) }
}

/// What a load is: exactly one of a force in newtons, a moment in newton metres and a pressure
/// in MPa. `key` names the load in the message.
fn action(force: Option<&Point3>, moment: Option<&Point3>, pressure: Option<f64>, key: &str) -> Result<Action> {
    let size = |f: V3| f.iter().map(|c| c * c).sum::<f64>().sqrt();
    match (force, moment, pressure) {
        (Some(f), None, None) if size(v3(f.0)) > 1e-6 && size(v3(f.0)) < 1e7 => Ok(Action::Force(v3(f.0))),
        (Some(_), None, None) => Err(bad(CMD, format!("{key}: `force` must be a vector in newtons, not zero"))),
        // The study counts in newtons and millimetres.
        (None, Some(m), None) if size(v3(m.0)) > 1e-9 && size(v3(m.0)) < 1e6 => Ok(Action::Moment(v3(m.0).map(|c| 1000.0 * c))),
        (None, Some(_), None) => Err(bad(CMD, format!("{key}: `moment` must be a vector in newton metres, not zero"))),
        (None, None, Some(p)) if p.is_finite() && p.abs() > 1e-9 && p.abs() <= 1000.0 => Ok(Action::Pressure(p)),
        (None, None, Some(_)) => Err(bad(CMD, format!("{key}: `pressure` must be a number in MPa up to 1000, not zero"))),
        _ => Err(bad(CMD, format!("{key}: give one of `force`, `moment` and `pressure`"))),
    }
}

fn design(s: &mut Session, p: &Value) -> Result<Value> {
    let q: GenerativeDesignParams = serde_json::from_value(p.clone()).map_err(|e| bad(CMD, e.to_string()))?;
    let (slide, pin) = (q.slide.as_deref().unwrap_or(&[]), q.pin.as_deref().unwrap_or(&[]));
    let further = q.loads.as_deref().unwrap_or(&[]);
    if q.held.is_empty() && slide.is_empty() && pin.is_empty() {
        return Err(bad(CMD, "give at least one face that is held: `held`, `slide` or `pin`"));
    }
    if q.load.is_empty() && further.is_empty() {
        return Err(bad(CMD, "give at least one load: `load` faces with a `force`, or `loads`"));
    }
    if q.held.len() > 64 || slide.len() > 64 || pin.len() > 64 || q.load.len() > 64 {
        return Err(bad(CMD, "at most 64 faces each in `held`, `slide`, `pin` and `load`"));
    }
    if further.len() > 8 || further.iter().any(|l| l.faces.is_empty() || l.faces.len() > 64) {
        return Err(bad(CMD, "`loads`: at most 8, each with 1 to 64 faces"));
    }
    if further.iter().any(|l| l.case.is_some_and(|c| !(1..=8).contains(&c))) {
        return Err(bad(CMD, "`loads`: each `case` must be a number from 1 to 8"));
    }
    // Every load with its faces and its case counted from 0, the case of `load`.
    let mut wanted: Vec<(&[GenerativeFace], Action, usize)> = Vec::new();
    if !q.load.is_empty() {
        wanted.push((&q.load, action(q.force.as_ref(), q.moment.as_ref(), q.pressure, "`load`")?, 0));
    } else if q.force.is_some() || q.moment.is_some() || q.pressure.is_some() {
        return Err(bad(CMD, "`force`, `moment` and `pressure` act on the `load` faces: give them"));
    }
    for l in further {
        let case = l.case.map_or(0, |c| c.saturating_sub(1) as usize);
        wanted.push((&l.faces, action(l.force.as_ref(), l.moment.as_ref(), l.pressure, "`loads`")?, case));
    }
    // The study's first load is in its case 0: with no `load`, the first of `loads` is the
    // first, and its case changes places with case 1.
    let first = wanted.first().map_or(0, |w| w.2);
    for w in &mut wanted {
        w.2 = match w.2 {
            c if c == first => 0,
            0 => first,
            c => c,
        };
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
    let clearance = in_range(q.clearance, 0.0, 0.0, 100.0, "clearance")?;
    let time = in_range(q.time, 900.0, 10.0, 7200.0, "time")?;
    let volume = match q.goal.as_deref().unwrap_or("lightest") {
        "lightest" => None,
        "stiffest" => Some(in_range(q.keep, 20.0, 2.0, 90.0, "keep")? / 100.0),
        _ => return Err(bad(CMD, "`goal` must be lightest or stiffest")),
    };
    let mirror = q.symmetry.as_deref().unwrap_or("").trim().to_ascii_lowercase();
    if !mirror.bytes().all(|c| matches!(c, b'x' | b'y' | b'z')) {
        return Err(bad(CMD, "`symmetry` must name axes: x, y, z or several, such as xy"));
    }
    let symmetry = (*b"xyz").map(|axis| mirror.bytes().any(|c| c == axis));
    let print = match q.print.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
        None => None,
        Some(way) => {
            let axis = ["x", "y", "z"].iter().position(|a| way.to_ascii_lowercase().ends_with(a));
            let sign = way.len() == 2 && (way.starts_with('+') || way.starts_with('-'));
            Some((axis.filter(|_| sign).ok_or_else(|| bad(CMD, "`print` must be +x, -x, +y, -y, +z or -z"))?, way.starts_with('+')))
        }
    };
    let extrude = match q.extrude.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
        None => None,
        Some(axis) => Some(["x", "y", "z"].iter().position(|a| a.eq_ignore_ascii_case(axis)).ok_or_else(|| bad(CMD, "`extrude` must be x, y or z"))?),
    };

    let st = s.model.state();
    let picks = |faces: &[GenerativeFace], key: &str| -> Result<Vec<Picked>> { faces.iter().map(|f| pick(s, &st, f, key)).collect() };
    let held = picks(&q.held, "held")?;
    let slide = picks(slide, "slide")?;
    let pin = picks(pin, "pin")?;
    let mut loads: Vec<(Vec<Picked>, Action, usize)> = Vec::new();
    for (faces, action, case) in &wanted {
        loads.push((picks(faces, if loads.is_empty() && !q.load.is_empty() { "load" } else { "loads" })?, *action, *case));
    }
    let mut loads = loads.into_iter();
    let Some((load, first_action, _)) = loads.next() else { return Err(bad(CMD, "give at least one load")) };
    let more: Vec<(Vec<Picked>, Action, usize)> = loads.collect();
    // Springs hold a face that slides or turns, and those the built-in solver computes.
    let guided = !slide.is_empty() || !pin.is_empty();
    let space = match q.space.as_deref().filter(|n| !n.trim().is_empty()) {
        Some(n) => Some(s.own_body_name(n).ok_or_else(|| bad(CMD, format!("`space`: no body `{n}`")))?),
        None => None,
    };
    // The bodies to build around: those of the picked faces (unless the structure is carved
    // out of one body) and the ones to avoid.
    let mut obstacles: Vec<String> = Vec::new();
    if space.is_none() {
        obstacles.extend(held.iter().chain(&slide).chain(&pin).chain(&load).chain(more.iter().flat_map(|l| &l.0)).map(|f| f.body.clone()));
    }
    obstacles.sort();
    obstacles.dedup();
    // The structure keeps the clearance to the bodies to avoid, not to the ones it stands on.
    let mut avoid: Vec<String> = Vec::new();
    for n in q.avoid.iter().flat_map(|l| &l.0) {
        avoid.push(s.own_body_name(n).ok_or_else(|| bad(CMD, format!("`avoid`: no body `{n}`")))?);
    }
    avoid.sort();
    avoid.dedup();
    avoid.retain(|n| space.as_deref() != Some(n.as_str()) && !obstacles.contains(n));

    let space = match &space {
        Some(name) => Some(body_mesh(&st, name).ok_or_else(|| bad(CMD, format!("`space`: no body `{name}`")))?),
        None => None,
    };
    let study = Study {
        held: held.into_iter().map(|f| f.tris).collect(),
        slide: slide.into_iter().map(|f| f.tris).collect(),
        pin: pin.into_iter().map(|f| f.tris).collect(),
        load: load.into_iter().map(|f| f.tris).collect(),
        action: first_action,
        more: more.into_iter().map(|(faces, action, case)| (faces.into_iter().map(|f| f.tris).collect(), action, case)).collect(),
        space,
        obstacles: obstacles.iter().filter_map(|n| body_mesh(&st, n)).collect(),
        avoid: avoid.iter().filter_map(|n| body_mesh(&st, n)).collect(),
        clearance,
        margin,
        voxels,
        settings: Settings {
            style,
            material: Material { strength, modulus, density: 1.27 },
            safety,
            volume,
            min_wall: wall,
            max_deflection: deflection,
            accelerate: true,
            symmetry,
            extrude,
            print,
        },
    };
    let began = web_time::Instant::now();
    gv::set_clock(clock);
    gv::take_solve_seconds();
    let Report { mesh, outcome, cell, cells } = study::run(&study, &mut || began.elapsed().as_secs_f64() < time).map_err(|e| bad(CMD, e))?;
    // A graphics card that failed on the way was dropped, and the CPU finished the work.
    let report = GenerativeDesignReport {
        solver: gv::grid_solver_name().filter(|_| !guided).unwrap_or_else(|| "CPU".into()),
        seconds: began.elapsed().as_secs_f64(),
        solving: gv::take_solve_seconds(),
        cells: cells as u64,
        cell,
        stress: outcome.stress,
        allowed: outcome.allowed,
        deflection: outcome.deflection,
        passed: outcome.passed,
        notes: outcome.note.clone(),
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
    data.generate_crease_normals(60.0).map_err(|e| bad(CMD, e.to_string()))?;
    let mut normals = data.corner_normals.to_vec();
    even_normals(&data.triangles, &mut normals, 4);
    data.corner_normals = normals.into();
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::Session;

    fn wall_and_shelf() -> Session {
        let mut s = Session { param_check: crate::schema::ParamCheck::Strict, ..Session::default() };
        for (name, corner, size) in [("Wall", [0, 0, 0], [5, 40, 50]), ("Shelf", [20, 5, 50], [40, 30, 4])] {
            let params = json!({"length": size[0], "width": size[1], "height": size[2], "corner": corner, "operation": "new", "body_name": name});
            s.execute("solid.box", &params).unwrap();
        }
        s
    }

    fn design(extra: serde_json::Value) -> serde_json::Value {
        let mut params = json!({
            "held": [{"body": "Wall", "point": [5, 20, 25]}],
            "load": [{"body": "Shelf", "point": [40, 20, 50]}],
            "force": [0, 0, -100],
            "voxels": 3000,
        });
        for (key, value) in extra.as_object().unwrap() {
            params[key] = value.clone();
        }
        params
    }

    /// The structure is one closed mesh body, so a section view fills it, shaded as a smooth
    /// shape, and the reply says how it was computed.
    #[test]
    fn a_design_is_a_closed_smooth_body_with_a_report() {
        let mut s = wall_and_shelf();
        let reply = s.execute("generative.design", &design(json!({"symmetry": "y", "print": "+z", "clearance": 1, "time": 600}))).unwrap();
        assert!(reply["design"]["seconds"].as_f64().is_some() && reply["design"]["notes"].is_array(), "{reply}");
        let st = s.model.state();
        let body = st.bodies.iter().find(|b| b.name == "Generative").expect("the new body");
        assert!(body.body.is_mesh() && body.body.is_closed_mesh());
        assert!(body.body.triangle_corner_normals().is_some_and(|n| !n.is_empty()));
    }

    /// A second force on a face of a third body: the structure reaches that body too, whether
    /// the force acts together with the first (case 1) or as a load case of its own.
    #[test]
    fn a_further_load_is_carried_too() {
        for case in [1, 2] {
            let mut s = wall_and_shelf();
            let rail = json!({"length": 40, "width": 4, "height": 10, "corner": [20, 36, 20], "operation": "new", "body_name": "Rail"});
            s.execute("solid.box", &rail).unwrap();
            let loads = json!([{"faces": [{"body": "Rail", "point": [40, 36, 25]}], "force": [0, 30, 0], "case": case}]);
            let reply = s.execute("generative.design", &design(json!({"loads": loads, "voxels": 6000}))).unwrap();
            assert!(reply["design"]["cells"].as_u64().is_some(), "{reply}");
            let st = s.model.state();
            let body = st.bodies.iter().find(|b| b.name == "Generative").expect("the new body");
            let mesh = body.mesh();
            let near_rail = mesh.triangles.iter().filter_map(|t| mesh.tri(t)).flatten().any(|p| p.y > 34.0 && p.z > 19.0 && p.z < 31.0);
            assert!(body.body.is_closed_mesh() && near_rail, "case {case}");
        }
    }

    /// Two faces of one body that meet in an edge: neither face's plane shuts the other face
    /// out, and a structure that carries both comes out.
    #[test]
    fn two_faces_of_a_body_that_meet_in_an_edge_are_both_loaded() {
        let mut s = wall_and_shelf();
        let rail = json!({"length": 40, "width": 4, "height": 10, "corner": [20, 36, 20], "operation": "new", "body_name": "Rail"});
        s.execute("solid.box", &rail).unwrap();
        let loads = json!([
            {"faces": [{"body": "Rail", "point": [40, 36, 25]}], "force": [0, 30, 0]},
            {"faces": [{"body": "Rail", "point": [40, 38, 30]}], "force": [0, 0, -20]},
        ]);
        let reply = s.execute("generative.design", &design(json!({"loads": loads, "voxels": 6000}))).unwrap();
        assert!(reply["design"]["stress"].as_f64().is_some_and(|stress| stress > 0.0), "{reply}");
        let st = s.model.state();
        let body = st.bodies.iter().find(|b| b.name == "Generative").expect("the new body");
        assert!(body.body.is_closed_mesh());
    }

    /// A moment and a pressure are loads as a force is, a face may slide on what holds it, and
    /// a hole may turn on its pin: the structure carries them, computed by the built-in solver.
    #[test]
    fn moments_pressures_and_guided_supports_are_carried() {
        let mut s = wall_and_shelf();
        let params = json!({"length": 5, "width": 40, "height": 20, "corner": [75, 0, 0], "operation": "new", "body_name": "Foot"});
        s.execute("solid.box", &params).unwrap();
        // Held on the wall and resting on the foot's inner side, which only holds along x; the
        // shelf is turned about y and pressed on from below.
        let shelf = json!([{"body": "Shelf", "point": [40, 20, 50]}]);
        let params = json!({
            "held": [{"body": "Wall", "point": [5, 20, 25]}],
            "slide": [{"body": "Foot", "point": [75, 20, 10]}],
            "load": shelf,
            "moment": [0, 2, 0],
            "loads": [{"faces": [{"body": "Shelf", "point": [20, 20, 52]}], "pressure": 0.2, "case": 2}],
            "voxels": 4000,
        });
        let reply = s.execute("generative.design", &params).unwrap();
        assert_eq!(reply["design"]["solver"], "CPU", "{reply}");
        assert!(reply["design"]["stress"].as_f64().is_some_and(|stress| stress > 0.0 && stress.is_finite()), "{reply}");
        assert!(reply["design"]["deflection"].as_f64().is_some_and(|d| d > 0.0 && d < 50.0), "{reply}");
        let st = s.model.state();
        assert!(st.bodies.iter().find(|b| b.name == "Generative").expect("the new body").body.is_closed_mesh());
    }

    /// Normals that waver from corner to corner of a flat stretch come out alike, and the two
    /// sides of a crease keep their own.
    #[test]
    fn shading_normals_are_evened_out_along_the_surface_only() {
        // Two triangles of a flat stretch (vertices 0 to 3) and one behind a crease at 1-2.
        let triangles = [0, 1, 2, 2, 1, 3, 1, 4, 2];
        let tilt = |x: f64| [x, 0.0, (1.0 - x * x).sqrt()];
        let side = [1.0, 0.0, 0.0];
        let mut normals: Vec<f64> = [tilt(0.1), tilt(-0.1), tilt(0.05), tilt(0.05), tilt(-0.1), tilt(-0.05), side, side, side].concat();
        super::even_normals(&triangles, &mut normals, 20);
        let n = normals.as_chunks::<3>().0;
        assert!(n[..6].iter().all(|c| (c[0] - n[0][0]).abs() < 1e-3 && c[2] > 0.99), "{n:?}");
        assert!(n[6..].iter().all(|c| *c == side));
    }

    #[test]
    fn settings_out_of_range_are_refused() {
        let mut s = wall_and_shelf();
        let before = s.doc.to_json();
        for extra in [
            json!({"goal": "cheapest"}),
            json!({"goal": "stiffest", "keep": 500}),
            json!({"symmetry": "q"}),
            json!({"extrude": "w"}),
            json!({"print": "z"}),
            json!({"clearance": -1}),
            json!({"time": 1}),
            json!({"loads": [{"faces": [], "force": [0, 0, 1]}]}),
            json!({"loads": [{"faces": [{"body": "Shelf", "point": [40, 20, 50]}], "force": [0, 0, 1], "case": 9}]}),
            json!({"loads": [{"faces": [{"body": "Shelf", "point": [40, 20, 50]}], "force": [0, 0, 0]}]}),
            json!({"loads": [{"faces": [{"body": "Shelf", "point": [40, 20, 50]}], "force": [0, 0, 1], "pressure": 1}]}),
            json!({"loads": [{"faces": [{"body": "Shelf", "point": [40, 20, 50]}]}]}),
            json!({"moment": [0, 1, 0]}),
            json!({"pressure": 2000, "force": null}),
            // A pin needs a round face.
            json!({"pin": [{"body": "Wall", "point": [5, 20, 25]}], "held": []}),
        ] {
            assert!(s.execute("generative.design", &design(extra.clone())).is_err(), "{extra}");
            assert_eq!(s.doc.to_json(), before);
        }
    }
}
