//! What the Generative Design window's settings mean, drawn over the model while the window is
//! open: the design space as a box (which Room around widens), the held faces and those that
//! slide or turn on a pin, each load on its faces (a force as an arrow, a moment as an arrow with
//! two heads along its axis, a pressure as a row of arrows), the mirror planes, the way the structure is made, the room kept
//! to the bodies to avoid, and how large a cell and the thinnest member are.
//!
//! Display only: the command works its own design space out from the faces.

use egui::{Align2, Color32, FontId, Pos2, Shape, Stroke, vec2};
use solidcore_record::Vec3;
use solidcraft_engine::Sel;
use solidcraft_engine::geom::prelude::*;
use solidcraft_theme::Tokens;

use crate::SolveApp;
use crate::dialogs::{Dialog, GENERATIVE_DETAIL, GENERATIVE_MAKING, GENERATIVE_MIRRORS, Kind, generative_load_input};
use crate::viewport::{Proj, projected_polygon};

/// A face drawn in its role's colour takes at most this many triangles; a finer one gets its
/// mark only.
const MOST_TRIANGLES: usize = 600;

/// A load on a loaded face.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Force {
    /// The middle of the face.
    pub at: Vec3,
    /// The way a force or a pressure points, or the axis a moment turns about (unit).
    pub dir: Vec3,
    /// Whether it presses onto the face (its arrow ends on the face) or pulls at it (its arrow
    /// starts there).
    pub onto: bool,
    /// How much: newtons, newton metres or MPa.
    pub amount: f64,
    /// 0: a force; 1: a moment; 2: a pressure.
    pub kind: usize,
    /// Which load, from 0.
    pub load: usize,
}

/// How the structure is going to be made.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Making {
    /// Printed from the middle of the box's lowest side (`from`) upwards along `up`, `height`
    /// high, in layers (`layer`: the first one's corners).
    Print { from: Vec3, up: Vec3, height: f64, layer: [Vec3; 4] },
    /// One section all along an axis: the box's section at both ends.
    Cut { ends: [[Vec3; 4]; 2] },
}

/// Everything the window's settings show in the view.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Preview {
    /// The design space: the box around the picked faces with the room around, or the design
    /// space body's box.
    pub space: Option<(Vec3, Vec3)>,
    /// The held faces: their middles and triangles.
    pub held: Vec<(Vec3, Vec<[Vec3; 3]>)>,
    /// The faces that slide (`false`) or turn on a pin (`true`): their middles and triangles.
    pub guided: Vec<(bool, Vec3, Vec<[Vec3; 3]>)>,
    /// The loaded faces' triangles, with the load they belong to.
    pub loaded: Vec<(usize, Vec<[Vec3; 3]>)>,
    pub forces: Vec<Force>,
    /// The mirror planes' corners in the design space.
    pub mirrors: Vec<[Vec3; 4]>,
    pub making: Option<Making>,
    /// The boxes the structure stays out of: the bodies to avoid, wider by the clearance.
    pub keep_out: Vec<(Vec3, Vec3)>,
    /// A cell's size in mm.
    pub cell: Option<f64>,
    /// The thinnest member in mm: as typed, or what the command takes when the field is empty.
    pub member: Option<f64>,
}

fn tr(text: &str) -> &str {
    solidcraft_ui_text::tr_label(text)
}

/// A number as typed, with a decimal comma or point, above zero.
fn number(text: &str) -> Option<f64> {
    text.trim().replace(',', ".").parse().ok().filter(|n: &f64| n.is_finite() && *n > 0.0)
}

/// The triangles of a picked face.
fn face_triangles(st: &solidcraft_engine::doc::ModelState, x: &Sel) -> Option<Vec<[Vec3; 3]>> {
    let Sel::Face { body, index, .. } = x else { return None };
    let mesh = st.body(body)?.mesh();
    let tris: Vec<[Vec3; 3]> =
        mesh.triangles.iter().zip(&mesh.tri_face).filter(|(_, face)| **face as usize == *index).filter_map(|(t, _)| mesh.tri(t)).collect();
    (!tris.is_empty()).then_some(tris)
}

/// The middle of a face's triangles (by area) and its outward normal when it is flat.
fn middle(tris: &[[Vec3; 3]]) -> (Vec3, Option<Vec3>) {
    let mut sum = Vec3::default();
    let mut normal = Vec3::default();
    let mut area = 0.0;
    for [a, b, c] in tris {
        let n = (*b - *a).cross(*c - *a);
        let size = n.len();
        sum += (*a + *b + *c) * (size / 3.0);
        normal += n;
        area += size;
    }
    if !(area > 1e-12) {
        return (tris.first().map_or(Vec3::default(), |t| t[0]), None);
    }
    // Flat: the triangles' normals add up to as much as their areas.
    (sum * (1.0 / area), (normal.len() > 0.999 * area).then(|| normal.normalized()).flatten())
}

/// The corners of the box's section through `at` square to axis `axis` (0, 1, 2).
fn section(lo: Vec3, hi: Vec3, axis: usize, at: f64) -> [Vec3; 4] {
    let corner = |u: bool, v: bool| {
        let pick = |a: usize, low: f64, high: f64| {
            if a == axis {
                at
            } else if (a == (axis + 1) % 3 && u) || (a == (axis + 2) % 3 && v) {
                high
            } else {
                low
            }
        };
        Vec3::new(pick(0, lo.x, hi.x), pick(1, lo.y, hi.y), pick(2, lo.z, hi.z))
    };
    [corner(false, false), corner(true, false), corner(true, true), corner(false, true)]
}

fn component(v: Vec3, axis: usize) -> f64 {
    [v.x, v.y, v.z][axis.min(2)]
}

/// What the open Generative Design window shows in the view; `None` for any other window.
pub(crate) fn preview(app: &SolveApp, d: &Dialog) -> Option<Preview> {
    let Kind::Generative { loads, detail, wall, margin, mirror, making, clearance, .. } = &d.kind else {
        return None;
    };
    let st = crate::open_meshing::session_world(&app.session);
    let faces = |i: usize| -> Vec<Vec<[Vec3; 3]>> {
        d.inputs.get(i).map(|input| input.items.iter().filter_map(|x| face_triangles(&st, x)).collect()).unwrap_or_default()
    };
    let (held, slide, pin) = (faces(0), faces(4), faces(5));
    let mut out = Preview::default();

    // Each load on its faces. A pressure acts against a flat face's normal.
    let mut loaded: Vec<Vec<[Vec3; 3]>> = Vec::new();
    for (load_index, load) in loads.iter().enumerate() {
        let way = load.way().map(|v| Vec3::new(v[0], v[1], v[2]));
        for tris in faces(generative_load_input(load_index)) {
            let (at, normal) = middle(&tris);
            let dir = if load.kind == 2 { normal.map(|n| -n) } else { way };
            if let (Some(dir), Some(amount)) = (dir, number(&load.amount)) {
                let onto = load.kind == 2 || normal.is_none_or(|n| n.dot(dir) <= 0.0);
                out.forces.push(Force { at, dir, onto, amount, kind: load.kind.min(2), load: load_index });
            }
            out.loaded.push((load_index, tris.clone()));
            loaded.push(tris);
        }
    }
    out.guided = [(false, &slide), (true, &pin)]
        .into_iter()
        .flat_map(|(turns, list)| list.iter().map(move |tris| (turns, middle(tris).0, tris.clone())))
        .collect();
    out.held = held.iter().map(|tris| (middle(tris).0, tris.clone())).collect();

    // The design space: a body, or the box around the faces, wider by the room around; a flat
    // box is given some depth, as the command does.
    let body_box = |name: &str| {
        let b = st.body(name)?.mesh().bounds();
        Some((b.min, b.max))
    };
    let bodies = |i: usize| -> Vec<&str> {
        d.inputs
            .get(i)
            .map(|input| input.items.iter().filter_map(|x| if let Sel::Body { name } = x { Some(name.as_str()) } else { None }).collect())
            .unwrap_or_default()
    };
    let around = number(margin).unwrap_or(0.0);
    out.space = match bodies(3).first() {
        Some(name) => body_box(name),
        None => {
            let mut points = held.iter().chain(&slide).chain(&pin).chain(&loaded).flatten().flatten();
            points.next().map(|p| {
                let (mut lo, mut hi) = (*p, *p);
                for p in points {
                    lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                    hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
                }
                let pad = Vec3::new(around, around, around);
                let (lo, hi) = (lo - pad, hi + pad);
                let size = hi - lo;
                let longest = size.x.max(size.y).max(size.z);
                let grow = |s: f64| ((0.3 * longest - s) / 2.0).max(0.0);
                let grow = Vec3::new(grow(size.x), grow(size.y), grow(size.z));
                (lo - grow, hi + grow)
            })
        }
    };
    let clear = number(clearance).unwrap_or(0.0);
    out.keep_out = bodies(2)
        .iter()
        .filter_map(|name| body_box(name))
        .map(|(lo, hi)| (lo - Vec3::new(clear, clear, clear), hi + Vec3::new(clear, clear, clear)))
        .collect();

    let Some((lo, hi)) = out.space else { return Some(out) };
    let size = hi - lo;
    let centre = (lo + hi) * 0.5;
    if let Some((_, axes)) = GENERATIVE_MIRRORS.get(*mirror) {
        out.mirrors = (0..3).filter(|a| axes.contains(["x", "y", "z"][*a])).map(|a| section(lo, hi, a, component(centre, a))).collect();
    }
    out.making = GENERATIVE_MAKING.get(*making).and_then(|(_, key, value)| {
        let a = ["x", "y", "z"].iter().position(|x| value.ends_with(x))?;
        match *key {
            "print" => {
                let upwards = !value.starts_with('-');
                let base = if upwards { component(lo, a) } else { component(hi, a) };
                let layer = section(lo, hi, a, base);
                let from = (layer[0] + layer[2]) * 0.5;
                let up = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)][a] * if upwards { 1.0 } else { -1.0 };
                Some(Making::Print { from, up, height: component(size, a), layer })
            }
            "extrude" => Some(Making::Cut { ends: [section(lo, hi, a, component(lo, a)), section(lo, hi, a, component(hi, a))] }),
            _ => None,
        }
    });
    let volume = size.x * size.y * size.z;
    if let Some((_, cells)) = GENERATIVE_DETAIL.get(*detail).filter(|_| volume > 1e-9) {
        let cell = (volume / cells).cbrt().max(0.05);
        out.cell = Some(cell);
        out.member = Some(number(wall).unwrap_or((2.5 * cell).max(0.015 * size.x.max(size.y).max(size.z))));
    }
    Some(out)
}

/// A colour seen through: `alpha` of 255.
fn faint(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

fn arrow(painter: &egui::Painter, a: Pos2, b: Pos2, width: f32, col: Color32) {
    let v = b - a;
    let l = v.length();
    if l < 1.0 {
        return;
    }
    let u = v / l;
    let n = vec2(-u.y, u.x);
    let head = (5.0 * width).min(l * 0.6);
    painter.line_segment([a, b - u * head * 0.8], Stroke::new(width, col));
    painter.add(Shape::convex_polygon(vec![b, b - u * head + n * head * 0.45, b - u * head - n * head * 0.45], col, Stroke::NONE));
}

fn outline(painter: &egui::Painter, proj: &Proj, corners: &[Vec3; 4], fill: Color32, stroke: Stroke) {
    let pts: Vec<Pos2> = corners.iter().filter_map(|c| proj.to_screen(*c)).collect();
    if pts.len() == 4 {
        painter.extend(projected_polygon(&pts, fill, stroke));
    }
}

fn wire_box(painter: &egui::Painter, proj: &Proj, lo: Vec3, hi: Vec3, stroke: Stroke) {
    let corner =
        |i: usize| Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z });
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            if i & bit == 0
                && let (Some(a), Some(b)) = (proj.to_screen(corner(i)), proj.to_screen(corner(i | bit)))
            {
                painter.line_segment([a, b], stroke);
            }
        }
    }
}

fn tint(painter: &egui::Painter, proj: &Proj, tris: &[[Vec3; 3]], fill: Color32) {
    if tris.len() > MOST_TRIANGLES {
        return;
    }
    let mut mesh = egui::Mesh::default();
    for t in tris {
        let [Some(a), Some(b), Some(c)] = t.map(|p| proj.to_screen(p)) else { continue };
        let at = mesh.vertices.len() as u32;
        for p in [a, b, c] {
            mesh.colored_vertex(p, fill);
        }
        mesh.add_triangle(at, at + 1, at + 2);
    }
    painter.add(Shape::mesh(mesh));
}

/// Draw the window's settings over the model.
pub(crate) fn overlay(app: &SolveApp, painter: &egui::Painter, proj: &Proj, d: &Dialog) {
    let Some(p) = preview(app, d) else { return };
    let t = Tokens::get();
    let font = FontId::proportional(t.font_small);
    // Millimetres per screen point, for marks that keep their size on screen.
    let mm = 2.0 * app.cam.half_height() / f64::from(proj.rect.height().max(1.0));
    let colours = [t.warning, t.sketch_accent];
    let units = ["N", "Nm", "MPa"];

    if let Some((lo, hi)) = p.space {
        wire_box(painter, proj, lo, hi, Stroke::new(1.0, faint(t.manipulator, 150)));
    }
    for (lo, hi) in &p.keep_out {
        wire_box(painter, proj, *lo, *hi, Stroke::new(1.0, faint(t.error, 190)));
    }
    for corners in &p.mirrors {
        outline(painter, proj, corners, faint(t.construction_plane, 40), Stroke::new(1.0, t.construction_plane));
    }
    match &p.making {
        Some(Making::Print { from, up, height, layer }) => {
            // The first layers, and the way the print grows.
            for i in 0..4 {
                let lift = *up * (f64::from(i) * 0.04 * *height);
                outline(painter, proj, &layer.map(|c| c + lift), Color32::TRANSPARENT, Stroke::new(1.0, faint(t.axis_z, 200)));
            }
            if let (Some(a), Some(b)) = (proj.to_screen(*from), proj.to_screen(*from + *up * *height)) {
                arrow(painter, a, b, 2.0, t.axis_z);
            }
        }
        Some(Making::Cut { ends }) => {
            for end in ends {
                outline(painter, proj, end, Color32::TRANSPARENT, Stroke::new(1.0, faint(t.axis_z, 200)));
            }
            let middles = ends.map(|e| (e[0] + e[2]) * 0.5);
            if let (Some(a), Some(b)) = (proj.to_screen(middles[0]), proj.to_screen(middles[1])) {
                arrow(painter, a, b, 2.0, t.axis_z);
                arrow(painter, b, a, 2.0, t.axis_z);
            }
        }
        None => {}
    }

    // The held faces: tinted, with the mark of a fixed support (a bar on hatching).
    for (at, tris) in &p.held {
        tint(painter, proj, tris, faint(t.accent, 70));
        let Some(c) = proj.to_screen(*at) else { continue };
        painter.line_segment([c + vec2(-11.0, 0.0), c + vec2(11.0, 0.0)], Stroke::new(2.0, t.accent));
        for i in 0..5 {
            let x = -10.0 + 5.0 * i as f32;
            painter.line_segment([c + vec2(x, 0.0), c + vec2(x - 4.0, 6.0)], Stroke::new(1.5, t.accent));
        }
    }
    // A face that slides stands on rollers, one on a pin has the pin's ring.
    for (turns, at, tris) in &p.guided {
        tint(painter, proj, tris, faint(t.accent, 70));
        let Some(c) = proj.to_screen(*at) else { continue };
        if *turns {
            painter.circle_stroke(c, 8.0, Stroke::new(2.0, t.accent));
            painter.circle_filled(c, 2.5, t.accent);
        } else {
            painter.line_segment([c + vec2(-11.0, -3.0), c + vec2(11.0, -3.0)], Stroke::new(2.0, t.accent));
            for x in [-7.0, 0.0, 7.0] {
                painter.circle_stroke(c + vec2(x, 1.5), 3.0, Stroke::new(1.5, t.accent));
            }
            painter.line_segment([c + vec2(-11.0, 6.0), c + vec2(11.0, 6.0)], Stroke::new(1.0, t.accent));
        }
    }
    for (load, tris) in &p.loaded {
        tint(painter, proj, tris, faint(colours[*load % 2], 60));
    }
    // Each load: a force as an arrow that ends on the face it presses on, or starts at the
    // face it pulls; a moment as an arrow with two heads along the axis it turns about; a
    // pressure as a row of arrows onto the face.
    let length = 70.0 * mm;
    let mut labelled = [false; crate::dialogs::GENERATIVE_LOADS];
    for f in &p.forces {
        let col = colours[f.load % 2];
        let long = if f.kind == 2 { 0.6 * length } else { length };
        let (from, to) = if f.onto { (f.at - f.dir * long, f.at) } else { (f.at, f.at + f.dir * long) };
        let (Some(a), Some(b)) = (proj.to_screen(from), proj.to_screen(to)) else { continue };
        arrow(painter, a, b, 3.0, col);
        let along = (b - a).normalized();
        match f.kind {
            1 => arrow(painter, a, b - along * 12.0, 3.0, col),
            2 => {
                let across = vec2(-along.y, along.x) * 14.0;
                for side in [-2.0, -1.0, 1.0, 2.0] {
                    arrow(painter, a + across * side, b + across * side, 2.0, col);
                }
            }
            _ => {}
        }
        // The load's size once per load, at the far end of its first arrow.
        if let Some(seen) = labelled.get_mut(f.load).filter(|seen| !**seen) {
            *seen = true;
            let far = if f.onto { a } else { b };
            let text = format!("{} {}", round(f.amount), units[f.kind.min(2)]);
            painter.text(far + vec2(8.0, -8.0), Align2::LEFT_BOTTOM, text, font.clone(), col);
        }
    }

    // A cell and the thinnest member in their true size, at the design space's first corner.
    if let (Some((lo, _)), Some(cell), Some(member)) = (p.space, p.cell, p.member) {
        let stroke = Stroke::new(1.0, faint(t.manipulator, 200));
        // Fewer cells drawn when they are too small to tell apart.
        let step = if cell / mm < 4.0 { 0.0 } else { cell };
        if step > 0.0 {
            for i in 0..=4 {
                let along = step * f64::from(i);
                for (a, b) in
                    [(Vec3::new(along, 0.0, 0.0), Vec3::new(along, 4.0 * step, 0.0)), (Vec3::new(0.0, along, 0.0), Vec3::new(4.0 * step, along, 0.0))]
                {
                    if let (Some(a), Some(b)) = (proj.to_screen(lo + a), proj.to_screen(lo + b)) {
                        painter.line_segment([a, b], stroke);
                    }
                }
            }
        }
        if let Some(c) = proj.to_screen(lo) {
            let radius = (0.5 * member / mm) as f32;
            let centre = c + vec2(-radius - 10.0, radius + 10.0);
            painter.circle(centre, radius, faint(t.manipulator, 60), stroke);
            let text = format!("{}  {} mm\n{}  {} mm", tr("Cell"), round(cell), tr("Thinnest member"), round(member));
            painter.text(centre + vec2(-radius - 6.0, 0.0), Align2::RIGHT_CENTER, text, font, t.text);
        }
    }
}

/// A length with at most two decimals, as short as it reads.
fn round(v: f64) -> String {
    let text = format!("{v:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
#[path = "generative_preview_tests.rs"]
mod tests;
