//! Runs a small study and writes the structure as an STL file, to look at a change quickly:
//! `cargo run --release -p solidcraft-generative-gpu --example study -- shelf out.stl [voxels] [style] [newtons] [wall] [cpu|gpu|any] [deflection] [mirror: xyz|-] [extrude: x|y|z|-] [keep: 0..1]`.
//! `gpu` (the default) computes on a hardware adapter when there is one, `any` also on a software
//! adapter, `cpu` never on an adapter.

use std::io::Write;

use solidcraft_generative::Material;
use solidcraft_generative::study::{Action, Study, run};
use solidcraft_generative::voxel::{Mesh, Settings, Style, V3, box_mesh};

fn quad(a: V3, b: V3, c: V3, d: V3) -> Vec<[V3; 3]> {
    vec![[a, b, c], [a, c, d]]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("shelf");
    let out = args.get(2).map(String::as_str).unwrap_or("study.stl");
    let voxels: f64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(40_000.0);
    let style = match args.get(4).map(String::as_str) {
        Some("tree") => Style::Tree,
        Some("lattice") => Style::Lattice,
        _ => Style::Organic,
    };
    let newtons: f64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(300.0);
    let wall: f64 = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(4.0);
    match args.get(7).map(String::as_str) {
        Some("cpu") => println!("computing on the CPU"),
        mode => {
            let policy =
                if mode == Some("any") { solidcraft_viewport_gpu::select::Policy::Auto } else { solidcraft_viewport_gpu::select::Policy::Hardware };
            match solidcraft_generative_gpu::GpuSolver::open(policy) {
                Ok(solver) => {
                    println!("computing on {}", solidcraft_generative::voxel::GridSolver::name(&solver));
                    solidcraft_generative::voxel::set_grid_solver(Some(Box::new(solver)));
                }
                Err(e) => println!("computing on the CPU: {e}"),
            }
        }
    }
    // A slanted round bar as a smooth field, to look at the surface alone.
    if case == "tube" {
        let g = solidcraft_generative::voxel::box_grid([0.0; 3], [40.0; 3], 1.0);
        let dir = [0.48, 0.6, 0.64];
        let field: Vec<f64> = (0..g.len())
            .map(|v| {
                let c = g.center(v).map(|x| x - 20.0);
                let along = c[0] * dir[0] + c[1] * dir[1] + c[2] * dir[2];
                let off = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - along * along).max(0.0).sqrt();
                (0.5 + (wall - off) / 3.0).clamp(0.0, 1.0) * if along.abs() < 14.0 { 1.0 } else { 0.0 }
            })
            .collect();
        let mesh = solidcraft_generative::voxel::surface(&g, &field, voxels.max(1.0) as usize, 40, &[], None, &[]);
        return write_stl(out, &mesh);
    }
    let deflection: f64 = args.get(8).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let mut force = [0.0, 0.0, -newtons];
    let mut more = Vec::new();
    let (held, load, obstacles): (Vec<Vec<[V3; 3]>>, Vec<Vec<[V3; 3]>>, Vec<Mesh>) = match case {
        // A plate held above five feet spread out below it, each pushed upwards.
        "bracket" => {
            force = [0.0, 0.0, newtons];
            let feet = [[-120.0, -40.0], [-80.0, 45.0], [0.0, 70.0], [80.0, 45.0], [120.0, -40.0]];
            let mut bodies = vec![box_mesh([-30.0, -25.0, 90.0], [30.0, 25.0, 98.0])];
            let mut load = Vec::new();
            for [x, y] in feet {
                bodies.push(box_mesh([x - 8.0, y - 8.0, 0.0], [x + 8.0, y + 8.0, 10.0]));
                load.push(quad([x - 8.0, y - 8.0, 10.0], [x + 8.0, y - 8.0, 10.0], [x + 8.0, y + 8.0, 10.0], [x - 8.0, y + 8.0, 10.0]));
            }
            (vec![quad([-30.0, -25.0, 90.0], [-30.0, 25.0, 90.0], [30.0, 25.0, 90.0], [30.0, -25.0, 90.0])], load, bodies)
        }
        // Two plates facing each other, 115 mm apart.
        "plates" => (
            vec![quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 40.0], [5.0, 0.0, 40.0])],
            vec![quad([120.0, 10.0, 0.0], [120.0, 10.0, 40.0], [120.0, 50.0, 40.0], [120.0, 50.0, 0.0])],
            vec![box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 40.0]), box_mesh([120.0, 10.0, 0.0], [125.0, 50.0, 40.0])],
        ),
        // A long beam and a small plate 180 mm in front of its side.
        "beam" => (
            vec![quad([20.0, 0.0, 0.0], [20.0, 400.0, 0.0], [20.0, 400.0, 40.0], [20.0, 0.0, 40.0])],
            vec![quad([200.0, 250.0, -10.0], [200.0, 250.0, 30.0], [200.0, 330.0, 30.0], [200.0, 330.0, -10.0])],
            vec![box_mesh([0.0, 0.0, 0.0], [20.0, 400.0, 40.0]), box_mesh([200.0, 250.0, -10.0], [210.0, 330.0, 30.0])],
        ),
        // A wall plate and a round pin in front of it, loaded on its round face.
        "pin" => {
            let ring = |y: f64| -> Vec<V3> {
                (0..32).map(|i| std::f64::consts::TAU * f64::from(i) / 32.0).map(|t| [70.0 + 6.0 * t.cos(), y, 50.0 + 6.0 * t.sin()]).collect()
            };
            let (a, b) = (ring(10.0), ring(50.0));
            let mut side = Vec::new();
            let mut pin = Mesh::default();
            for i in 0..32 {
                let j = (i + 1) % 32;
                side.extend(quad(a[j], a[i], b[i], b[j]));
                // The ends close the pin: fans around the middles of the two rings.
                side.iter().rev().take(2).for_each(|t| {
                    let at = pin.pos.len();
                    pin.pos.extend(*t);
                    pin.tri.push([at, at + 1, at + 2]);
                });
                for (middle, p, q) in [([70.0, 10.0, 50.0], a[i], a[j]), ([70.0, 50.0, 50.0], b[j], b[i])] {
                    let at = pin.pos.len();
                    pin.pos.extend([middle, p, q]);
                    pin.tri.push([at, at + 1, at + 2]);
                }
            }
            (
                vec![quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 80.0], [5.0, 0.0, 80.0])],
                vec![side],
                vec![box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 80.0]), pin],
            )
        }
        // The shelf, and a small plate beside it that is pushed sideways with half the force:
        // at the same time (`two`) or as a load case of its own (`cases`). With `edge`, the
        // plate is also pushed down on its top face, which meets the first in an edge.
        "two" | "cases" | "edge" => {
            more.push((
                vec![quad([60.0, 80.0, 30.0], [80.0, 80.0, 30.0], [80.0, 80.0, 50.0], [60.0, 80.0, 50.0])],
                [0.0, -0.5 * newtons, 0.0],
                usize::from(case == "cases"),
            ));
            if case == "edge" {
                more.push((
                    vec![quad([60.0, 80.0, 50.0], [80.0, 80.0, 50.0], [80.0, 85.0, 50.0], [60.0, 85.0, 50.0])],
                    [0.0, 0.0, -0.2 * newtons],
                    0,
                ));
            }
            (
                vec![quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 80.0], [5.0, 0.0, 80.0])],
                vec![quad([30.0, 5.0, 80.0], [30.0, 55.0, 80.0], [100.0, 55.0, 80.0], [100.0, 5.0, 80.0])],
                vec![
                    box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 80.0]),
                    box_mesh([30.0, 5.0, 80.0], [100.0, 55.0, 85.0]),
                    box_mesh([60.0, 80.0, 30.0], [80.0, 85.0, 50.0]),
                ],
            )
        }
        // A wall plate and a shelf plate in front of it.
        _ => (
            vec![quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 80.0], [5.0, 0.0, 80.0])],
            vec![quad([30.0, 5.0, 80.0], [30.0, 55.0, 80.0], [100.0, 55.0, 80.0], [100.0, 5.0, 80.0])],
            vec![box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 80.0]), box_mesh([30.0, 5.0, 80.0], [100.0, 55.0, 85.0])],
        ),
    };
    let study = Study {
        held,
        load,
        slide: Vec::new(),
        pin: Vec::new(),
        // STUDY_ACTION=moment turns the loaded faces about x instead of pushing them, with an
        // arm of 100 mm; STUDY_ACTION=pressure presses on them with the force's worth per 1000 mm2.
        action: match std::env::var("STUDY_ACTION").as_deref() {
            Ok("moment") => Action::Moment([100.0 * newtons, 0.0, 0.0]),
            Ok("pressure") => Action::Pressure(newtons / 1000.0),
            _ => Action::Force(force),
        },
        more: more.into_iter().map(|(faces, force, case)| (faces, Action::Force(force), case)).collect(),
        space: None,
        obstacles,
        avoid: Vec::new(),
        clearance: 0.0,
        margin: 0.0,
        voxels,
        settings: Settings {
            style,
            material: Material { strength: 35.0, modulus: 1900.0, density: 1.27 },
            safety: 2.0,
            volume: args.get(11).and_then(|v| v.parse().ok()),
            min_wall: wall,
            max_deflection: deflection,
            accelerate: true,
            symmetry: (*b"xyz").map(|axis| args.get(9).is_some_and(|a| a.bytes().any(|c| c == axis))),
            extrude: args.get(10).and_then(|a| ["x", "y", "z"].iter().position(|x| x == a)),
            print: args.get(12).and_then(|a| Some((["x", "y", "z"].iter().position(|x| a.ends_with(x))?, !a.starts_with('-')))),
        },
    };
    let t = std::time::Instant::now();
    let r = run(&study, &mut || true)?;
    let o = &r.outcome;
    println!(
        "cells {} cell {:.2} mm design {} solid {} iterations {} stress {:.1}/{:.1} MPa deflection {:.2} mm passed {} time {:.1} s",
        r.cells,
        r.cell,
        o.design_voxels,
        o.solid_voxels,
        o.iterations,
        o.stress,
        o.allowed,
        o.deflection,
        o.passed,
        t.elapsed().as_secs_f64()
    );
    println!("notes {:?}", o.note);
    write_stl(out, &r.mesh)
}

fn write_stl(out: &str, mesh: &Mesh) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    f.write_all(&[0u8; 80])?;
    f.write_all(&(mesh.tri.len() as u32).to_le_bytes())?;
    for t in &mesh.tri {
        f.write_all(&[0u8; 12])?;
        for v in t {
            for c in mesh.pos[*v] {
                f.write_all(&(c as f32).to_le_bytes())?;
            }
        }
        f.write_all(&[0u8; 2])?;
    }
    Ok(())
}
