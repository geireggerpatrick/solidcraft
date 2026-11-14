//! Runs a small study and writes the structure as an STL file, to look at a change quickly:
//! `cargo run --release -p solidcraft-generative-gpu --example study -- shelf out.stl [voxels] [style] [newtons] [wall] [cpu|gpu|any]`.
//! `gpu` (the default) computes on a hardware adapter when there is one, `any` also on a software
//! adapter, `cpu` never on an adapter.

use std::io::Write;

use solidcraft_generative::Material;
use solidcraft_generative::study::{Study, run};
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
    let (held, load, obstacles): (Vec<[V3; 3]>, Vec<[V3; 3]>, Vec<Mesh>) = match case {
        // Two plates facing each other, 115 mm apart.
        "plates" => (
            quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 40.0], [5.0, 0.0, 40.0]),
            quad([120.0, 10.0, 0.0], [120.0, 10.0, 40.0], [120.0, 50.0, 40.0], [120.0, 50.0, 0.0]),
            vec![box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 40.0]), box_mesh([120.0, 10.0, 0.0], [125.0, 50.0, 40.0])],
        ),
        // A long beam and a small plate 180 mm in front of its side.
        "beam" => (
            quad([20.0, 0.0, 0.0], [20.0, 400.0, 0.0], [20.0, 400.0, 40.0], [20.0, 0.0, 40.0]),
            quad([200.0, 250.0, -10.0], [200.0, 250.0, 30.0], [200.0, 330.0, 30.0], [200.0, 330.0, -10.0]),
            vec![box_mesh([0.0, 0.0, 0.0], [20.0, 400.0, 40.0]), box_mesh([200.0, 250.0, -10.0], [210.0, 330.0, 30.0])],
        ),
        // A wall plate and a shelf plate in front of it.
        _ => (
            quad([5.0, 0.0, 0.0], [5.0, 60.0, 0.0], [5.0, 60.0, 80.0], [5.0, 0.0, 80.0]),
            quad([30.0, 5.0, 80.0], [30.0, 55.0, 80.0], [100.0, 55.0, 80.0], [100.0, 5.0, 80.0]),
            vec![box_mesh([0.0, 0.0, 0.0], [5.0, 60.0, 80.0]), box_mesh([30.0, 5.0, 80.0], [100.0, 55.0, 85.0])],
        ),
    };
    let study = Study {
        held: vec![held],
        load: vec![load],
        force: [0.0, 0.0, -newtons],
        space: None,
        obstacles,
        margin: 0.0,
        voxels,
        settings: Settings {
            style,
            material: Material { strength: 35.0, modulus: 1900.0, density: 1.27 },
            safety: 2.0,
            volume: None,
            min_wall: wall,
            max_deflection: 0.0,
            accelerate: true,
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
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    f.write_all(&[0u8; 80])?;
    f.write_all(&(r.mesh.tri.len() as u32).to_le_bytes())?;
    for t in &r.mesh.tri {
        f.write_all(&[0u8; 12])?;
        for v in t {
            for c in r.mesh.pos[*v] {
                f.write_all(&(c as f32).to_le_bytes())?;
            }
        }
        f.write_all(&[0u8; 2])?;
    }
    Ok(())
}
