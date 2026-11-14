use super::*;
use solidcraft_generative::voxel::unit_stiffness;

/// `K x` on the whole grid in double precision, the plain way, to check the GPU against.
fn apply(n: [usize; 3], ke: &[[f64; 24]; 24], stiff: &[f32], fixed: &[bool], x: &[f64]) -> Vec<f64> {
    let node = |i: usize, j: usize, k: usize| i + (n[0] + 1) * (j + (n[1] + 1) * k);
    let corner = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
    let mut y = vec![0.0; x.len()];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let s = f64::from(stiff[i + n[0] * (j + n[1] * k)]);
                let at: Vec<usize> = corner.iter().map(|c| node(i + c[0], j + c[1], k + c[2])).collect();
                for a in 0..24 {
                    let mut acc = 0.0;
                    for b in 0..24 {
                        acc += ke[a][b] * x[3 * at[b / 3] + b % 3];
                    }
                    y[3 * at[a / 3] + a % 3] += s * acc;
                }
            }
        }
    }
    for (m, held) in fixed.iter().enumerate() {
        if *held {
            for c in 0..3 {
                y[3 * m + c] = x[3 * m + c];
            }
        }
    }
    y
}

/// A beam held at one end and pushed down at the other, with a soft band through the middle:
/// the solver's result leaves a residual below the tolerance asked for, checked in double
/// precision on the CPU. Skipped on a machine without any adapter.
#[test]
fn solves_a_beam_to_the_tolerance() {
    let Ok(mut solver) = GpuSolver::open(solidcraft_viewport_gpu::select::Policy::Auto) else {
        eprintln!("no adapter: skipped");
        return;
    };
    let n = [20usize, 9, 7];
    let ke = unit_stiffness();
    let nodes = nodes_of(n);
    let mut stiff = vec![1.0f32; n[0] * n[1] * n[2]];
    for k in 0..n[2] {
        for j in 3..6 {
            for i in 5..15 {
                stiff[i + n[0] * (j + n[1] * k)] = 1e-3;
            }
        }
    }
    let node = |i: usize, j: usize, k: usize| i + (n[0] + 1) * (j + (n[1] + 1) * k);
    let mut fixed = vec![false; nodes];
    let mut rhs = vec![0.0f32; 3 * nodes];
    for k in 0..=n[2] {
        for j in 0..=n[1] {
            fixed[node(0, j, k)] = true;
            rhs[3 * node(n[0], j, k) + 2] = -1.0;
        }
    }
    let job = GridJob { n, ke: &ke, stiff: &stiff, fixed: &fixed, rhs: &rhs, tol: 1e-3, max_iter: 200 };
    let (d, iterations) = solver.solve(&job).unwrap_or_else(|e| panic!("{e}"));
    let x: Vec<f64> = d.iter().map(|v| f64::from(*v)).collect();
    let y = apply(n, &ke, &stiff, &fixed, &x);
    let b: Vec<f64> = rhs.iter().map(|v| f64::from(*v)).collect();
    let res = y.iter().zip(&b).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    let norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    eprintln!("{}: {iterations} iterations, residual {:.2e}", solver.name(), res / norm);
    assert!(res / norm < 5e-3, "residual {} after {iterations} iterations", res / norm);
    assert!(iterations < 120, "{iterations} iterations");
    // The free end moves down.
    assert!(x[3 * node(n[0], 4, 3) + 2] < 0.0);
}
