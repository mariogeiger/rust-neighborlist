extern crate nalgebra as na;

use na::{Matrix3, Vector3};
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{pymodule, types::PyModule, PyErr, PyResult, Python};
use std::collections::{HashMap, HashSet};

/// Bounding box in x space given a linear transformation
///
/// Given:
/// y = A x, y_lo and y_hi
///
/// Finds:
/// x_lo <= x < x_hi
///
/// Such that:
/// the set {A x for x in [x_lo, x_hi)} overlaps with the set [y_lo, y_hi)
/// with x_lo and x_hi being integers.
fn linear_bounding_box(
    inverse_matrix: Matrix3<f64>,
    y_lo: Vector3<f64>,
    y_hi: Vector3<f64>,
) -> (Vector3<i32>, Vector3<i32>) {
    let mut x_lo: Vector3<f64> = Vector3::zeros();
    let mut x_hi: Vector3<f64> = Vector3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            if inverse_matrix[(i, j)] < 0.0 {
                x_lo[i] += inverse_matrix[(i, j)] * y_hi[j];
                x_hi[i] += inverse_matrix[(i, j)] * y_lo[j];
            } else {
                x_lo[i] += inverse_matrix[(i, j)] * y_lo[j];
                x_hi[i] += inverse_matrix[(i, j)] * y_hi[j];
            }
        }
    }

    x_lo = x_lo.map(|x| x.floor());
    x_hi = x_hi.map(|x| x.ceil());

    let us_lo: Vector3<i32> = x_lo.map(|x| x as i32);
    let us_hi: Vector3<i32> = x_hi.map(|x| x as i32);

    (us_lo, us_hi)
}

fn foo(
    i: usize,
    pos: Vector3<f64>,
    cell: Matrix3<f64>,
    us_lo: Vector3<i32>,
    us_hi: Vector3<i32>,
    pbc: Vector3<bool>,
    box_size: f64,
    boxes: &mut HashMap<Vector3<i32>, HashSet<(usize, Vector3<i32>)>>,
) {
    for u0 in us_lo[0]..us_hi[0] {
        if pbc[0] == false && (u0 != 0) {
            continue;
        }
        for u1 in us_lo[1]..us_hi[1] {
            if pbc[1] == false && (u1 != 0) {
                continue;
            }
            for u2 in us_lo[2]..us_hi[2] {
                if pbc[2] == false && (u2 != 0) {
                    continue;
                }

                let us: Vector3<i32> = Vector3::new(u0, u1, u2);
                if us != Vector3::zeros() {
                    let shifted_pos = pos + cell.transpose() * us.map(|x| x as f64);
                    let shifted_key = (shifted_pos / box_size).map(|x| x as i32);
                    boxes
                        .entry(shifted_key)
                        .or_insert(HashSet::new())
                        .insert((i, us));
                }
            }
        }
    }
}

fn neighbor_list(
    positions: ArrayView2<'_, f64>,
    cutoff: f64,
    cell: ArrayView2<'_, f64>,
    pbc: ArrayView1<'_, bool>,
    self_interaction: bool,
) -> Result<
    (
        Array1<i32>,
        Array1<i32>,
        Array1<f64>,
        Array2<f64>,
        Array2<i32>,
    ),
    String,
> {
    let n = positions.shape()[0];
    let box_size = 1.001 * cutoff;
    let mut boxes: HashMap<Vector3<i32>, HashSet<(usize, Vector3<i32>)>> = HashMap::new();
    let mut bb_lo = Vector3::new(std::f64::MAX, std::f64::MAX, std::f64::MAX);
    let mut bb_hi = Vector3::new(std::f64::MIN, std::f64::MIN, std::f64::MIN);

    let cell: Matrix3<f64> = Matrix3::new(
        cell[(0, 0)],
        cell[(0, 1)],
        cell[(0, 2)],
        cell[(1, 0)],
        cell[(1, 1)],
        cell[(1, 2)],
        cell[(2, 0)],
        cell[(2, 1)],
        cell[(2, 2)],
    );
    let pbc: Vector3<bool> = Vector3::new(pbc[0], pbc[1], pbc[2]);

    for i in 0..n {
        let pos: Vector3<f64> =
            Vector3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]);

        let key: Vector3<i32> = (pos / box_size).map(|x| x as i32);
        boxes
            .entry(key)
            .or_insert(HashSet::new())
            .insert((i, Vector3::zeros()));

        bb_lo = bb_lo.zip_map(&pos, |x, y| x.min(y));
        bb_hi = bb_hi.zip_map(&pos, |x, y| x.max(y));
    }

    if let Some(cell_inv) = cell.try_inverse() {
        let approx_num_shifts =
            (bb_hi - bb_lo).map(|x| x + box_size).product() / cell.determinant();

        if approx_num_shifts > boxes.len() as f64 {
            // Less boxes than possible shifts

            for i in 0..n {
                let pos: Vector3<f64> =
                    Vector3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]);

                for (&key, _) in boxes.clone().iter() {
                    let (us_lo, us_hi) = {
                        let key: Vector3<f64> = key.map(|x| x as f64);
                        let b_lo: Vector3<f64> = key.map(|x| x - 1.0) * box_size - pos;
                        let b_hi: Vector3<f64> = key.map(|x| x + 2.0) * box_size - pos;
                        linear_bounding_box(cell_inv.transpose(), b_lo, b_hi)
                    };

                    foo(i, pos, cell, us_lo, us_hi, pbc, box_size, &mut boxes);
                }
            }
        } else {
            // less shifts than possible boxes: iterate over shifts

            // look for all cell that intersect with the bounding box
            let (us_lo, us_hi) = {
                let b_lo: Vector3<f64> = bb_lo.map(|x| x - box_size);
                let b_hi: Vector3<f64> = bb_hi.map(|x| x + box_size);
                println!("b = {:?}..{:?}", b_lo, b_hi);
                linear_bounding_box(cell_inv.transpose(), b_lo, b_hi)
            };
            println!("us = {:?}..{:?}", us_lo, us_hi);

            for i in 0..n {
                let pos: Vector3<f64> =
                    Vector3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]);

                foo(i, pos, cell, us_lo, us_hi, pbc, box_size, &mut boxes);
            }
        }
    } else {
        if pbc[0] || pbc[1] || pbc[2] {
            return Err(String::from(
                "Cannot use periodic boundary conditions with cell of determinant 0.",
            ));
        }
    }

    let mut src: Vec<i32> = Vec::new();
    let mut dst: Vec<i32> = Vec::new();
    let mut dist: Vec<f64> = Vec::new();
    let mut rel: Vec<[f64; 3]> = Vec::new();
    let mut shifts: Vec<[i32; 3]> = Vec::new();

    for (&key, value) in boxes.iter() {
        for &(i, i_us) in value {
            if i_us.map(|x| x * x).sum() != 0 {
                continue;
            }

            let i_pos = Vector3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]);
            for sx in -1..2 {
                for sy in -1..2 {
                    for sz in -1..2 {
                        let skey = Vector3::new(key[0] + sx, key[1] + sy, key[2] + sz);
                        if let Some(svalue) = boxes.get(&skey) {
                            for &(j, j_us) in svalue {
                                let j_pos = Vector3::new(
                                    positions[(j, 0)],
                                    positions[(j, 1)],
                                    positions[(j, 2)],
                                );
                                let delta =
                                    (j_pos + cell.transpose() * j_us.map(|x| x as f64)) - i_pos;
                                let r2 = delta.norm_squared();
                                if r2 < cutoff * cutoff {
                                    if self_interaction || i != j || j_us != Vector3::zeros() {
                                        src.push(i as i32);
                                        dst.push(j as i32);
                                        dist.push(r2.sqrt());
                                        rel.push([delta[0], delta[1], delta[2]]);
                                        shifts.push([j_us[0], j_us[1], j_us[2]]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((
        Array1::from(src),
        Array1::from(dst),
        Array1::from(dist),
        Array2::from(rel),
        Array2::from(shifts),
    ))
}

#[pymodule]
fn neighborlist(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// neighbor_list(positions, cutoff, /, self_interaction)
    /// --
    ///
    /// Computes the neighbor list of a set of points.
    #[pyfn(m)]
    #[pyo3(name = "neighbor_list")]
    fn neighbor_list_py<'py>(
        py: Python<'py>,
        positions: PyReadonlyArray2<'_, f64>,
        cutoff: f64,
        cell: PyReadonlyArray2<'_, f64>,
        pbc: PyReadonlyArray1<'_, bool>,
        self_interaction: bool,
    ) -> PyResult<(
        &'py PyArray1<i32>,
        &'py PyArray1<i32>,
        &'py PyArray1<f64>,
        &'py PyArray2<f64>,
        &'py PyArray2<i32>,
    )> {
        let positions = positions.as_array();
        let cell = cell.as_array();
        let pbc = pbc.as_array();

        match neighbor_list(
            positions.view(),
            cutoff,
            cell.view(),
            pbc.view(),
            self_interaction,
        ) {
            Ok((src, dst, dist, rel, shifts)) => Ok((
                src.into_pyarray(py),
                dst.into_pyarray(py),
                dist.into_pyarray(py),
                rel.into_pyarray(py),
                shifts.into_pyarray(py),
            )),
            Err(message) => Err(PyErr::new::<PyValueError, _>(message)),
        }
    }

    Ok(())
}
