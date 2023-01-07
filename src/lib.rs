extern crate nalgebra as na;

use na::{Matrix3, Vector3};
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::collections::{HashMap, HashSet};

fn neighbor_list(
    positions: ArrayView2<'_, f64>,
    cutoff: f64,
    cell: ArrayView2<'_, f64>,
    pbc: ArrayView1<'_, bool>,
    self_interaction: bool,
) -> (
    Array1<i32>,
    Array1<i32>,
    Array1<f64>,
    Array2<f64>,
    Array2<i32>,
) {
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
    let pinv = cell.try_inverse().unwrap();

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

    let approx_num_shifts = (bb_hi - bb_lo).map(|x| x + box_size).product() / cell.determinant();

    if approx_num_shifts as usize > boxes.len() {
        // Less boxes than possible shifts

        for i in 0..n {
            let pos: Vector3<f64> =
                Vector3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]);

            for (&key, _) in boxes.clone().iter() {
                // solve: (key - 1) * box_size - pos <= us @ cell < (key + 2) * box_size - pos
                let key: Vector3<f64> = key.map(|x| x as f64);
                let b_lo: Vector3<f64> = key.map(|x| x - 1.0) * box_size - pos;
                let b_hi: Vector3<f64> = key.map(|x| x + 2.0) * box_size - pos;

                let mut us_lo: Vector3<f64> = Vector3::zeros();
                let mut us_hi: Vector3<f64> = Vector3::zeros();

                for i in 0..3 {
                    for j in 0..3 {
                        if pinv[(i, j)] < 0.0 {
                            us_lo[i] += pinv[(i, j)] * b_hi[j];
                            us_hi[i] += pinv[(i, j)] * b_lo[j];
                        } else {
                            us_lo[i] += pinv[(i, j)] * b_lo[j];
                            us_hi[i] += pinv[(i, j)] * b_hi[j];
                        }
                    }
                }

                let us_lo: Vector3<i32> = us_lo.map(|x| x.ceil() as i32);
                let us_hi: Vector3<i32> = us_hi.map(|x| x.ceil() as i32);

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
        }
    } else {
        // less shifts than possible boxes: iterate over shifts

        let b_lo: Vector3<f64> = bb_lo.map(|x| x - box_size);
        let b_hi: Vector3<f64> = bb_hi.map(|x| x + box_size);

        let mut us_lo: Vector3<f64> = Vector3::zeros();
        let mut us_hi: Vector3<f64> = Vector3::zeros();

        for i in 0..3 {
            for j in 0..3 {
                if pinv[(i, j)] < 0.0 {
                    us_lo[i] += pinv[(i, j)] * b_hi[j];
                    us_hi[i] += pinv[(i, j)] * b_lo[j];
                } else {
                    us_lo[i] += pinv[(i, j)] * b_lo[j];
                    us_hi[i] += pinv[(i, j)] * b_hi[j];
                }
            }
        }

        let us_lo: Vector3<i32> = us_lo.map(|x| x.ceil() as i32);
        let us_hi: Vector3<i32> = us_hi.map(|x| x.ceil() as i32);

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
                        for i in 0..n {
                            let pos: Vector3<f64> = Vector3::new(
                                positions[(i, 0)],
                                positions[(i, 1)],
                                positions[(i, 2)],
                            );

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

    (
        Array1::from(src),
        Array1::from(dst),
        Array1::from(dist),
        Array2::from(rel),
        Array2::from(shifts),
    )
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
    ) -> (
        &'py PyArray1<i32>,
        &'py PyArray1<i32>,
        &'py PyArray1<f64>,
        &'py PyArray2<f64>,
        &'py PyArray2<i32>,
    ) {
        let positions = positions.as_array();
        let cell = cell.as_array();
        let pbc = pbc.as_array();

        let (src, dst, dist, rel, shifts) = neighbor_list(
            positions.view(),
            cutoff,
            cell.view(),
            pbc.view(),
            self_interaction,
        );
        (
            src.into_pyarray(py),
            dst.into_pyarray(py),
            dist.into_pyarray(py),
            rel.into_pyarray(py),
            shifts.into_pyarray(py),
        )
    }

    Ok(())
}
