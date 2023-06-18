use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod boxes;

fn neighbor_list_ijdd(
    positions: ArrayView2<f64>,
    cutoff: f64,
    self_interaction: bool,
) -> (Array1<i32>, Array1<i32>, Array1<f64>, Array2<f64>) {
    let n = positions.shape()[0];
    let mut boxes = boxes::Boxes::new(cutoff);

    for i in 0..n {
        let x = positions[(i, 0)];
        let y = positions[(i, 1)];
        let z = positions[(i, 2)];

        boxes.insert(i, x, y, z);
    }

    let mut src: Vec<i32> = Vec::new();
    let mut dst: Vec<i32> = Vec::new();
    let mut dist: Vec<f64> = Vec::new();
    let mut rel: Vec<[f64; 3]> = Vec::new();

    for (&key, value) in boxes.iter() {
        for &j in boxes.iter_neighbors(&key) {
            for &i in value {
                let dx = positions[(j, 0)] - positions[(i, 0)];
                let dy = positions[(j, 1)] - positions[(i, 1)];
                let dz = positions[(j, 2)] - positions[(i, 2)];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < cutoff * cutoff {
                    if self_interaction || i != j {
                        src.push(i as i32);
                        dst.push(j as i32);
                        dist.push(r2.sqrt());
                        rel.push([dx, dy, dz]);
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
    )
}

fn neighbor_list_ij(
    positions: ArrayView2<f64>,
    cutoff: f64,
    self_interaction: bool,
) -> (Array1<i32>, Array1<i32>) {
    let n = positions.shape()[0];
    let mut boxes = boxes::Boxes::new(cutoff);

    for i in 0..n {
        let x = positions[(i, 0)];
        let y = positions[(i, 1)];
        let z = positions[(i, 2)];

        boxes.insert(i, x, y, z);
    }

    let mut src: Vec<i32> = Vec::new();
    let mut dst: Vec<i32> = Vec::new();

    for (&key, value) in boxes.iter() {
        for &j in boxes.iter_neighbors(&key) {
            for &i in value {
                let dx = positions[(j, 0)] - positions[(i, 0)];
                let dy = positions[(j, 1)] - positions[(i, 1)];
                let dz = positions[(j, 2)] - positions[(i, 2)];
                if dx * dx + dy * dy + dz * dz < cutoff * cutoff {
                    if self_interaction || i != j {
                        src.push(i as i32);
                        dst.push(j as i32);
                    }
                }
            }
        }
    }

    (Array1::from(src), Array1::from(dst))
}

#[pymodule]
fn neighborlist(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// neighbor_list_ijdD(positions, cutoff, /, self_interaction)
    /// --
    ///
    /// Computes the neighbor list of a set of points.
    #[pyfn(m)]
    #[pyo3(name = "neighbor_list_ijdD")]
    fn neighbor_list_ijdd_py<'py>(
        py: Python<'py>,
        positions: PyReadonlyArray2<f64>,
        cutoff: f64,
        self_interaction: bool,
    ) -> (
        &'py PyArray1<i32>,
        &'py PyArray1<i32>,
        &'py PyArray1<f64>,
        &'py PyArray2<f64>,
    ) {
        let positions: ArrayView2<f64> = positions.as_array();
        let (src, dst, dist, rel) = neighbor_list_ijdd(positions, cutoff, self_interaction);
        (
            src.into_pyarray(py),
            dst.into_pyarray(py),
            dist.into_pyarray(py),
            rel.into_pyarray(py),
        )
    }

    /// neighbor_list_ij(positions, cutoff, /, self_interaction)
    /// --
    ///
    /// Computes the neighbor list of a set of points.
    #[pyfn(m)]
    #[pyo3(name = "neighbor_list_ij")]
    fn neighbor_list_ij_py<'py>(
        py: Python<'py>,
        positions: PyReadonlyArray2<f64>,
        cutoff: f64,
        self_interaction: bool,
    ) -> (&'py PyArray1<i32>, &'py PyArray1<i32>) {
        let positions: ArrayView2<f64> = positions.as_array();
        let (src, dst) = neighbor_list_ij(positions, cutoff, self_interaction);
        (src.into_pyarray(py), dst.into_pyarray(py))
    }

    Ok(())
}
