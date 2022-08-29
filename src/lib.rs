use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::collections::HashMap;

#[pymodule]
fn rust_neighborlist(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn neighbor_list(
        positions: ArrayView2<'_, f64>,
        cutoff: f64,
        self_interaction: bool,
    ) -> (Array1<i32>, Array1<i32>, Array1<f64>, Array2<f64>) {
        let n = positions.shape()[0];
        let box_size = 1.01 * cutoff;
        let mut boxes = HashMap::new();

        for i in 0..n {
            let x = positions[(i, 0)];
            let y = positions[(i, 1)];
            let z = positions[(i, 2)];

            let key = (
                (x / box_size) as i32,
                (y / box_size) as i32,
                (z / box_size) as i32,
            );
            if !boxes.contains_key(&key) {
                boxes.insert(key, Vec::new());
            }
            boxes.get_mut(&key).unwrap().push(i);
        }

        let mut src: Vec<i32> = Vec::new();
        let mut dst: Vec<i32> = Vec::new();
        let mut dist: Vec<f64> = Vec::new();
        let mut rel: Vec<[f64; 3]> = Vec::new();

        for (&key, value) in boxes.iter() {
            for &i in value {
                let ix = positions[(i, 0)];
                let iy = positions[(i, 1)];
                let iz = positions[(i, 2)];
                for sx in -1..2 {
                    for sy in -1..2 {
                        for sz in -1..2 {
                            let skey = (key.0 + sx, key.1 + sy, key.2 + sz);
                            if let Some(svalue) = boxes.get(&skey) {
                                for &j in svalue {
                                    let jx = positions[(j, 0)];
                                    let jy = positions[(j, 1)];
                                    let jz = positions[(j, 2)];
                                    let dx = jx - ix;
                                    let dy = jy - iy;
                                    let dz = jz - iz;
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

    #[pyfn(m)]
    #[pyo3(name = "neighbor_list")]
    fn neighbor_list_py<'py>(
        py: Python<'py>,
        positions: PyReadonlyArray2<'_, f64>,
        cutoff: f64,
        self_interaction: bool,
    ) -> (
        &'py PyArray1<i32>,
        &'py PyArray1<i32>,
        &'py PyArray1<f64>,
        &'py PyArray2<f64>,
    ) {
        let positions = positions.as_array();
        let (src, dst, dist, rel) = neighbor_list(positions.view(), cutoff, self_interaction);
        (
            src.into_pyarray(py),
            dst.into_pyarray(py),
            dist.into_pyarray(py),
            rel.into_pyarray(py),
        )
    }

    Ok(())
}
