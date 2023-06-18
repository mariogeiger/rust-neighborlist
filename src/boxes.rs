use std::collections::HashMap;

pub struct Boxes {
    box_size: f64,
    boxes: HashMap<[i32; 3], Vec<usize>>,
}

impl Boxes {
    pub fn new(cutoff: f64) -> Self {
        Self {
            box_size: 1.001 * cutoff,
            boxes: HashMap::new(),
        }
    }

    pub fn insert(&mut self, i: usize, x: f64, y: f64, z: f64) {
        let key = [
            (x / self.box_size).floor() as i32,
            (y / self.box_size).floor() as i32,
            (z / self.box_size).floor() as i32,
        ];
        self.boxes.entry(key).or_default().push(i);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[i32; 3], &Vec<usize>)> {
        self.boxes.iter()
    }

    pub fn iter_neighbors(&self, key: &[i32; 3]) -> impl Iterator<Item = &usize> {
        let mut neighbors = Vec::new();
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    let key = [key[0] + i, key[1] + j, key[2] + k];
                    if let Some(value) = self.boxes.get(&key) {
                        neighbors.extend(value);
                    }
                }
            }
        }
        neighbors.into_iter()
    }
}
