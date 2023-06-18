use std::collections::HashMap;

pub struct Boxes {
    box_size: f64,
    boxes: HashMap<[i32; 3], Vec<usize>>,
}

impl Boxes {
    pub fn new(box_size: f64) -> Self {
        Self {
            box_size,
            boxes: HashMap::new(),
        }
    }

    pub fn insert(&mut self, i: usize, x: f64, y: f64, z: f64) {
        let key = [
            (x / self.box_size) as i32,
            (y / self.box_size) as i32,
            (z / self.box_size) as i32,
        ];
        self.boxes.entry(key).or_insert(Vec::new()).push(i);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[i32; 3], &Vec<usize>)> {
        self.boxes.iter()
    }

    pub fn iter_neighbors(&self, key: &[i32; 3]) -> impl Iterator<Item = &usize> {
        let mut neighbors = Vec::new();
        for sx in -1..2 {
            for sy in -1..2 {
                for sz in -1..2 {
                    let skey = [key[0] + sx, key[1] + sy, key[2] + sz];
                    if let Some(svalue) = self.boxes.get(&skey) {
                        neighbors.extend(svalue);
                    }
                }
            }
        }
        neighbors.into_iter()
    }
}
