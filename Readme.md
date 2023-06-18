# Neighborlist implementation in rust with python interface

## Only faster than [matscipy](https://github.com/libAtoms/matscipy/tree/master) in few cases
This rust implementation is only faster than matscipy at high density.
i.e. when there is a lot of points within the cutoff distance.

![image](https://github.com/mariogeiger/rust-neighborlist/assets/333780/d4cacfc3-d509-4b46-9ba2-efec7fe42867)

## Install
```
pip install rust-neighborlist
```

## Test
```python
import numpy as np
pos = np.random.uniform(-4.0, 3.0, (100, 3))
cutoff = 2.0


# Using matscipy.neighbours
from matscipy.neighbours import neighbour_list

i, j = neighbour_list('ij', positions=pos, cutoff=cutoff)


# Using rust neighborlist
from neighborlist import neighbor_list_ij

i, j = neighbor_list_ij(pos, cutoff, self_interaction=False)
```


## Install from source
```
git clone https://github.com/mariogeiger/rust-neighborlist.git
cd rust-neighborlist
pip install .
```

## Publish to pypi
```
maturin publish
```
