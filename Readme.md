# Neighborlist implementation in rust with python interface

## Install
```
pip install rust-neighborlist
```

## Test
```python
import numpy as np
pos = np.random.uniform(-4.0, 3.0, (100, 3))
cutoff = 2.0


# Using ase
import ase
import ase.neighborlist

a = ase.Atoms(positions=pos)
i2, j2, d2, D2 = ase.neighborlist.neighbor_list("ijdD", a, cutoff, self_interaction=False)


# Using rust neighborlist
from neighborlist import neighbor_list

cell = np.eye(3)
pbc = np.array([False, False, False])
i1, j1, d1, D1 = neighbor_list(pos, cutoff, cell, pbc, self_interaction=False)
```
