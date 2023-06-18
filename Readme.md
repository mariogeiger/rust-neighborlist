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