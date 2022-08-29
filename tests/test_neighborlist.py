from neighborlist import neighbor_list
import numpy as np
import ase
import ase.neighborlist


def test_neighbor_list():
    pos = np.random.uniform(-4.0, 3.0, (100, 3))
    cutoff = 2.0
    i1, j1, d1, D1 = neighbor_list(pos, cutoff, False)

    i = np.lexsort((j1, i1))
    i1 = i1[i]
    j1 = j1[i]
    d1 = d1[i]
    D1 = D1[i]

    a = ase.Atoms(positions=pos)
    i2, j2, d2, D2 = ase.neighborlist.neighbor_list("ijdD", a, cutoff, False)

    i = np.lexsort((j2, i2))
    i2 = i2[i]
    j2 = j2[i]
    d2 = d2[i]
    D2 = D2[i]

    assert len(i1) == len(i2)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(j1, j2)
    np.testing.assert_allclose(d1, d2)
    np.testing.assert_allclose(D1, D2)
