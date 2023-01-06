from neighborlist import neighbor_list
import numpy as np
import ase
import ase.neighborlist


def compare_with_ase(pos, cutoff, cell, pbc):
    i1, j1, d1, D1, S1 = neighbor_list(pos, cutoff, cell, pbc, False)

    i = np.lexsort((j1, i1, S1[:, 0], S1[:, 1], S1[:, 2]))
    i1 = i1[i]
    j1 = j1[i]
    d1 = d1[i]
    D1 = D1[i]
    S1 = S1[i]

    a = ase.Atoms(positions=pos, cell=cell, pbc=pbc)
    i2, j2, d2, D2, S2 = ase.neighborlist.neighbor_list("ijdDS", a, cutoff, False)

    i = np.lexsort((j2, i2, S2[:, 0], S2[:, 1], S2[:, 2]))
    i2 = i2[i]
    j2 = j2[i]
    d2 = d2[i]
    D2 = D2[i]
    S2 = S2[i]

    assert len(i1) == len(i2)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(j1, j2)
    np.testing.assert_allclose(d1, d2)
    np.testing.assert_allclose(D1, D2)
    np.testing.assert_allclose(S1, S2)


def test_with_no_pbc():
    compare_with_ase(
        pos=np.random.uniform(-4.0, 3.0, (100, 3)),
        cutoff=2.0,
        cell=np.eye(3),
        pbc=np.array([False, False, False]),
    )


def test_with_pbc_thin_skin():
    compare_with_ase(
        pos=np.random.uniform(0.0, 1.0, (100, 3)),
        cutoff=0.2,
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )


def test_with_pbc_thick_skin():
    compare_with_ase(
        pos=np.random.uniform(0.0, 1.0, (20, 3)),
        cutoff=1.1,
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )


def test_with_nontrivial_cell():
    compare_with_ase(
        pos=np.random.uniform(0.0, 1.0, (20, 3)),
        cutoff=1.1,
        cell=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        pbc=np.array([True, True, True]),
    )
