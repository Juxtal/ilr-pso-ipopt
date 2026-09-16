import numpy as np
import pytest

from ilr_pso_ipopt.core import build_helmert_basis, ilr_inverse, ilr_transform


def test_helmert_basis_is_orthonormal():
    for dimension in (2, 3, 7):
        basis = build_helmert_basis(dimension)
        assert basis.shape == (dimension, dimension - 1)
        # columns are orthonormal
        assert np.allclose(basis.T @ basis, np.eye(dimension - 1), atol=1e-12)
        # every column is centred, so the inverse map stays on the simplex
        assert np.allclose(basis.sum(axis=0), 0.0, atol=1e-12)


def test_helmert_basis_rejects_scalar_allocation():
    with pytest.raises(ValueError):
        build_helmert_basis(1)


@pytest.mark.parametrize("total", [1.0, 10.0, 137.5])
def test_ilr_roundtrip_preserves_allocation(total):
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = total * rng.dirichlet(np.ones(6))
        recovered = ilr_inverse(ilr_transform(z), total)
        assert np.allclose(recovered, z, rtol=1e-9, atol=1e-12)


def test_ilr_inverse_always_lands_on_the_simplex():
    rng = np.random.default_rng(1)
    total = 4.0
    # deliberately extreme coordinates, including beyond the default ilr_bound
    for y in rng.normal(0.0, 20.0, size=(50, 5)):
        z = ilr_inverse(y, total)
        assert np.all(np.isfinite(z))
        assert np.all(z >= 0.0)
        assert np.isclose(z.sum(), total, rtol=0.0, atol=1e-10)


def test_ilr_transform_is_scale_invariant():
    # ILR reads a composition, so only the ratios matter, not the total
    z = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(ilr_transform(z), ilr_transform(100.0 * z))


def test_ilr_transform_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ilr_transform(np.zeros(4))
    with pytest.raises(ValueError):
        ilr_transform(np.array([1.0, np.nan, 2.0]))
