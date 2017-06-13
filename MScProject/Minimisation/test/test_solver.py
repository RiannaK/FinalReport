import numpy as np
from numpy.testing import assert_array_almost_equal as array_assert
from numpy.testing import assert_almost_equal as assert_almost
import pytest

from Minimisation.solver import SigmaSolver, DeterminantMinimiser


def test_solver_solve():
    # Arrange
    drift = np.array([[1, 2], [3, 4]])
    diffusion = -np.array([[7, 8], [9, 10]])

    expected11 = -0.1
    expected12 = 1.7
    expected21 = 1.9
    expected22 = -0.1
    expected = np.array([[expected11, expected12], [expected21, expected22]])

    sut = SigmaSolver()

    # Act
    sigma = sut.solve(drift, diffusion)

    # Assert
    array_assert(expected, sigma, 6)


def test_determinant_minimiser():
    # Arrange
    a, b, c = 4, 3, 5

    expected_det = 6.24859462836

    sut = DeterminantMinimiser(2)

    # Act
    det = sut.calculate_determinant(a, b, c)

    # Assert
    assert_almost(det, expected_det, 7, "determinants are not equal")

def test_minimiser():
    # Arrange
    sut = DeterminantMinimiser(2)

    # Act
    res = sut.minimise()

    # Assert
    print('Solution is: {0}'.format(res.x))
    print('Minimum determinant is: {0}'.format(res.fun))

def test_plot_landscape():
    # Arrange
    sut = DeterminantMinimiser(2)

    # Act
    det = sut.plot_landscape()

    # Assert