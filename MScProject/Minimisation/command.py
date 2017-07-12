from Minimisation.solver import DeterminantMinimiser
from numpy import linalg as alg

chi = 1
number_of_modes = 2
minimiser = DeterminantMinimiser(chi, number_of_modes)

res, hamiltonian, sigma = minimiser.minimise()
eigs = alg.eigvals(sigma)

print('Solution, x, is:')
print(res.x)

print('Minimum determinant is: {0}'.format(res.fun))

print('Hamiltonian is:/')
print(hamiltonian)

print('Sigma is:/')
print(sigma)

print('Eigenvalues of sigma:')
print(eigs)

#det = minimiser.plot_landscape()
