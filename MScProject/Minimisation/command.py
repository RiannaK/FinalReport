from Minimisation.solver import DeterminantMinimiser, TwoModeProblem, TwoModeCoolingProblem
from numpy import linalg as alg
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters for the standard two mode cooling problem
chi = 1
number_of_modes = 2

# Parameters for the two mode cooling problem
chi1 = 1
chi2 = 1
kappa1 = 2
kappa2 = 3

'''
kappas = []
results = []

for kappa2 in np.arange(2,5,0.1):
'''
problem = TwoModeProblem(number_of_modes, chi)
problem_cool = TwoModeCoolingProblem(number_of_modes, chi1, chi2, kappa1, kappa2)

minimiser = DeterminantMinimiser(problem_cool)

res, hamiltonian, sigma = minimiser.minimise()
'''
    print(kappa2)
    kappas.append(kappa2)
    results.append(res.fun)

    # eigs = alg.eigvals(sigma)

plt.plot(kappas, results)
plt.show()
'''
print('Solution, x, is:')
print(res.x)

print('Minimum determinant is: {0}'.format(res.fun))

print('Hamiltonian is:/')
print(hamiltonian)

print('Sigma is:/')
print(sigma)

#print('Eigenvalues of sigma:')
#print(eigs)

#det = minimiser.plot_landscape()
