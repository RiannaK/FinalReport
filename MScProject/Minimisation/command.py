from Minimisation.solver import DeterminantMinimiser
chi = 2
minimiser = DeterminantMinimiser(chi)

res = minimiser.minimise()
print('Solution is: {0}'.format(res.x))
print('Minimum determinant is: {0}'.format(res.fun))
det = minimiser.plot_landscape()
