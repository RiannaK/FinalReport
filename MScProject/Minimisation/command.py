from Minimisation.solver import DeterminantMinimiser, SingleChiProblem, TwoModeCoolingProblem
import math

# Single mode investigation
print('Chi,Optimised,Model,Agreement')
for chi in range(1, 10):
    one_mode_problem = SingleChiProblem(1, chi)
    minimiser = DeterminantMinimiser(one_mode_problem)
    res, hamiltonian, sigma = minimiser.minimise()

    theoretical = chi * chi
    percentage_agree = res.fun / theoretical
    print('{0} & {1:.4f} & {2:.4f} & {3:.4%} \\\\'.format(chi, res.fun, theoretical, percentage_agree))

one_mode_problem = SingleChiProblem(1, 2)
minimiser = DeterminantMinimiser(one_mode_problem)
minimiser.plot_landscape()

# Parameters for the two mode cooling problem
# chi1 >= chi2 >= kappa2 >= kappa1
print('Chi1,Chi2,Kappa1,Kappa2,Optimised,Model,Agreement')
for chi1 in range(1, 6):
    for chi2 in range(1, chi1 + 1):
        for kappa2 in range(1, chi2 + 1):
            for kappa1 in range(1, kappa2 + 1):
                # problem = SingleChiProblem(number_of_modes, chi)
                problem_cool = TwoModeCoolingProblem(2, chi1, chi2, kappa1, kappa2)

                minimiser = DeterminantMinimiser(problem_cool)

                res, hamiltonian, sigma = minimiser.minimise()

                theoretical = math.pow( (chi1 + chi2) / (kappa1 + kappa2),2)
                percentage_agree = res.fun / theoretical
                print('{0} & {1} & {2} & {3} & {4:.4f} & {5:.4f} & {6:.4%} \\\\'.format(chi1, chi2, kappa1, kappa2, res.fun, theoretical, percentage_agree))

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
'''