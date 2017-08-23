from Minimisation.solver import DeterminantMinimiser, SingleChiProblem, TwoModeCoolingProblem
import math
import numpy as np

# Single mode investigation
print('Chi,Optimised,Model,Agreement')
for chi in range(1, 21):
    one_mode_problem = SingleChiProblem(1, chi)
    minimiser = DeterminantMinimiser(one_mode_problem)
    res, hamiltonian, sigma = minimiser.minimise()

    theoretical = chi * chi
    percentage_agree = res.fun / theoretical
    print('{0} & {1:.16f} & {2:.4f} & {3:.4%} \\\\'.format(chi, res.fun, theoretical, percentage_agree))

one_mode_problem = SingleChiProblem(1, 2)
minimiser = DeterminantMinimiser(one_mode_problem)
# minimiser.plot_landscape()

# Parameters for the two mode cooling problem
# chi1 >= chi2 >= kappa2 >= kappa1
print('Chi,Optimised,Model,Agreement')
for chi in range(1, 21):
    # problem = SingleChiProblem(number_of_modes, chi)
    problem_cool = SingleChiProblem(2, chi)

    minimiser = DeterminantMinimiser(problem_cool)

    res, hamiltonian, sigma = minimiser.minimise()

    theoretical = chi ** 2
    percentage_agree = res.fun / theoretical
    print('{0} & {1} & {2:.4f} & {3:.4%} \\\\'.format(chi, res.fun, theoretical, percentage_agree))

# Parameters for the two mode cooling problem
# chi1 >= chi2 >= kappa2 >= kappa1
model_points = []
observed_points = []

print('Chi1,Chi2,Kappa1,Kappa2,Optimised,Model,Agreement')
for chi1 in range(1, 21):
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

                model_points.append(theoretical)
                observed_points.append(res.fun)

model_points = np.array(model_points)
observed_points = np.array(observed_points)

sse = np.dot(model_points - observed_points, model_points - observed_points)
sst = np.dot(observed_points - np.mean(observed_points), observed_points - np.mean(observed_points))
r_squared = 1 - sse/sst

print('sse: {0:.14f}, sse: {1:.14f}, r-Squares: {2:.14f}'.format(sse, sst, r_squared))

print(r_squared)