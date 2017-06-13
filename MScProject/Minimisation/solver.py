import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class SigmaSolver:
    def __init__(self):
        pass

    '''To do - fill me in:
    Solves for sigma explain what drift and diffusion are (2x2 numpy arrays) and equation I am trying to solve'''

    def solve(self, drift, diffusion):
        identity = np.eye(drift.shape[1])
        matrix = np.kron(identity, drift) + np.kron(drift, identity)

        # flatten into column-wise vector (F implies column-wise)
        diffusion_vec = -diffusion.flatten('F')

        # Left multiply inverse matrix by diffusion vector
        sigma_vec = np.linalg.solve(matrix, diffusion_vec)

        # Convert back to matrix (F implies column-wise)
        sigma = np.reshape(sigma_vec, drift.shape, order='F')

        return sigma


class DeterminantMinimiser:
    def __init__(self, chi):
        self.chi = chi
        self.size = 2
        self.sigma_solver = SigmaSolver()

    def minimise(self):

        # x represent [a, b, c, z} where a/b/c form a lower triangle of a
        # Cholesky factor and z is the element in the squeezer state
        x = np.array([6, -7, 2, -30000])

        # fourth parameter must be > 1 for a squeezer state. All other parameters are unbounded.
        bnds = [-float("inf"), float("inf")], [-float("inf"), float("inf")], [-float("inf"), float("inf")], [1, float("inf")],
        res = minimize(self.calculate_determinant, x, bounds=bnds)
        return res

    '''
    def test_function(self, x):
        q = np.eye(3)
        f = 0.5 * x.dot(q.dot(x))
        f += np.array([1, 2, 3]).dot(x)
        return f
    '''

    def calculate_determinant(self, x):
        hamiltonian = self.get_hamiltonian(*x[0:3])

        drift = self.calculate_drift(hamiltonian)
        diffusion = self.calculate_diffusion(x[3])
        sigma = self.sigma_solver.solve(drift, diffusion)

        determinant = np.linalg.det(sigma)

        return determinant

    def get_hamiltonian(self, a, b, c):
        lower = np.array([[a, 0], [b, c]])
        hamiltonian = lower.dot(lower.T)
        return hamiltonian

    def calculate_drift(self, hamiltonian):
        symplectic_matrix = np.array([[0, 1], [-1, 0]])

        drift_matrix = np.eye(self.size) * -0.5 + symplectic_matrix.dot(hamiltonian)

        return drift_matrix

    def plot_landscape(self):

        max = 2.50
        points = np.arange(-max, max, 0.025)

        a = np.zeros((len(points), len(points)))
        for i, x in enumerate(points):
            for j, y in enumerate(points):
                a[i, j] = self.calculate_determinant((x, y, 0, 1))

        b = np.zeros((len(points), len(points)))
        for j, y in enumerate(points):
            for k, z in enumerate(points):
                b[j, k] = self.calculate_determinant((0, y, z, 1))

        c = np.zeros((len(points), len(points)))
        for i, x in enumerate(points):
            for k, z in enumerate(points):
                c[i, k] = self.calculate_determinant((x, 0, z, 1))

        d = np.zeros((len(points), len(points)))
        for i, x in enumerate(points):
            for j, y in enumerate(points):
                d[i, j] = self.calculate_determinant((x, y, x, 1))

        ''' todo but need to get z points > 1 
        e = np.zeros((len(points), len(points)))
        for i, x in enumerate(points):
            for j, y in enumerate(points):
                e[i, j] = self.calculate_determinant((1, x, 1, y))


        f = np.zeros((len(points), len(points)))
        for i, x in enumerate(points):
            for j, y in enumerate(points):
                f[i, j] = self.calculate_determinant((x, 0, x, y))
        '''

        fig = plt.figure()

        ax11 = fig.add_subplot(2, 2, 1)
        ax11.imshow(a, cmap='viridis', extent=[-max, max, -max, max], interpolation='nearest')
        ax11.set_xlabel('a')
        ax11.set_ylabel('b')

        ax12 = fig.add_subplot(2, 2, 2)
        ax12.imshow(b, cmap='viridis', extent=[-max, max, -max, max], interpolation='nearest')
        ax12.set_xlabel('b')
        ax12.set_ylabel('c')

        ax21 = fig.add_subplot(2, 2, 3)
        ax21.imshow(c, cmap='viridis', extent=[-max, max, -max, max], interpolation='nearest')
        ax21.set_xlabel('a')
        ax21.set_ylabel('c')

        ax22 = fig.add_subplot(2, 2, 4)
        ax22.imshow(d, cmap='viridis', extent=[-max, max, -max, max], interpolation='nearest')
        ax22.set_xlabel('b')
        ax22.set_ylabel('a=c')

        plt.show()

    def calculate_diffusion(self, z):
        return self.chi * np.array([[z, 0], [0, 1/z]])
