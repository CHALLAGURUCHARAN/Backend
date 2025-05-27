import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
from scipy.spatial.distance import euclidean

class MobiusStrip:
    def __init__(self, R=1.0, w=0.5, n=100):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._compute_coordinates()

    def _compute_coordinates(self):
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        dU = 2 * np.pi / (self.n - 1)
        dV = self.w / (self.n - 1)

        Xu = np.gradient(self.X, axis=1) / dU
        Yu = np.gradient(self.Y, axis=1) / dU
        Zu = np.gradient(self.Z, axis=1) / dU

        Xv = np.gradient(self.X, axis=0) / dV
        Yv = np.gradient(self.Y, axis=0) / dV
        Zv = np.gradient(self.Z, axis=0) / dV

        cross_prod = np.sqrt(
            (Yu * Zv - Zu * Yv) ** 2 +
            (Zu * Xv - Xu * Zv) ** 2 +
            (Xu * Yv - Yu * Xv) ** 2
        )

        area = simps(simps(cross_prod, self.v), self.u)
        return area

    def compute_edge_length(self):
        edge_upper = np.array([
            (self.X[-1, i], self.Y[-1, i], self.Z[-1, i])
            for i in range(self.n)
        ])
        edge_lower = np.array([
            (self.X[0, i], self.Y[0, i], self.Z[0, i])
            for i in range(self.n)
        ])

        def total_length(edge):
            return sum(euclidean(edge[i], edge[i + 1]) for i in range(len(edge) - 1))

        return total_length(edge_upper) + total_length(edge_lower)

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, color='skyblue', edgecolor='k', alpha=0.7)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()