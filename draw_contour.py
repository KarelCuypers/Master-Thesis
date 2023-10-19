import numpy as np
import matplotlib.pyplot as plt
import pybinding as pb
from math import sqrt, pi
from pybinding.repository import graphene


def draw_contour(solver, kx_space, ky_space):

    def calculate_surfaces(kx, ky):
        solver.set_wave_vector([kx, ky])
        return solver.eigenvalues

    KX, KY = np.meshgrid(kx_space, ky_space)
    kx_list, ky_list = list(KX.flatten()), list(KY.flatten())

    conduction_E = np.zeros(shape=len(kx_space) * len(ky_space))
    valence_E = np.zeros(shape=len(kx_space) * len(ky_space))

    conduction_index = 4
    valence_index = 5

    for i, (kx, ky) in enumerate(zip(kx_list, ky_list)):

        eigenvalues = calculate_surfaces(kx, ky)

        valence_E[i] = eigenvalues[valence_index]
        conduction_E[i] = eigenvalues[conduction_index]

    conduction_E = conduction_E.reshape(len(ky_space), len(kx_space))
    valence_E = valence_E.reshape(len(ky_space), len(kx_space))

    plt.figure(dpi=100)

    cmap = plt.get_cmap('coolwarm')
    plt.contourf(conduction_E, 50, cmap=cmap)
    plt.colorbar()
    plt.show()
