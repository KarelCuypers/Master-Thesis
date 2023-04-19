import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi


def uniaxial_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = c*y
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-beta*w)

    return displacement, strained_hopping


model = pb.Model(graphene.bilayer(),
                 uniaxial_strain(0.12),
                 pb.translational_symmetry()
                 )
model.plot()
plt.show()

solver = pb.solver.lapack(model)

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*np.sqrt(3)*a_cc), 0]
S = [0, 2*pi / (3*a_cc)]
R = [2*pi / (3*np.sqrt(3)*a_cc), 2*pi / (3*a_cc)]


bands = solver.calc_bands(Gamma, R, S)
model.lattice.plot_brillouin_zone()
bands.plot_kpath(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()
bands.plot(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()

bands = solver.calc_bands(K1, Gamma, S, R)
bands.plot(point_labels=['K', r'$\Gamma$', 'S', 'R'])
plt.show()
