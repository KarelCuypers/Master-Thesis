import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from export_xyz import export_xyz
from draw_contour import draw_contour


def sinusoidal_strain(c, k, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = 0
        uz = c * np.cos(k[0]*x + k[1]*y)
        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w_intra = l / graphene.a_cc - 1
        w_inter = l / 0.335 - 1

        for i in np.subtract(z1, z2):  # changed because the wave in the z direction would break earlier code
            if i < 0.33:
                return energy * np.exp(-beta * w_intra)
            else:
                return 0.48 * np.exp(-beta * w_inter)

    return displacement, strained_hopping


times_l1 = 5  # determines size of the unit cell of the graphene sheet
a = graphene.a_cc * sqrt(3)  # graphene unit cell length

l1_size = times_l1 * a # lattice vector length * number of l1
l2_size = l1_size * 2

k = 2 * pi / l1_size * graphene.bilayer(gamma3=False).vectors[0]/a  # wavelength = 2*pi/lattice vector length * number of l1
# this makes sure there is exactly one period in the unit cell

strained_model = pb.Model(
    graphene.bilayer(gamma3=False),
    pb.rectangle(x=l1_size*5, y=l2_size*5),
    pb.translational_symmetry(a1=l1_size, a2=l2_size),  # always needs some overlap with the rectangle
    sinusoidal_strain(0.2, k)
)

#unstrained_model = pb.Model(
#    graphene.bilayer(),
#    pb.rectangle(x=l1_size * 5, y=l2_size * 2.5),
#    pb.translational_symmetry(a1=l1_size * 2.5, a2=l2_size),
#)

strained_model.plot()
strained_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
plt.show()

#unstrained_model.plot()
#plt.show()

solver = pb.solver.arpack(strained_model, 10)

bands = solver.calc_bands(-pi / a, pi / a)
#bands.plot()

for e in range(0, 10):
    plt.scatter(bands.k_path, bands.energy[:, e], s=1) # methode to make much nicer looking plot or plot bands
    # independently
plt.show()

position = strained_model.system.xyz

#solver2 = pb.solver.arpack(unstrained_model, 10)
#bands = solver2.calc_bands(-pi / a, pi / a)
#for e in range(0, 10):
#    plt.scatter(bands.k_path, bands.energy[:, e], s=1)
#plt.show()

kpm_strain = pb.kpm(strained_model)
for sub_name in ['A1', 'B1', 'A2', 'B2']:
    ldos = kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
                                position=[0, 0], sublattice=sub_name)
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()

lat = [[i[0]*(2*times_l1), i[1]*(2*times_l1)] for i in strained_model.lattice.vectors]
full_lattice = pb.Lattice(a1=lat[0], a2=lat[1])
full_lattice.plot_brillouin_zone()
k_points = [i for i in full_lattice.brillouin_zone()]
plt.scatter(k_points[4][0], k_points[4][1])
plt.show()

Gamma = [0, 0]
K1 = [k_points[0][0], k_points[0][1]]
M = [0, k_points[4][1]]
K2 = [k_points[5][0], k_points[5][1]]

bands = solver.calc_bands(K1, K2, Gamma, K1)
bands.plot(point_labels=['K1', 'K2', r'$\Gamma$', 'K1'])

kx = k_points[0][0]*1.5
ky = k_points[5][1]*1.5

kx_space = np.linspace(kx, -kx, 100)
ky_space = np.linspace(ky, -ky, 100)

draw_contour(solver, kx_space, ky_space)

export_xyz("square_xyz", position, strained_model.lattice.vectors[0] * times_l1, strained_model.lattice.vectors[1]
           * times_l1 * 2, np.array([0, 0, 1]), ['c'] * position.shape[0])
