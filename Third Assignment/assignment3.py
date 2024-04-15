import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numpy.random import Generator, SeedSequence, MT19937
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ss = SeedSequence(12345)    # Generate a sequence with intial seed
sworker = ss.spawn(size)    # spawn the sequence on the workers
rangen = Generator(MT19937(sworker[rank]))   # random number generator

L = 10  # Grid size = 10cm
lattice_points = 10  # Number of Points in lattice
h = L / lattice_points  # Lattice size
N = 20  # Number of Random Walks


def f(x):
    """
    Returns Laplace equation nabla^2(phi)=0
    """
    return 0

def g(x):
    """
    Boundary conditions a) all edges uniforly at +1V
    """
    if x[0] <= 0 or x[0] >= L:
         return 1.0
    if x[1] <= 0 or x[1] >= L:
         return 1.0

def g2(x):
    """
    Boundary conditions b) top and bottom edges at +1V, left and rigth edges at -1V
    """
    if x[0] <= 0 or x[0] >= L:
         return 1.0
    if x[1] <= 0 or x[1] >= L:
         return -1.0

def g3(x):
    """
    Boundary conditions c) top and left edges at +2V, bottom at 0V and rigth edge at -4V
    """
    if x[0] >= L or x[1] <= 0:
        return 2.0
    if x[0] <= 0:
        return 0.0
    if x[1] >= L:
        return -4.0

@np.vectorize
def poisson_approximation(*A):
    """
    Returns the potential across the grid for given boundary conditions
    """
    exit = []
    result = 0
    F = 0
    for i in range(N):
        x = list(A)
        while True:
            if x[0] <= 0 or x[0] >= L or x[1] <= 0 or x[1] >= L:
                exit.append(x)
	    # conclude RW if exits boundaries and save exit point
                break
            r = rangen.random()
            if r < 0.25:
            # step left
                x[0] -= h
            elif r >= 0.25 and r < 0.5:
            # step down
                x[1] -= h
            elif r >= 0.5 and r < 0.75:
            # step rigth
                x[0] += h
            elif r >= 0.75:
            # step up
                x[1] += h

            F += f(x) * h ** 2
        result += g(x) / N
    result = result - F
    return result, exit

def plot_potential(x, y, res):
    # Function for plotting the potential
    pot = res[0]
    pot = pot.reshape(lattice_x.shape)

    plt.imshow(np.array(pot), cmap=cm.jet)
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    cb = plt.colorbar(location='right', label = "Potential (V)")
    plt.show()

if __name__ == "__main__":
    lattice_x, lattice_y = np.meshgrid(np.linspace(0, L, lattice_points), 
			 (0, L, lattice_points))
    # a) all edges at 1V
    z =  poisson_approximation(lattice_x.ravel(), lattice_y.ravel())
    pot = z[0].reshace(xx.shape)

    plot(lattice_x, lattice_y, res)
    plt.savefig("case_a.png")
    potA1 = pot[np.where(lattice_x <= 5), np.where(lattice_y <= 5)]
    potA2 = pot[np.where(lattice_x <= 2.5), np.where(lattice_y <= 2.5)]
    potA3 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 2.5)]
    potA4 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 0)]

    print("case a)", potA[0,0], potB[0, 0], potC[0, 0], potD[0, 0])
    plot_potential(lattice_x, lattice_y, z)
    plt.savefig("potential_a.png")
    plt.clf()

    # b) top and bottom edges at +1V, left and right at -1V
    f = f
    g = g2
	
    z =  poisson_approximation(lattice_x.ravel(), lattice_y.ravel())
    pot = z[0]
    pot = pot.reshape(lattice_x.shape)
    potA1 = pot[np.where(lattice_x <= 5), np.where(lattice_y <= 5)]
    potA2 = pot[np.where(lattice_x <= 2.5), np.where(lattice_y <= 2.5)]
    potA3 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 2.5)]
    potA4 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 0)]

    print("case b)", potA[0,0], potB[0, 0], potC[0, 0], potD[0, 0])
    plot_potential(lattice_x, lattice_y, z)
    plt.savefig("potential_b.png")
    plt.clf()

    # c) top and left edges at +2V, bottom at 0V and right at -4V
    f = f
    g = g3
    lattice_x, lattice_y = np.mgrid[
        0 : L : lattice_points * 1j, 0 : L : lattice_points * 1j
    ]
    z =  poisson_approximation(lattice_x.ravel(), lattice_y.ravel())
    pot = z[0]
    pot = pot.reshape(lattice_x.shape)
    potA1 = pot[np.where(lattice_x <= 5), np.where(lattice_y <= 5)]
    potA2 = pot[np.where(lattice_x <= 2.5), np.where(lattice_y <= 2.5)]
    potA3 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 2.5)]
    potA4 = pot[np.where(lattice_x <= 0), np.where(lattice_y <= 0)]

    print("case c)", potA[0,0], potB[0, 0], potC[0, 0], potD[0, 0])
    plot_potential(lattice_x, lattice_y, z)
    plt.savefig("potential_c.png")
    plt.clf()
