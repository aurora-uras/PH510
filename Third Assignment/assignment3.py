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

L = 10e-2  # Grid size = 10cm
lattice_points = 10  # Number of Points in lattice
h = L / lattice_points  # Lattice size

EPSILON0 = 8.854e-12  # Permittivity of Vaccum
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
    
    """
    gf = 0
    result = 0
    F = 0
    for i in range(N):
        x = list(A)
        while True:
            if x[0] <= 0 or x[0] >= L or x[1] <= 0 or x[1] >= L:
			# conclude random walk if exits boundaries
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
        gf += g(x) / N
    result = gf - F
    return result, gf

def plot(x, y, res):
    # Function for plotting the potential and the green function
    pot = res[0]
    pot = pot.reshape(lattice_x.shape)
    gf = res[1]
    gf = gf.reshape(lattice_x.shape)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(np.array(pot), cmap=cm.jet)
    ax1.xlabel("X (Meters)")
    ax1.ylabel("Y (Meters)")
    cb = ax1.colorbar(location='right', label = "Potential (V)")

    ax2.imshow(np.array(gf), cmap=cm.jet)
    ax2.xlabel("X (Meters)")
    ax2.ylabel("Y (Meters)")
    cb = ax2.colorbar(location='right', label = "Green Function")

if __name__ == "__main__":
    # a) all edges at 1V
    lattice_x, lattice_y = np.mgrid[
        0 : L : lattice_points * 1j, 0 : L : lattice_points * 1j
    ]
    res =  poisson_approximation_fixed_step(lattice_x.ravel(), lattice_y.ravel())

    res1 = poisson_approximation_fixed_step(5, 5)
    res2 = poisson_approximation_fixed_step(0, 2.5)
    res3 = poisson_approximation_fixed_step(0, 0)
    print("case a)", res1, res2, res3)

    plot(lattice_x, lattice_y, res)
    plt.savefig("case_a.png")

    # b) top and bottom edges at +1V, left and right at -1V
    f = f
    g = g2
    lattice_x, lattice_y = np.mgrid[
        0 : L : lattice_points * 1j, 0 : L : lattice_points * 1j
    ]
    z = poisson_approximation_fixed_step(lattice_x.ravel(), lattice_y.ravel()).reshape(
        lattice_x.shape
    )

    plot(lattice_x, lattice_y, z)
    plt.savefig("potential_b.png")

    # c) top and left edges at +2V, bottom at 0V and right at -4V
    f = f
    g = g3
    lattice_x, lattice_y = np.mgrid[
        0 : L : lattice_points * 1j, 0 : L : lattice_points * 1j
    ]
    print("case c)", poisson_approximation_fixed_step(5,5))
    z = poisson_approximation_fixed_step(lattice_x.ravel(), lattice_y.ravel()).reshape(
        lattice_x.shape
    )
    plot(lattice_x, lattice_y, z)
    plt.savefig("potential_c.png")
