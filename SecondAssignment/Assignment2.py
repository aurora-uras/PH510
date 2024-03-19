import numpy as np
import matplotlib.pyplot as plt
import MonteCarlo._MonteCarlo as MC
from numpy.random import Generator, SeedSequence, MT19937
from mpi4py import MPI



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 100
datapoints = n//size

ss = SeedSequence(12345)    # Generate a sequence with intial seed 
sworker = ss.spawn(size)    # spawn the sequence on the workers
rangen = Generator(MT19937(sworker[rank]))   # random number generator

xi = rangen.random(datapoints)
sigma = rangen.random()
x0 = rangen.random()
def function(x):
    return 1/(sigma*2*np.sqrt(np.pi)) * np.exp(-(x-x0)**2/(2*sigma**2))

#print(rank, sigma, x0)
monte = MC.MonteCarlo(function, n, xi)
print("rank =", rank, "xi =", xi, "sigma =", sigma, "x0 =", x0, monte.evaluate())
