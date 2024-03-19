import numpy as np
import matplotlib.pyplot as plt
import MonteCarlo._MonteCarlo as MC
from numpy.random import Generator, SeedSequence, MT19937
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 100
datapoints = N//size

ss = SeedSequence(12345)    # Generate a sequence with intial seed 
sworker = ss.spawn(size)    # spawn the sequence on the workers
rangen = Generator(MT19937(sworker[rank]))   # random number generator

sigma = rangen.random()
x0 = rangen.random()
def function(x):
    """
    Return a Gaussian function with mean x0 and standard deviation sigma
    """
    return 1/(sigma*2*np.sqrt(np.pi)) * np.exp(-(x-x0)**2/(2*sigma**2))

def xi(sigma):
    """
    Returns sampling function, this is chosen to go from -4sigma to 
    +4sigma as 99.99% of the gaussian function is expected in this range
    """

    x = np.linspace(-4*sigma, 4*sigma, datapoints)
    return x

monte = MC.MonteCarlo(function, N, xi(sigma))
print("rank =", rank, "xi =", xi(sigma), "sigma =", sigma, "x0 =", x0, monte.evaluate())
