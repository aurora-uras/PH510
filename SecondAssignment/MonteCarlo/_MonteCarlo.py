"""
Module providing Monte Carlo parallel evaluation
"""
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD()
size = comm.Get_size()
rank = comm.Get_rank()

class MonteCarlo:
    """
    Monte Carlo integral evaluation in series and parallel
    """
    def __init__(self, function, n_points, seed):
        self.MC_Integration = []
        self.function = function
        self.npoints = npoints
        self.rs = np.random.SeedSequence(seed)

    def integration(self):

        datapoints = self.N/self.n_workers

        partial_integral = np.zeros(datapoints)
        integral = np.array(0.0,dtype = np.double)
        
        for i in range(datapoints):
            x = random_x_loc[i]
            partial_integral[i] = self.f(x)
            partial_integral = partial_integral * (b-a)/self.N
        return np.sum(partial_integral)
    
