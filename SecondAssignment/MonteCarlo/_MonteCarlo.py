# ====================================
# Monte Carlo integration in parallel
# ====================================
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD()

class MonteCarlo(object):
    def __init__(self, f, a, b, x, N, nworkers, rank):
        self.MC_Integration = []
        self.f = function
        self.a = limit_a
        self.b = limit_b
        self.x = x_array
        self.N = data
        self.nworkers = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def integration(self):

        datapoints = self.N/self.n_workers

        partial_integral = np.zeros(datapoints)
        integral = np.array(0.0,dtype = np.double)
        
        for i in range(datapoints):
            x = random_x_loc[i]
            partial_integral[i] = self.f(x)
            partial_integral = partial_integral * (b-a)/self.N
        return np.sum(partial_integral)
    
