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

    def montecarlo(self):
        """
        Creates a random distribution shaped as exp(|x-x0|) and 
        calculates the mean of the given function, the standard deviation and the integral 
        with Monte Carlo method
        """
        datapoints = self.N/self.n_workers
        ranworker = self.rs.spawn(self.nworkers)    # spawn over workers
        randata = ranworker.spawn(datapoints)
        sigma, x0 = randata.spawn(2)
        xarray = np.linspace(-4*sigma, 4*sigma, datapoints)
        xi = np.exp(np.abs(xarray-x0))
        
        for i in range(datapoints):
            partial_avg = np.mean(self.function(xi[i]))
            partial_std = np.std(self.function(xi[i]))
            partial_integral = partial_avg*(sigma[i])
       
    
