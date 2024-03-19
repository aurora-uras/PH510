"""
Module providing Monte Carlo parallel evaluation of integral of a function, 
together with mean value and variance of the function
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
    def __init__(self, function, npoints, seed):
        self.function = function
        self.npoints = npoints
        self.rs = np.random.SeedSequence(seed)

    def evaluate(self):
        """
        calculates the mean of the given function, the standard deviation 
        and the integral with Monte Carlo method
        """
        datapoints = self.npoints/size
        
        for i in range(0, int(datapoints)):
            partial_avg = np.mean(self.function(self.xi[i]))
            partial_std = np.std(self.function(self.xi[i]))
            partial_integral = (max(self.xi)-min(self.xi))/datapoints *np.sum(self.function(self.xi))
			
            avg = comm.reduce(partial_avg, MPI.SUM, 0)/size
            std = comm.reduce(partial_integral, MPI.SUM, 0)
            integral = comm.reduce(partial_integral, MPI.SUM, 0)

			if i == 0:
				print("Integral =", integral, "\n average =", avg,
					"standard deviation =", std,)

		return

                
       
    
