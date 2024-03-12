# ====================================
# Monte Carlo integration in parallel
# ====================================


import numpy as np
from mpi4py import MPI

class MonteCarlo(object):
    def __init__(self, f, a, b, x, n):
        self.MC_Integration = []
        self.f = function
        self.a = limit_a
        self.b = limit_b
        self.x = x_array
        self.n = n

    def integration(f, a, b, x, n):

        rank = comm.Get_rank()
        size = comm.Get_size()
        
        nproc = comm.Get_size()
    
        # The first processor is leader, so one fewer available to be a worker
        nworkers = nproc - 1
        
        integral = np.array(0.0,dtype = np.double)
        
        for i in range(rank, n-1, size):
            
            partial_integral = partial_integral + (b-a)/n * np.sum(f(x))
        
        integral = com.reduce(partial_integral, MPI.SUM, 0)
        
        if i ==0:
            print("result = ", integral)
        MPI.Finalize()
        return 
    
