#!/usr/bin/env python3 

from mpi4py import MPI

comm = MPI.COMM_WORLD

n_proc = comm.Get_size()    # gives the number of processors available
n_workers = n_proc - 1   # gives the number of workers (one processor is the leader)

N_SAMPLES = 100000000    # number of samples taken
DELTA = 1.0/N_SAMPLES

# integral
INTEGRAL = 0.0
def integrand(x):
    """
    Returns the integrated function f(x) = 4/(1+x^2) for x
    """
    return 4.0 / (1.0 + x**2)

if comm.Get_rank() == 0:   # this is the leader
  # Leader: choose points to sample function, send to workers and
  # collect their contributions. Also calculate a sub-set of points.
    for i in range(0,N_SAMPLES):
        # decide which rank evaluates this point
        j = i % n_proc
        # Using the mid-point rule
        x_i = (i+0.5)/N_SAMPLES
        if j == 0:
            # so do this locally using the leader machine
            y = integrand(x_i) * DELTA
        else:
            # communicate to a worker
            comm.send(x_i, dest=j)
            y = comm.recv(source=j)
        
        INTEGRAL += y

  # Shut down the workers
    for i in range(1, n_proc):
        comm.send(-1.0, dest=i)
    print("\u03C0 \u2248 {:.15f}".format(INTEGRAL))

else:
  # Worker: waiting for something to happen, then stop if sent message
  # outside the integral limits
    while True:
        x_i = comm.recv(source=0)
        if x_i < 0.0:
            # stop the worker
            break
        else:
            comm.send(integrand(x_i) * DELTA, dest=0)
