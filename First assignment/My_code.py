#!/usr/bin/env python3
args = "pip install -r requirements.txt ; pylint **/*.py"

from mpi4py import MPI

comm = MPI.COMM_WORLD

nproc = comm.Get_size()
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1

N_samples = 100    # number of samples taken
delta = 1.0/N_samples

# integral
integral = 0.0
def integrand(x):
  """
  Returns the integrated function f(x) = 4/(1+x^2) for x
  """

  return(4.0 / (1.0 + x*x))

if comm.Get_rank() == 0:   # this is the leader

  # Leader: choose points to sample function, send to workers and
  # collect their contributions. Also calculate a sub-set of points.

  for i in range(0,N_samples):

    # decide which rank evaluates this point
    j = i % nproc

    # Using the mid-point rule
    x_i = (i+0.5)/N_samples

    if j == 0:
      # so do this locally using the leader machine
      y = integrand(x_i) * delta
    else:
      # communicate to a worker
      comm.send(x_i, dest=j)
      y = comm.recv(source=j)

    integral += y

  # Shut down the workers
  for i in range(1, nproc):
    comm.send(-1.0, dest=i)

  print("Integral %.15f" % integral)

else:

  # Worker: waiting for something to happen, then stop if sent message
  # outside the integral limits

  while True:

    x = comm.recv(source=0)

    if x_i < 0.0:
      # stop the worker
      break

    else:
      comm.send(integrand(x) * delta, dest=0)