import numpy as np

# Constants for field solver
n_sub = 1.44

min_N_bars = 3
max_N_bars = 10

eps_0 = 8.85418782e-12
mu_0 = 1.25663706e-6
C_0 = 1 / np.sqrt(eps_0 * mu_0)  # speed of light in vacuum

dL = 10

Nx = 500
Ny = 350
pml_x = 60
pml_y = 60


nn_padding = 3