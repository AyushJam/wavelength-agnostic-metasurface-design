'''
Code to run an optimization. 

Author: Ayush M. Jamdar

Before running: 
	1. Check config settings in config.py
	2. Check/Enter materials in materials.py

Output:
	1. Creates a folder "outcmaes" will all simulation data
	2. Prints iteration info in the terminal 
	3. Saves absorption spectrum data in text files for plotting

Use plotter.py for plotting.
'''

import numpy as np
import cma
from materials import refractive_indices
from config import *
from helper import *
 

# the objective function
def objective_function(params, Nx=200, Ny=200):
    """
    Objective function to minimize during optimization.

    Args:
        params: Array containing optimization parameters (l, h, center coordinates, radii).
        Nx, Ny: Grid points in x and y directions (default: 200).

    Returns:
        float: Sum of reflection and transmission coefficients for all frequencies.
    """

    l = params[0]
    h = params[1]
    c_x = params[2 : 2 + N_HOLES]
    c_y = params[2 + N_HOLES : 2 + 2 * N_HOLES]
    centers = [(cx, cy) for cx in c_x for cy in c_y]

    L1 = [l, 0]
    L2 = [0, l]
    pthick = [absorber_thickness, h, pec_thickness]

    radii = params[-N_HOLES:]

    epgrid2 = get_pattern_epgrid(radii, centers, L1, L2, ep1_diel, epbkg, Nx, Ny)
    epgrid3 = np.ones((Nx, Ny)) * epN

    R, T = [], []
    for f in f_sampled:
        R1, T1, _ = get_rt_nk(L1, L2, pthick, f, epgrid2, epgrid3, Nx, Ny)
        R.append(R1)
        T.append(T1)

    # Objective function: minimize sum of reflection and transmission
    return sum(R) + sum(T)


# Define optimization bounds
L_MIN = 0.5  # Minimum unit cell length (um)
L_MAX = 1.5  # Maximum unit cell length (um)
H_MIN = 0.1  # Minimum hole height (um)
H_MAX = 0.25  # Maximum hole height (um)
C_MIN = 0.0   # Minimum center coordinate (relative to unit cell)
C_MAX = 1.5   # Maximum center coordinate (relative to unit cell)
R_MIN = 0.05  # Minimum hole radius (um)
R_MAX = 0.5   # Maximum hole radius (um)

lower_bounds = [L_MIN, H_MIN] + [C_MIN] * 2 * N_HOLES + [R_MIN] * N_HOLES
upper_bounds = [L_MAX, H_MAX] + [C_MAX] * 2 * N_HOLES + [R_MAX] * N_HOLES


# Define initial guess for optimization parameters
x = get_random_params(
    L_MIN, L_MAX, H_MIN, H_MAX, C_MIN, C_MAX, R_MIN, R_MAX, N_HOLES
)

options = cma.CMAOptions()
options.set('tolfun', 1e-6)  # stop if change in objective is less than tolerance
options["bounds"] = [lower_bounds, upper_bounds] 

xopt, es = cma.fmin2(
    objective_function, x, SIGMA, options=options
)

# save the result to a file
with open("result.txt", "w") as f:
    f.write(str(es.result_pretty()))

# Save spectrum and absorption data
save_spectrum_data(
    xopt, 
    f_spectrum, 
    N_HOLES,
    ep1_diel,
    epbkg,
    epN,
    Nx,
    Ny,
)

