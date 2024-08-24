import matplotlib.pyplot as plt
import numpy as np
from helper import (  # Assuming helper.py has these functions
    plot_metasurface,
    plot_absorption_spectrum,
)
from config import *

# Load optimized parameters from result.txt
with open("result.txt", "r") as f:
    result_text = f.read()
    xopt_str = result_text.split("]")[0].split("[")[1]  # Extract xopt as a string
    xopt = np.fromstring(xopt_str, sep=",", dtype=float)  # Convert to numpy array

# Plot metasurface
fig_metasurface = plot_metasurface(xopt, N_HOLES, ep1_diel, epbkg, Nx, Ny)
plt.savefig("metasurface.png", dpi=300)
plt.show()

# Plot absorption spectrum
fig_absorption, ax_absorption = plot_absorption_spectrum(angles_filename="angles.txt", abs_filename="absorption.txt")
plt.savefig("absorption_spectrum.png", dpi=300)
plt.show()

