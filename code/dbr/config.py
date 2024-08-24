import numpy as np
from materials import refractive_indices

# specify the target wavelength (for single resonance)
CENTER_WAV = 2.1 

# width around the center wavelength (for plotting)
WIDTH = 0.2  

# thickness of the absorber layer in micron
absorber_thickness = 0.004  # 4 nm

# choose the materials from materials.py
absorber_material = "BP"  
dielectric_material = "Silicon"
cavity_material = "Air"

# start to end wavelengths for plotting
start_wavelength = CENTER_WAV - WIDTH
end_wavelength = CENTER_WAV + WIDTH

# array of target wavelengths
f_sampled = np.array([1 / CENTER_WAV])

# number of holes per unit cell
N_HOLES = 1

# initial step-size for CMA-ES
SIGMA = 0.7

# grid size
Nx, Ny = 200, 200

# dielectric permitivitty
ep1_diel = refractive_indices[dielectric_material] ** 2  

# cavity permittivity (generally air)
epbkg = refractive_indices[cavity_material]

