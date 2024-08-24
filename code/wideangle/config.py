import numpy as np
from materials import refractive_indices

# wavelength of incident light
wavelength = 1.55
f = 1 / wavelength

# specify the angle of incidence upto which absorption is to be maximised
target_angle = 30  # in degrees

# thickness of the absorber layer in micron
absorber_thickness = 0.004  # 4 nm

# PEC / Metal thickness
pec_thickness = 0.1 

# choose the materials from materials.py
absorber_material = "BP"  
dielectric_material = "Silicon"
cavity_material = "Air"

# start to end incidence angles for plotting
start_angle = 0
end_angle = 45

# array of sampled angles 
angle_nsamples = 5
angle_sampled = np.deg2rad(np.linspace(0, target_angle, angle_nsamples))
anglesx = np.deg2rad(np.linspace(start_angle, end_angle, 200))

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

# PEC / metal layer
epN = refractive_indices["PEC"] ** 2