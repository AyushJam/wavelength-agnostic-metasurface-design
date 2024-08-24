import numpy as np
import matplotlib.pyplot as plt
import grcwa
import seaborn as sns
from materials import refractive_indices
from config import absorber_material, absorber_thickness, pec_thickness


def get_layer_absorption(layer_number, layer_thickness, area, Nx, Ny, rcwa_obj):
    """
    Calculates the fractional absorption of a specific layer in a structure.

    Args:
        layer_number: Layer index (starts from 1).
        layer_thickness: Thickness of the layer.
        area: Total area of the structure.
        Nx, Ny: Number of grid points in x and y directions.
        rcwa_obj: The grcwa object representing the structure.

    Returns:
        float: The fractional absorption of the specified layer.
    """

    tol = 1e-9
    E_top, H_top = rcwa_obj.Solve_FieldOnGrid(layer_number, tol * layer_thickness)
    E_bottom, H_bottom = rcwa_obj.Solve_FieldOnGrid(
        layer_number, (1 - tol) * layer_thickness
    )
    E_top = np.array(E_top)
    E_bottom = np.array(E_bottom)
    H_top = np.array(H_top)
    H_bottom = np.array(H_bottom)

    dA = area / (Nx * Ny)

    S_top = (
        np.sum(np.real(-E_top[1] * np.conj(H_top[0]) + E_top[0] * np.conj(H_top[1])))
        * dA
        / 2
    )

    S_bottom = (
        np.sum(
            np.real(
                -E_bottom[1] * np.conj(H_bottom[0]) + E_bottom[0] * np.conj(H_bottom[1])
            )
        )
        * dA
        / 2
    )

    absorption_fraction = (S_top - S_bottom) / (area / 2)

    return absorption_fraction


def get_pattern_epgrid(radii, centers, L1, L2, ep_dielectric, ep_hole, Nx, Ny):
    """
    Generates a permittivity grid representing a patterned hole structure.

    Args:
        radii: List of hole radii.
        centers: List of hole centers as tuples (x, y).
        L1, L2: Lengths of the unit cell in x and y directions.
        ep_dielectric: Permittivity of the dielectric material.
        ep_hole: Permittivity of the hole material.
        Nx, Ny: Number of grid points in x and y directions.

    Returns:
        ndarray: The permittivity grid for the patterned structure.
    """

    x0 = np.linspace(0, L1[0], Nx)
    y0 = np.linspace(0, L2[1], Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")

    epgrid = np.ones((Nx, Ny)) * ep_dielectric
    for i in range(len(radii)):
        ind = (x - centers[i][0]) ** 2 + (y - centers[i][1]) ** 2 < radii[i] ** 2
        epgrid[ind] = ep_hole

    return epgrid


def get_random_params(L_MIN, L_MAX, H_MIN, H_MAX, C_MIN, C_MAX, R_MIN, R_MAX, N_HOLES):
    """
    Generates random parameters for the structure.

    Args:
        L_MIN, L_MAX, H_MIN, H_MAX, C_MIN, C_MAX, R_MIN, R_MAX: Minimum and maximum values for various parameters.
        N_HOLES: Number of holes in the structure.

    Returns:
        ndarray: Array containing the randomly generated parameters.
    """

    l = np.random.uniform(L_MIN, L_MAX)
    h = np.random.uniform(H_MIN, H_MAX)

    # Initialize hole centers at the center
    c_x = [l / 2 for i in range(N_HOLES)]  
    c_y = [l / 2 for i in range(N_HOLES)]

    r = [np.random.uniform(R_MIN, R_MAX) for i in range(N_HOLES)]

    # initial parameter vector
    x = np.array(([l] + [h] + c_x + c_y + r))
    return x


def plot_metasurface(xopt, N_HOLES, ep1_diel, epbkg, Nx, Ny, tiles=(3, 3)):
  """
  Plots a repeating array of the unit cell structure.

  Args:
      xopt: Optimized parameters from the optimization process.
      N_HOLES: Number of holes in the structure.
      ep1_diel: Permittivity of the dielectric material.
      epbkg: Permittivity of the background material.
      Nx, Ny: Number of grid points in x and y directions for a single unit cell.
      tiles: Tuple specifying the number of unit cells to tile in x and y directions (default: (3, 3)).

  Returns:
      matplotlib.figure.Figure: The plot of the repeating unit cell structure.
  """

  # Extract hole radii and center coordinates
  radii = xopt[-N_HOLES:]
  c_x = xopt[2: 2 + N_HOLES]
  c_y = xopt[2 + N_HOLES: 2 + 2 * N_HOLES]
  centers = list(zip(c_x, c_y))  # Combine x and y center coordinates

  # Unit cell dimensions
  L1 = [xopt[0], 0]
  L2 = [0, xopt[0]]

  # Calculate permittivity grid for a single unit cell
  epgrid = get_pattern_epgrid(radii, centers, L1, L2, ep1_diel, epbkg, Nx, Ny)

  # Create a larger grid for tiling the unit cell
  x_tiles, y_tiles = tiles
  x_ = np.linspace(0, xopt[0] * x_tiles, Nx * x_tiles)
  y_ = np.linspace(0, xopt[0] * y_tiles, Ny * y_tiles)
  x, y = np.meshgrid(x_, y_, indexing="ij")

  # Tile the permittivity grid to match the larger grid
  plane_epgrid = np.tile(epgrid, (x_tiles, y_tiles))

  # Create the plot
  fig, ax = plt.subplots(figsize=(7, 7))
  ax.contourf(x, y, plane_epgrid, cmap="binary")
  ax.set_aspect("equal")
  ax.set_xlim(0, max(x_))
  # ax.set_xticks([0.0 + i * xopt[0] for i in range(x_tiles + 1)], fontsize=20)
  # ax.set_yticks([0.0 + i * xopt[0] for i in range(y_tiles + 1)], fontsize=20)
  ax.set_xlabel(r"$\mu m$", fontsize=20)
  ax.set_ylabel(r"$\mu m$", fontsize=20)
  # Optional title: ax.set_title("Metasurface")

  return fig


def get_epgrid_nk(f, epgrid2, epgrid3, Nx, Ny):
    """
    Constructs the permittivity grid considering varying refractive indices.
    NOTE: 
    - This function is specific to the DBR case.
    - nk in the function name implies that refractive index is a function of wavelength
    - the same must be specified in materials.py and config.py

    Args:
        f: Frequency.
        epgrid2, epgrid3, epgrid4: Permittivity grids for different layers.
        Nx, Ny: Number of grid points in x and y directions.

    Returns:
        ndarray: The combined permittivity grid with nk dependence.
    """

    e_top_f = refractive_indices[absorber_material](f) ** 2
    epgrid_1 = np.ones((Nx, Ny)) * e_top_f
    epgrid = np.concatenate((epgrid_1.flatten(), epgrid2.flatten(), epgrid3.flatten()))
    return epgrid


def get_rt_nk(L1, L2, pthick, f, theta, epgrid2, epgrid3, Nx, Ny):
  """
  Calculates reflection and transmission coefficients using GRCWA.

  Args:
      L1, L2: Lengths of the unit cell in x and y directions.
      pthick: List of thicknesses for each layer.
      f: Frequency.
      theta: angle of incidence in radians
      epgrid2, epgrid3, epgrid4: Permittivity grids for different layers.
      Nx, Ny: Number of grid points in x and y directions.

  Returns:
      tuple: A tuple containing reflection coefficient (R), transmission coefficient (T),
             and the GRCWA object for further analysis (optional).
  """

  # Truncation order (adjust as needed)
  nG = 101
  phi = 0.0  # No in-plane polarization

  # Create a GRCWA object
  obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

  # Add vacuum layer
  obj.Add_LayerUniform(0.1, 1)

  # Add patterned layer(s) based on thickness list
  for layer_thickness in pthick:
    obj.Add_LayerGrid(layer_thickness, Nx, Ny)

  # Add another vacuum layer
  obj.Add_LayerUniform(0.1, 1)

  # Initialize the setup
  obj.Init_Setup()

  # Set permittivity grids for each layer
  obj.GridLayer_geteps(get_epgrid_nk(f, epgrid2, epgrid3, Nx, Ny))

  # Set plane wave excitation (adjust polarization as needed)
  planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
  obj.MakeExcitationPlanewave(planewave["p_amp"], planewave["p_phase"], planewave["s_amp"], planewave["s_phase"], order=0)

  # Solve for reflection and transmission coefficients
  R, T = obj.RT_Solve(normalize=1)

  # Return reflection, transmission, and optionally the GRCWA object
  return R, T, obj


def save_spectrum_data(
    xopt,
    f,
    anglesx,
    N_HOLES,
    ep1_diel,
    epbkg,
    epN,
    Nx,
    Ny,
    angles_filename="angles.txt",
    abs_filename="absorption.txt",
):
  """
  Calculates and saves absorption spectra data.

  Args:
      xopt: Optimized parameters from the optimization process.
      f: frequency at which we're calculating absorption
      anglesx: sampled incidence angle points (x axis) for plotting 
      N_HOLES: Number of holes in the structure.
      ep1_diel: Permittivity of the dielectric material.
      epbkg: Permittivity of the background material.
      epSbS: Permittivity of the antimony sulfide (SbS) material.
      epSiO: Permittivity of the silicon dioxide (SiO2) material.
      thick_sbs: Thickness of the SbS layer.
      thick_sio: Thickness of the SiO2 layer.
      Nx, Ny: Number of grid points in x and y directions.
      spectrum_filename: Filename to save the spectrum data (default: "spectrum.txt").
      abs_filename: Filename to save the absorption data (default: "absorption.txt").
  """

  L1 = [xopt[0], 0]
  L2 = [0, xopt[0]]
  pthick = [absorber_thickness, xopt[1], pec_thickness]

  radii = xopt[-N_HOLES:]  # Radii of cylinders
  c_x = xopt[2: 2 + N_HOLES]
  c_y = xopt[2 + N_HOLES: 2 + 2 * N_HOLES]
  centers = [(cx, cy) for cx in c_x for cy in c_y]

  epgrid2 = get_pattern_epgrid(radii, centers, L1, L2, ep1_diel, epbkg, Nx, Ny)
  epgrid3 = np.ones((Nx, Ny)) * epN

  absorptivity = []

  for theta in anglesx:
    R, T, obj = get_rt_nk(L1, L2, pthick, f, theta, epgrid2, epgrid3, Nx, Ny)
    a = get_layer_absorption(1, pthick[0], (L1[0] ** 2) * 1e-12, Nx, Ny, obj)
    absorptivity.append(a)

  absorptivity = np.array(absorptivity)

  # Save spectrum and absorption data
  np.savetxt(angles_filename, anglesx)
  np.savetxt(abs_filename, absorptivity)

  return


def plot_absorption_spectrum(angles_filename="angles.txt", abs_filename="absorption.txt", save_fig=True):
  """
  Plots the absorption spectrum data from saved files.

  Args:
      spectrum_filename: Filename containing the spectrum data (default: "spectrum.txt").
      abs_filename: Filename containing the absorption data (default: "absorption.txt").
      save_fig: Boolean flag to save the plot as an image (default: True).
  """

  anglesx = np.loadtxt(angles_filename)
  absorption = np.loadtxt(abs_filename)

  fig, ax = plt.subplots()  # Create figure and subplot
  ax = sns.lineplot(x=np.rad2deg(anglesx), y=absorption)
  ax.set_xlabel(r"Incidence Angle (in degrees)", fontsize=16)
  ax.set_ylabel("Absorption", fontsize=16)
  ax.grid(True)
  ax.tick_params(which="both", direction="in")

  plt.locator_params(axis="y", nbins=5)
  plt.locator_params(axis="x", nbins=8)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.tight_layout()

  if save_fig:
    plt.savefig("abs_plot.png")
  
  return fig, ax


def load_spectrum_data(spectrum_filename="spectrum.txt", abs_filename="absorption.txt"):
  """
  Loads spectrum and absorption data from text files.

  Args:
      spectrum_filename: Name of the text file containing spectrum data (default: "spectrum.txt").
      abs_filename: Name of the text file containing absorption data (default: "absorption.txt").

  Returns:
      tuple: A tuple containing two NumPy arrays - spectrum data and absorption data.
  """

  # Load spectrum data
  with open(spectrum_filename, "r") as f:
    spectrum_data = np.genfromtxt(f, delimiter=",")  # Assuming comma-separated data

  # Load absorption data
  with open(abs_filename, "r") as f:
    absorption_data = np.genfromtxt(f, delimiter=",")  # Assuming comma-separated data

  return spectrum_data, absorption_data





