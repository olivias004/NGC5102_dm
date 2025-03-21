# TITLE: Testing Manual Values
# AUTHOR: Olivia Silcock
# DATE: March 2025

# IMPORTS ===================
import numpy as np
import matplotlib.pyplot as plt
from mgefit.mge_fit_1d import mge_fit_1d
from jampy.jam_axi_proj import jam_axi_proj
import pickle

# VARIABLES =================
inc = 88.92104007844380  # Inclination in degrees
beta = -0.09046500178482920  # Anisotropy parameter β
mbh = 0.9647188135357420 * ((1.9*(50/200)**5.1)*1e8)  # Black hole mass (in solar masses)
ml = 3.28816707274668  # Mass-to-light ratio
arcsec_to_pc = 4e6 / 206265  # Conversion factor from arcseconds to parsecs

# Define ranges for Rs and p0
Rs_range = (100, 2000, 15)  # Number of points instead of step size
p0_range = (0.001, 2.0, 15)  # Number of points as an integer


# FUNCTIONS =================
def mge_pot(Rs, p0, arcsec_to_pc):
    # Ensure max(r) is at least Rs
    r_max = max(500, 1.2 * Rs)  # Scale dynamically with Rs
    r = np.linspace(0.1, r_max, 50)  # Linear scale

    # Convert radius to parsecs
    r_parsec = r * arcsec_to_pc

    # Intrinsic density - M_sun/pc^3
    R = r_parsec / Rs
    intrinsic_density = p0 / (R * ((1 + R) ** 2))

    # 1D MGE process
    p = mge_fit_1d(
        r_parsec, intrinsic_density,
        negative=False,
        ngauss=11,
        rbounds=None,
        inner_slope=1,
        outer_slope=3,
        quiet=False,
        plot=True
    )

    # Outputs
    surf = p.sol[0, :]
    sigma = p.sol[1, :]

    # Convert sigma to arcsec
    sigma = sigma / arcsec_to_pc

    qobs = np.ones_like(surf)
    return surf, sigma, qobs


# Likelihood function for JAM + NFW model
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Compute DM MGE component
    DM_surface_density, DM_sigma_arcsec, DM_q = mge_pot(Rs, p0, arcsec_to_pc)

    # Ensure necessary data variables exist
    try:
        combined_surface_density = np.concatenate((d['surf_pot'], DM_surface_density))
        combined_sigma = np.concatenate((d['sigma_pot'], DM_sigma_arcsec))
        combined_q = np.concatenate((d['qObs_pot'], DM_q))
    except NameError:
        print("Error: Ensure `surf_pot`, `sigma_pot`, and `qObs_pot` are defined before running.")
        return np.inf

    # Run JAM model with ml set
    jam = jam_axi_proj(
        d["surf_lum"],
        d["sigma_lum"],
        d["qObs_lum"],
        combined_surface_density * ml,
        combined_sigma,
        combined_q,
        inc,
        mbh * d["bhm"],  # Scaling MBH correctly
        d["dist"],
        d["rot_x"],
        d["rot_y"],
        align="cyl",
        moment="zz",
        plot=False,
        pixsize=d["pixsize"],
        quiet=1,
        sigmapsf=d["sigmapsf"],
        normpsf=d["normpsf"],
        goodbins=d['goodbins'],  # Ensure only good bins are used
        beta=np.full_like(d["qObs_lum"], beta),
        data=d['rms'],  # Only use good bins
        errors=d['erms'],  # Only use good bins
        ml=1 
    )

    chi2 = -0.5 * jam.chi2 * d['goodbins'].sum()  # Use goodbins.sum() instead of len(rms)
    return chi2


def compute_chi2_grid():
    """
    Computes chi2 values for a grid of (Rs, p0) values and stores them in a 2D array.
    """
    Rs_values = np.linspace(*Rs_range)
    p0_values = np.linspace(*p0_range)
    chi2_grid = np.zeros((len(p0_values), len(Rs_values)))
    
    for i, p0 in enumerate(p0_values):
        for j, Rs in enumerate(Rs_values):
            pars = [inc, beta, mbh, ml, Rs, p0]
            chi2_grid[i, j] = -2 * jam_nfw_lnprob(pars)  # Convert log-likelihood to chi2
    
    return Rs_values, p0_values, chi2_grid


def plot_chi2_grid(Rs_values, p0_values, chi2_grid):
    """
    Plots a color-coded chi2 grid for (Rs, p0) parameter space.
    """
    plt.figure(figsize=(8, 6))
    plt.contourf(Rs_values, p0_values, np.log10(chi2_grid), levels=20, cmap='RdBu_r')
    plt.colorbar(label='log_10 χ²')
    plt.xlabel('$r_0$ (arcseconds)')
    plt.ylabel('$\\rho_0$ ($M_\\odot$/arcsec$^2$)')
    plt.title('Delta χ² Fitting DM Only')
    plt.show()


if __name__ == "__main__":
    # Load the dictionary from the saved pickle file
    with open("/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_NFW/kwargs.pkl", "rb") as f:
        d = pickle.load(f)
    
    Rs_values, p0_values, chi2_grid = compute_chi2_grid()
    plot_chi2_grid(Rs_values, p0_values, chi2_grid)
