
# Name: JAM - NFW process
# Author: Olivia Silcock
# Date: February 2025



#IMPORTS=====================
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from pylab import figure, cm
import copy


import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from jampy.mge_radial_mass import mge_radial_mass

from astropy.coordinates import Angle
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.models import Ellipse2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.visualization import (ZScaleInterval,MinMaxInterval,SqrtStretch,ImageNormalize)

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (detect_sources,
                                    make_2dgaussian_kernel)
from photutils.segmentation import SourceCatalog

from scipy import special


from mgefit.mge_fit_1d import mge_fit_1d

# NFW =======================
M200 = 5e9  # Msun (Smaller halo mass)
R200_arcsec = 500  # Set directly in arcseconds (consistent with plot)
c = 6  # Shallower concentration

# Compute Rs in arcseconds directly
Rs_arcsec = R200_arcsec / c  # This ensures the scale radius is within 500 arcsec

# Compute rho_0 using the standard NFW formula
rho_0 = M200 / (4 * np.pi * Rs_arcsec**3 * (np.log(1 + c) - c / (1 + c)))

# Ensure NFW_radius does not start at exactly 0
NFW_radius = np.linspace(1e-2, 500, 1000)  # Avoids division by zero

# Compute NFW density profile in Msun/pc^3
DM_profile = rho_0 / ((NFW_radius / Rs_arcsec) * (1 + NFW_radius / Rs_arcsec)**2)

# Convert to surface density in Msun/pc^2
DM_surface_density = DM_profile * Rs_arcsec  # Ensure proper scaling

# Plot to verify correct scaling
plt.figure(figsize=(8, 6))
plt.plot(NFW_radius, DM_profile, label="NFW Density (Msun/pc^3)")
plt.plot(NFW_radius, DM_surface_density, label="NFW Surface Density (Msun/pc^2)")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Density")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()


# Now fit the MGE model safely
p = mge_fit_1d(
    NFW_radius, DM_profile, 
    negative=False, 
    ngauss=12,  
    rbounds=None, 
    inner_slope=1, 
    outer_slope=3, 
    quiet=False, 
    plot=True
)

plt.show()



# Extract fitted parameters from the NFW MGE fit
DM_TotalCounts = p.sol[0, :]  # Total counts for NFW Gaussians
DM_sigma = p.sol[1, :]  # Dispersion (in arcseconds, if input was arcseconds)

# Convert `TotalCounts` to surface density (Msun/pc^2)
DM_surface_density = DM_TotalCounts / (np.sqrt(2 * np.pi) * DM_sigma)  # Msun/pc^2

# Convert sigma to arcseconds (if needed)
DM_sigma_arcsec = DM_sigma * HST_SCALE  # Convert to arcseconds if necessary

# Axis ratio q for dark matter: assume spherical (q ≈ 1)
DM_q = np.ones_like(DM_sigma)

# CONCATENATION =============
# Combine the stellar and dark matter MGE components
combined_surface_density = np.concatenate((surf_total, DM_surface_density))  # Surface density
combined_sigma = np.concatenate((sigma_total, DM_sigma_arcsec))  # Sigma in arcseconds
combined_q = np.concatenate((qobs_total, DM_q))  # Axis ratios


# Compute mass profiles for stellar, dark matter, and combined MGE models
rad = np.linspace(0, 500, 1000) 
dist = 4  # Distance in Mpc
inc = 87.5  # Inclination angle

# Original mass profiles
stellar_mass = mge_radial_mass(surf_total, sigma_total, qobs_total, inc, rad, dist)
dm_mass = mge_radial_mass(DM_surface_density, DM_sigma_arcsec, DM_q, inc, rad, dist)
combined_mass = mge_radial_mass(combined_surface_density, combined_sigma, combined_q, inc, rad, dist)

# Plot comparison of mass profiles
plt.figure(figsize=(8, 6))
plt.scatter(rad, stellar_mass, label='Stellar MGE Mass', color='red')
plt.scatter(rad, dm_mass, label='Dark Matter MGE Mass', color='blue')
plt.scatter(rad, combined_mass, label='Total (Stellar + DM) MGE Mass', color='black', marker='+')
plt.legend()
plt.xlabel('Radius (arcseconds)')
plt.ylabel('Mass (Msun)')
plt.title("Comparison of Stellar, Dark Matter, and Combined MGE Mass Profiles")
plt.yscale("log")
plt.show()























#!/usr/bin/env python
# Name: JAM - NFW process (with free Rs and p0)
# Author: Olivia Silcock
# Date: February 2025

# IMPORTS ======================
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pickle
import sys
import copy

import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from jampy.jam_axi_proj import jam_axi_proj
from jampy.mge_radial_mass import mge_radial_mass

from astropy.coordinates import Angle
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.models import Ellipse2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.visualization import (ZScaleInterval, MinMaxInterval, SqrtStretch, ImageNormalize)

from photutils.datasets import make_100gaussians_image
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel)
from photutils.segmentation import SourceCatalog

from scipy import special

# For MCMC sampling with MPI
import emcee
from schwimmbad import MPIPool

# FUNCTIONS =================
# Parameter boundaries
def param(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Define boundaries (example values; adjust as needed)
    inc_bounds = [70, 90]       # Inclination (degrees)
    beta_bounds = [-0.99, 0.99] # Anisotropy parameter beta
    mbh_bounds = [0.8, 1.2]     # Black hole mass scaling
    ml_bounds = [0.5, 5.0]      # Mass-to-light ratio
    Rs_bounds = [10, 300]       # Scale radius (arcseconds) boundaries (example)
    p0_bounds = [1e5, 1e9]       # Normalization of NFW profile (units consistent with your data)

    if (inc_bounds[0] <= inc <= inc_bounds[1] and
        beta_bounds[0] <= beta <= beta_bounds[1] and
        mbh_bounds[0] <= mbh <= mbh_bounds[1] and
        ml_bounds[0] <= ml <= ml_bounds[1] and
        Rs_bounds[0] <= Rs <= Rs_bounds[1] and
        p0_bounds[0] <= p0 <= p0_bounds[1]):
        return True
    else:
        return False

# Priors
def priors(pars):
    """
    Gaussian priors for the parameters.
    """
    inc, beta, mbh, ml, Rs, p0 = pars

    # Define priors (example means and sigmas; adjust as needed)
    priors_dict = {
        "inc": [80, 5],     # Mean 80, sigma 5 for inclination
        "beta": [0.0, 0.5],   # Mean 0, sigma 0.5 for beta
        "mbh": [1.0, 0.1],    # Mean 1.0, sigma 0.1 for black hole scaling
        "ml": [2.0, 1.0],     # Mean 2.0, sigma 1.0 for M/L
        "Rs": [100, 50],      # Example: mean 100 arcsec, sigma 50
        "p0": [1e7, 5e6]      # Example: mean 1e7, sigma 5e6 (units as appropriate)
    }

    ln_prior = 0.0
    for value, (mean, sigma) in zip(pars, priors_dict.values()):
        ln_prior += -0.5 * ((value - mean) / sigma) ** 2

    return ln_prior

# Updated mge_pot function to compute the DM component
def mge_pot(Rs, p0):
    """
    Compute the dark matter NFW density profile and fit an MGE model.
    Returns:
       DM_surface_density: fitted DM MGE amplitudes (Msun/pc^2)
       DM_sigma_arcsec: fitted DM MGE dispersions (arcsec)
       DM_q: fitted DM MGE axis ratios
    """
    # Here we assume d['NFW_radius'] is provided (radii in arcsec) in the input data.
    r = d['NFW_radius']
    # Standard NFW density profile:
    DM_profile = p0 / ((r / Rs) * (1 + r / Rs)**2)
    
    # A rough conversion to surface density (this may require refinement)
    DM_surface_density_init = DM_profile * Rs  
    
    # Fit the MGE model to the DM profile
    p = mge_fit_1d(
        r, DM_profile, 
        negative=False, 
        ngauss=12,  
        rbounds=None, 
        inner_slope=1, 
        outer_slope=3, 
        quiet=False, 
        plot=True
    )
    
    # Extract fitted parameters
    DM_TotalCounts = p.sol[0, :]  # Amplitudes for the NFW Gaussians
    DM_sigma = p.sol[1, :]        # Dispersions (in same units as r)
    # Recompute surface density for each Gaussian
    DM_surface_density = DM_TotalCounts / (np.sqrt(2 * np.pi) * DM_sigma)
    # Convert dispersions to arcseconds if needed (assume d contains HST_SCALE)
    if 'HST_SCALE' in d:
        DM_sigma_arcsec = DM_sigma * d['HST_SCALE']
    else:
        DM_sigma_arcsec = DM_sigma
    DM_q = np.ones_like(DM_sigma)  # Assume circular Gaussians for DM

    return DM_surface_density, DM_sigma_arcsec, DM_q

# Likelihood function for JAM + NFW model
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Check parameter boundaries
    if not param(pars):
        return -np.inf

    # Evaluate priors
    ln_prior = priors(pars)
    if not np.isfinite(ln_prior):
        return -np.inf

    # Compute dark matter MGE component using the free parameters Rs and p0
    DM_surface_density, DM_sigma_arcsec, DM_q = mge_pot(Rs, p0)
    
    # Concatenate the stellar and dark matter MGE components
    combined_surface_density = np.concatenate((d['surf_pot'], DM_surface_density))
    combined_sigma = np.concatenate((d['sigma_pot'], DM_sigma_arcsec))
    combined_q = np.concatenate((d['qObs_pot'], DM_q))

    # Run the JAM model with the combined potential
    jam = jam_axi_proj(
        d["surf_lum"],
        d["sigma_lum"],
        d["qObs_lum"],
        combined_surface_density * ml,  # Scale the combined potential
        combined_sigma,
        combined_q,
        inc,
        mbh * d["bhm"],
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
        goodbins=d["goodbins"],
        beta=np.full_like(d["qObs_lum"], beta),
        data=d["rms"],
        errors=d["erms"],
    )

    # Compute chi-squared (here assumed scaled by the number of data points)
    chi2 = -0.5 * jam.chi2 * len(d["rms"])
    return ln_prior + chi2

# MCMC Sampling Function
def run_mcmc(output_path, ndim=6, nwalkers=20, nsteps=10000):
    """
    Run MCMC sampling using emcee for the full JAM + NFW model.
    Args:
        output_path (str): Filepath to save MCMC samples.
        ndim (int): Number of dimensions (here 6).
        nwalkers (int): Number of walkers.
        nsteps (int): Number of steps.
    """
    # Initial guess for parameters: [inc, beta, mbh, ml, Rs, p0]
    initial = np.array([80.0, 0.0, 1.0, 2.0, 100.0, 1e7])
    p0_init = [initial + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_nfw_lnprob, pool=pool)
        sampler.run_mcmc(p0_init, nsteps, progress=True)

    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# MAIN EXECUTION ================================
if __name__ == "__main__":
    # Set number of dimensions, walkers, and steps
    ndim = 6
    nwalkers = 20
    nsteps = 10000

    # Paths to data and output
    data_path = "/home/osilcock/DM_data/kwargs.pkl"
    output_path = "/fred/oz059/olivia/samples.pkl"

    # Load input data (this should provide at least keys such as 'surf_pot', 'sigma_pot',
    # 'qObs_pot', 'surf_lum', 'sigma_lum', 'qObs_lum', 'bhm', 'dist', 'rot_x', 'rot_y',
    # 'pixsize', 'sigmapsf', 'normpsf', 'goodbins', 'rms', 'erms', 'NFW_radius', etc.)
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # Run the MCMC sampling
    run_mcmc(output_path, ndim, nwalkers, nsteps)
