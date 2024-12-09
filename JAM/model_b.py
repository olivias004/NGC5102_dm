# Script to run Model B with JAM and MPI
# Author: Adapted for Olivia Silcock's project
# Date: Dec 2023

# IMPORTS =====================
import numpy as np
import pickle
import emcee
from schwimmbad import MPIPool
from jampy.jam_axi_proj import jam_axi_proj
import sys


# FUNCTIONS ===================
# Parameter boundaries
def param(pars):
    inc, beta, mbh, logalpha = pars

    # Define boundaries
    inc_bounds = [70, 90]           # Inclination (degrees)
    beta_bounds = [-0.99, 0.99]     # Anisotropy parameter beta
    mbh_bounds = [0.8, 1.2]         # Black hole mass scaling
    logalpha_bounds = [-0.5, 0.5]   # IMF scaling parameter logalpha

    # Check if all parameters are within their respective boundaries
    if (inc_bounds[0] <= inc <= inc_bounds[1] and
        beta_bounds[0] <= beta <= beta_bounds[1] and
        mbh_bounds[0] <= mbh <= mbh_bounds[1] and
        logalpha_bounds[0] <= logalpha <= logalpha_bounds[1]):
        return True
    else:
        return False

# Priors
def priors(pars):
    """
    Gaussian priors for the parameters.
    """
    inc, beta, mbh, logalpha = pars

    # Define priors
    priors_dict = {
        "inc": [80, 5],         # Mean 80, sigma 5 for inclination
        "beta": [0.0, 0.5],     # Mean 0, sigma 0.5 for beta
        "mbh": [1.0, 0.1],      # Mean 1.0, sigma 0.1 for black hole scaling
        "logalpha": [0.0, 0.2], # Mean 0.0, sigma 0.2 for IMF scaling
    }

    ln_prior = 0.0
    for value, (mean, sigma) in zip(pars, priors_dict.values()):
        ln_prior += -0.5 * ((value - mean) / sigma) ** 2

    return ln_prior

# JAM likelihood function
def jam_lnprob(pars):
    """
    Combine priors and likelihood for the MCMC sampling.
    """
    inc, beta, mbh, logalpha = pars

    # Check parameter boundaries
    if not param(pars):
        return -np.inf

    # Compute priors
    ln_prior = priors(pars)
    if not np.isfinite(ln_prior):
        return -np.inf

    # Scale surface potential by IMF scaling
    surf_pot_scaled = d["surf_pot"] * 10 ** logalpha

    # Run JAM model
    jam = jam_axi_proj(
        surf_lum=d["surf_lum"],
        sigma_lum=d["sigma_lum"],
        qObs_lum=d["qObs_lum"],
        surf_pot=surf_pot_scaled,
        sigma_pot=d["sigma_pot"],
        qObs_pot=d["qObs_pot"],
        inc=inc,
        mbh=mbh * d["bhm"],
        dist=d["dist"],
        xbin=d["rot_x"],
        ybin=d["rot_y"],
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

    # Chi-squared computation
    chi2 = -0.5 * jam.chi2 * len(d["rms"])

    # Combine likelihood and priors
    return ln_prior + chi2

# MCMC Sampling Function
def run_mcmc(output_path, ndim=4, nwalkers=20, nsteps=5000):
    """
    Run MCMC sampling using emcee.
    Args:
        output_path (str): Filepath to save MCMC samples.
        ndim (int): Number of dimensions.
        nwalkers (int): Number of walkers.
        nsteps (int): Number of steps.
    """
    # Initialize walkers around a starting point
    p0 = [
        [80.0, 0.0, 1.0, 0.0] + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# MAIN EXECUTION ================================
if __name__ == "__main__":
    # CONSTANTS
    ndim = 4
    nwalkers = 5
    nsteps = 100

    # Paths
    data_path = "/home/osilcock/DM_data/kwargs.pkl"
    output_path = "/fred/oz059/olivia/NGC5102_samples_modelB.pkl"

    # Load input data
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # Run MCMC
    run_mcmc(output_path, ndim, nwalkers, nsteps)