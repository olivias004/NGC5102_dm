# Purpose: NGC5102_JAM with Mitzkus Code Integration
# Author: Olivia Silcock
# Date: Dec 2023

# IMPORTS
import numpy as np
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
from schwimmbad import MPIPool
import sys

# FUNCTIONS ========================================

# Check parameter boundaries and priors
def check_params(pars, d):
    """
    Check whether the parameters are within defined boundaries and compute priors.
    Args:
        pars (list): [inclination, anisotropy (beta), black hole mass, mass-to-light ratio]
        d (dict): Dictionary containing bounds and priors.
    Returns:
        tuple: (log_prior, within_bounds)
            log_prior (float): Logarithmic prior probability.
            within_bounds (bool): Whether the parameters are within bounds.
    """
    inc, beta, mbh, ml = pars

    # Check boundaries
    within_bounds = (
        d['inc_bounds'][0] <= inc <= d['inc_bounds'][1] and
        d['beta_bounds'][0] <= beta <= d['beta_bounds'][1] and
        d['mbh_bounds'][0] <= mbh <= d['mbh_bounds'][1] and
        d['ml_bounds'][0] <= ml <= d['ml_bounds'][1]
    )
    
    if not within_bounds:
        return -np.inf, False

    # Compute priors (Gaussian priors for beta, mbh, ml)
    prior_inc = 0  # Uniform prior for inclination
    prior_beta = -0.5 * ((beta - 0.1) / 0.1) ** 2  # Example Gaussian prior for beta
    prior_mbh = -0.5 * ((mbh - 1.0) / 0.2) ** 2  # Gaussian prior for black hole mass
    prior_ml = -0.5 * ((ml - 2.55) / 0.5) ** 2  # Gaussian prior for M/L_dyn

    log_prior = prior_inc + prior_beta + prior_mbh + prior_ml
    return log_prior, True


# Log-probability function for MCMC
def jam_lnprob(pars):
    """
    Compute the log-probability for the MCMC process.
    Args:
        pars (list): [inclination, anisotropy (beta), black hole mass, mass-to-light ratio]
    Returns:
        float: Log-probability (chi2).
    """
    # Check parameters and compute priors
    log_prior, within_bounds = check_params(pars, d)
    if not within_bounds:
        return -np.inf

    inc, beta, mbh, ml = pars

    # JAM model computation
    jam = jam_axi_proj(
        surf_lum=d['surf_lum'],
        sigma_lum=d['sigma_lum'],
        qObs_lum=d['qObs_lum'],
        surf_pot=d['surf_pot'] * ml,
        sigma_pot=d['sigma_pot'],
        qObs_pot=d['qObs_pot'],
        inc=np.radians(inc),  # Convert to radians for JAM
        mbh=mbh * d['bhm'],   # Scale black hole mass
        dist=d['dist'],
        xbin=d['rot_x'],
        ybin=d['rot_y'],
        align="cyl",
        moment="zz",
        plot=False,
        pixsize=d['pixsize'],
        quiet=1,
        sigmapsf=d['sigmapsf'],
        normpsf=d['normpsf'],
        goodbins=d['goodbins'],
        beta=np.full_like(d['qObs_lum'], beta),
        data=d['rms'],
        errors=d['erms'],
    )
    
    # Chi-squared computation
    chi2 = -0.5 * jam.chi2 * len(d['rms'])
    return log_prior + chi2


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
        [87.5, 0.0, 1.0, 2.55] + 0.5 * np.random.randn(ndim) for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)


# MAIN EXECUTION ===================================
if __name__ == "__main__":
    # CONSTANTS
    location = "local"  # Change to "server" if running on the server
    ndim = 4
    nwalkers = 20
    nsteps = 45

    # FILE PATHS
    if location == "server":
        data_path = "/home/osilcock/DM_data/kwargs.pkl"
        output_path = "/fred/oz059/olivia/NGC5102_samples.pkl"
    else:
        data_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/files/final_JAM/kwargs.pkl"
        output_path = "/Users/livisilcock/Documents/PROJECTS/NGC5102/files/final_JAM/NGC5102_samples.pkl"

    # LOAD INPUT DATA
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # RUN MCMC
    run_mcmc(output_path, ndim, nwalkers, nsteps)
    print("MCMC completed and saved!")
