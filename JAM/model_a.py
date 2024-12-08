#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to run Model A with JAM (MPI-enabled for server)
# Author: Adapted for Olivia Silcock's project
# Date: Dec 2023

import numpy as np
import pickle
from emcee import EnsembleSampler
from emcee.utils import MPIPool
import time
from jampy.jam_axi_proj import jam_axi_proj

def load_data(pickle_file):
    """Load data from a pickle file."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data

def parameter_boundaries():
    """Define parameter boundaries for the MCMC."""
    return {
        "beta": (-0.99, 0.99),   # Beta bounds
        "cosinc": (np.cos(np.radians(90)), np.cos(np.radians(70))),  # Inclination bounds
        "ml": (0.5, 5.0),        # Mass-to-light ratio bounds
        "mbh": (0.8, 1.2)        # Black hole mass scaling bounds
    }

def priors(params, boundaries):
    """Gaussian priors for MCMC parameters."""
    priors_dict = {
        "beta": [0.0, 0.5],     # Mean and sigma for beta
        "cosinc": [np.mean(boundaries["cosinc"]), 0.1],  # Centered inclination prior
        "ml": [2.0, 1.0],       # Mean and sigma for M/L
        "mbh": [1.0, 0.1]       # Mean and sigma for mbh scaling
    }
    ln_prior = 0.0

    for key, value in params.items():
        prior_mean, prior_sigma = priors_dict[key]
        if boundaries[key][0] <= value <= boundaries[key][1]:
            ln_prior += -0.5 * ((value - prior_mean) / prior_sigma) ** 2
        else:
            return -np.inf  # Reject sample if outside boundaries

    return ln_prior

def log_likelihood(params, d, boundaries):
    """Log-likelihood function for the JAM model."""
    beta, cosinc, ml, mbh = params
    inc = np.arccos(cosinc)  # Convert cosinc back to inclination

    # Validate axial ratio constraints
    qmin = 0.05
    q_obs_min = np.sqrt(np.cos(inc) ** 2 + (qmin * np.sin(inc)) ** 2)
    if not np.all(d["qObs_lum"] >= q_obs_min):
        return -np.inf  # Reject sample if inclination is too low

    # Run the JAM model
    jam = jam_axi_proj(
        d["surf_lum"],
        d["sigma_lum"],
        d["qObs_lum"],
        d["surf_lum"] * ml,  # Mass follows light
        d["sigma_lum"],
        d["qObs_lum"],
        inc,
        mbh * d["bhm"],  # Scaled black hole mass
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

    # Compute chi-squared
    chi2 = jam.chi2 * np.sum(d["goodbins"])
    return -0.5 * chi2

def log_probability(params, d, boundaries):
    """Log-probability function combining priors and likelihood."""
    param_dict = {
        "beta": params[0],
        "cosinc": params[1],
        "ml": params[2],
        "mbh": params[3],
    }
    ln_prior = priors(param_dict, boundaries)
    if not np.isfinite(ln_prior):
        return -np.inf
    return ln_prior + log_likelihood(params, d, boundaries)

def run_mcmc(pickle_file, n_walkers=32, n_steps=1000, burnin=200):
    """Run MCMC for Model A with MPI parallelization."""
    # Load data
    d = load_data(pickle_file)
    boundaries = parameter_boundaries()

    # Initialize walkers near the mean values
    initial_guess = [
        0.0,                            # beta
        np.mean(boundaries["cosinc"]),  # cosinc
        2.0,                            # ml
        1.0                             # mbh
    ]
    ndim = len(initial_guess)
    p0 = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)

    # Set up MPI pool
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        return

    # Set up the MCMC sampler
    sampler = EnsembleSampler(
        n_walkers, ndim, log_probability, args=(d, boundaries), pool=pool
    )

    # Burn-in phase
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burnin, progress=True)
    sampler.reset()

    # Production phase
    print("Running production...")
    sampler.run_mcmc(p0, n_steps, progress=True)

    # Close the MPI pool
    pool.close()

    return sampler.get_chain(flat=True)

def plot_mcmc(samples):
    """Plot MCMC results."""
    import matplotlib.pyplot as plt
    import corner

    labels = ["Beta", "CosInc", "M/L", "Mbh"]
    fig = corner.corner(samples, labels=labels, show_titles=True)
    plt.show()

if __name__ == "__main__":
    # Path to the pickle file
    pickle_file_path = "/path/to/kwargs.pkl"  # Update with your actual path

    # Run MCMC with reduced steps for testing
    n_walkers = 16
    n_steps = 500
    burnin = 100

    print("Starting MCMC for Model A with MPI...")
    samples = run_mcmc(pickle_file_path, n_walkers=n_walkers, n_steps=n_steps, burnin=burnin)
    print("MCMC complete. Plotting results...")
    plot_mcmc(samples)
