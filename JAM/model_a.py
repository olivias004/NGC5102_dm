#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to run Model A with JAM and MPI
# Author: Adapted for Olivia Silcock's project
# Date: Dec 2023

import numpy as np
import pickle
import emcee
from schwimmbad import MPIPool
from jampy.jam_axi_proj import jam_axi_proj
import sys

# Minimum axial ratio
QMIN = 0.05


def load_data(pickle_file):
    """Load data from a pickle file."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def parameter_boundaries():
    """Define parameter boundaries for the MCMC."""
    return {
        "beta": (-0.99, 0.99),  # Beta bounds
        "cosinc": (np.cos(np.radians(90)), np.cos(np.radians(70))),  # Inclination bounds
        "ml": (0.5, 5.0),  # Mass-to-light ratio bounds
        "mbh": (0.8, 1.2),  # Black hole mass scaling bounds
    }


def priors(params, boundaries):
    """
    Gaussian priors for MCMC parameters.
    Args:
        params (dict): Dictionary of parameters.
        boundaries (dict): Dictionary of parameter boundaries.
    Returns:
        float: Log-prior probability.
    """
    priors_dict = {
        "beta": [0.0, 0.5],  # Mean and sigma for beta
        "cosinc": [np.mean(boundaries["cosinc"]), 0.1],  # Centered inclination prior
        "ml": [2.0, 1.0],  # Mean and sigma for M/L
        "mbh": [1.0, 0.1],  # Mean and sigma for mbh scaling
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
    """
    Log-likelihood function for the JAM model.
    Args:
        params (list): Parameters [beta, cosinc, ml, mbh].
        d (dict): Data dictionary.
        boundaries (dict): Parameter boundaries.
    Returns:
        float: Log-likelihood value.
    """
    beta, cosinc, ml, mbh = params
    inc = np.arccos(cosinc)  # Convert cosinc back to inclination

    # Validate axial ratio constraints
    q_obs_min = np.sqrt(np.cos(inc) ** 2 + (QMIN * np.sin(inc)) ** 2)
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
        pixsize=d["pixsize"],
        sigmapsf=d["sigmapsf"],
        normpsf=d["normpsf"],
        goodbins=d["goodbins"],
        beta=np.full_like(d["qObs_lum"], beta),
        data=d["rms"],
        errors=d["erms"],
        quiet=1,
        plot=False,
    )

    chi2 = jam.chi2 * np.sum(d["goodbins"])
    return -0.5 * chi2


def log_probability(params, d, boundaries):
    """
    Log-probability function combining priors and likelihood.
    Args:
        params (list): Parameters [beta, cosinc, ml, mbh].
        d (dict): Data dictionary.
        boundaries (dict): Parameter boundaries.
    Returns:
        float: Log-probability.
    """
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


def run_mcmc(output_path, d, n_walkers=32, n_steps=500, burnin=100):
    """
    Run MCMC sampling for Model A with MPI.
    Args:
        output_path (str): Path to save the samples.
        d (dict): Data dictionary.
        n_walkers (int): Number of walkers.
        n_steps (int): Number of steps.
        burnin (int): Burn-in steps.
    """
    boundaries = parameter_boundaries()

    # Initialize walkers near the mean values
    initial_guess = [
        0.0,  # beta
        np.mean(boundaries["cosinc"]),  # cosinc
        2.0,  # ml
        1.0,  # mbh
    ]
    ndim = len(initial_guess)
    p0 = initial_guess + 1e-4 * np.random.randn(n_walkers, ndim)

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            return

        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_probability, args=(d, boundaries), pool=pool
        )

        # Burn-in phase
        p0, _, _ = sampler.run_mcmc(p0, burnin, progress=True)
        sampler.reset()

        # Production phase
        sampler.run_mcmc(p0, n_steps, progress=True)

    # Save the samples and metadata to a pickle file
    samples = sampler.get_chain(flat=True)
    metadata = {
        "parameters": ["beta", "cosinc", "ml", "mbh"],
        "burnin": burnin,
        "n_walkers": n_walkers,
        "n_steps": n_steps,
    }
    with open(output_path, "wb") as f:
        pickle.dump({"samples": samples, "metadata": metadata}, f)


if __name__ == "__main__":
    # Load data
    kwargs_file_path = "/home/osilcock/DM_data/kwargs.pkl"
    d = load_data(kwargs_file_path)

    # Run MCMC
    output_path = "/fred/oz059/olivia/NGC5102_samples.pkl"
    run_mcmc(output_path, d, n_walkers=8, n_steps=100, burnin=50)
