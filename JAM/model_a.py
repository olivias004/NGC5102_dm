#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script to run Model A with JAM and MPI
# Author: Adapted for Olivia Silcock's project
# Date: Dec 2023

import numpy as np
import pickle
import emcee
from schwimmbad import MPIPool
import sys
from jampy.jam_axi_proj import jam_axi_proj


def jam_lnprob(params):
    """
    Log-probability function for MCMC, combining priors and likelihood.
    Args:
        params (list): Parameters [inclination, beta, M/L, MBH scaling factor].
    Returns:
        float: Log-probability.
    """
    try:
        inc, beta, mbh, ml = params

        # Validate boundaries
        if not (80 <= inc <= 90) or not (-0.99 <= beta <= 0.99) or not (0.8 <= mbh <= 1.2) or not (0.5 <= ml <= 5):
            return -np.inf

        # Convert inclination to cos(inclination)
        cosinc = np.cos(np.radians(inc))

        # Run JAM model
        jam = jam_axi_proj(
            d["surf_lum"],
            d["sigma_lum"],
            d["qObs_lum"],
            d["surf_lum"] * ml,  # Mass follows light
            d["sigma_lum"],
            d["qObs_lum"],
            cosinc,
            mbh * d["bhm"],  # Black hole mass scaling
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
    except Exception as e:
        print(f"Error in likelihood computation: {e}", file=sys.stderr)
        return -np.inf


# MCMC Sampling Function
def run_mcmc(output_path, ndim=4, nwalkers=20, nsteps=500):
    """
    Run MCMC sampling using emcee.
    Args:
        output_path (str): Filepath to save MCMC samples.
        ndim (int): Number of dimensions.
        nwalkers (int): Number of walkers.
        nsteps (int): Number of steps.
    """
    # Starting point for walkers
    p0 = [[87.5, 0.0, 1.0, 2.55] + 0.5 * np.random.randn(ndim) for _ in range(nwalkers)]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    # Save sampler object with samples
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)


if __name__ == "__main__":
    # Load data
    kwargs_file_path = "/home/osilcock/DM_data/kwargs.pkl"
    with open(kwargs_file_path, "rb") as f:
        d = pickle.load(f)

    # MCMC parameters
    output_path = "/fred/oz059/olivia/NGC5102_samples.pkl"
    nwalkers = 20
    nsteps = 500
    ndim = 4

    # Run MCMC
    print("Starting MCMC...")
    run_mcmc(output_path, ndim=ndim, nwalkers=nwalkers, nsteps=nsteps)
    print("MCMC completed. Results saved.")
