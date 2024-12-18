# IMPORTS
import numpy as np
import pickle
import emcee
import matplotlib.pyplot as plt
from jampy.jam_axi_proj import jam_axi_proj
from time import time

# FUNCTIONS ===================
# Parameter boundaries
def param(pars):
    inc, beta, mbh, ml = pars
    inc_bounds = [70, 90]       # Inclination (degrees)
    beta_bounds = [-0.5, 0.5]   # Adjusted beta bounds
    mbh_bounds = [0.8, 1.2]     # Black hole mass scaling
    ml_bounds = [0.5, 5.0]      # Mass-to-light ratio

    if (inc_bounds[0] <= inc <= inc_bounds[1] and
        beta_bounds[0] <= beta <= beta_bounds[1] and
        mbh_bounds[0] <= mbh <= mbh_bounds[1] and
        ml_bounds[0] <= ml <= ml_bounds[1]):
        return True
    return False

# Priors
def priors(pars):
    inc, beta, mbh, ml = pars
    priors_dict = {"inc": [80, 5], "beta": [0.0, 0.2], "mbh": [1.0, 0.1], "ml": [2.0, 0.5]}
    ln_prior = sum(-0.5 * ((value - mean) / sigma) ** 2 for value, (mean, sigma) in zip(pars, priors_dict.values()))
    return ln_prior

# JAM likelihood function
def jam_lnprob(pars):
    if not param(pars): return -np.inf
    inc, beta, mbh, ml = pars
    ln_prior = priors(pars)
    if not np.isfinite(ln_prior): return -np.inf

    try:
        # Run JAM model
        jam = jam_axi_proj(
            d["surf_lum"], d["sigma_lum"], d["qObs_lum"], 
            d["surf_pot"] * ml, d["sigma_pot"], d["qObs_pot"],
            inc, mbh * d["bhm"], d["dist"], d["rot_x"], d["rot_y"],
            align="cyl", moment="zz", plot=False, pixsize=d["pixsize"],
            quiet=1, sigmapsf=d["sigmapsf"], normpsf=d["normpsf"], 
            goodbins=d["goodbins"], beta=np.full_like(d["qObs_lum"], beta),
            data=d["rms"], errors=d["erms"]
        )

        if not np.isfinite(jam.chi2):
            return -np.inf  # Safeguard against bad outputs
    except Exception as e:
        print(f"JAM failed: {e}")
        return -np.inf

    chi2 = -0.5 * jam.chi2 * len(d["rms"])
    return ln_prior + chi2

# MAIN ===========================
# ------------------ MAIN SCRIPT -------------------
if __name__ == "__main__":
    # CONSTANTS
    ndim = 4
    nwalkers = 10  # Low number for testing
    nsteps = 100   # Low number for testing

    # Paths
    data_path = "/home/osilcock/DM_data/kwargs.pkl"
    output_path = "/fred/oz059/olivia/NGC5102_samples_test.pkl"

    # Load input data
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # Run MCMC
    sampler = run_mcmc(output_path, ndim, nwalkers, nsteps)

    # Print summary of results
    print("MCMC completed. Chain shape:", sampler.chain.shape)
    print("Log-probability shape:", sampler.lnprobability.shape)

    # Extract and print median parameters
    flat_samples = sampler.get_chain(discard=10, thin=5, flat=True)
    medians = np.median(flat_samples, axis=0)
    print(f"Median parameters: inc={medians[0]:.2f}, beta={medians[1]:.2f}, mbh={medians[2]:.2f}, ml={medians[3]:.2f}")



