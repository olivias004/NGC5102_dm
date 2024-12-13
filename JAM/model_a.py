# IMPORTS =====================
import numpy as np
import pickle
import emcee
from schwimmbad import MPIPool
from jampy.jam_axi_proj import jam_axi_proj
import sys

# ------------------ FUNCTIONS ----------------------

# Parameter boundaries
boundary = {
    "inc": [70, 90],        # Inclination (degrees)
    "beta": [-0.99, 0.99],  # Anisotropy parameter beta
    "mbh": [0.8, 1.2],      # Black hole mass scaling
    "ml": [0.5, 5.0],       # Mass-to-light ratio
}

# Parameter priors (Gaussian)
prior = {
    "inc": [80, 5],  # Mean 80, sigma 5
    "beta": [0.0, 0.5],  # Mean 0, sigma 0.5
    "mbh": [1.0, 0.1],  # Mean 1.0, sigma 0.1
    "ml": [2.0, 1.0],  # Mean 2.0, sigma 1.0
}

def check_boundary(pars):
    """Check if parameters are within boundaries."""
    keys = list(boundary.keys())
    for i, key in enumerate(keys):
        if not (boundary[key][0] <= pars[i] <= boundary[key][1]):
            return False
    return True

def lnprior(pars):
    """Compute Gaussian priors."""
    ln_prior = 0.0
    for i, (key, (mean, sigma)) in enumerate(prior.items()):
        ln_prior += -0.5 * ((pars[i] - mean) / sigma) ** 2
    return ln_prior

def jam_lnprob(pars):
    """Log-probability combining priors and likelihood."""
    # Check parameter boundaries
    if not check_boundary(pars):
        return -np.inf

    # Compute priors
    ln_prior_val = lnprior(pars)
    if not np.isfinite(ln_prior_val):
        return -np.inf

    # Unpack parameters
    inc, beta, mbh, ml = pars

    # Run JAM model
    try:
        jam = jam_axi_proj(
            d["surf_lum"],
            d["sigma_lum"],
            d["qObs_lum"],
            d["surf_pot"] * ml,
            d["sigma_pot"],
            d["qObs_pot"],
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
    except Exception:
        return -np.inf

    # Compute likelihood (Chi-squared normalized over good bins)
    n_good_bins = len(d["goodbins"])
    chi2 = -0.5 * np.sum((jam.model - d["rms"])**2 / d["erms"]**2) / n_good_bins

    # Combine prior and likelihood
    return ln_prior_val + chi2

def run_mcmc(output_path, ndim=4, nwalkers=20, nsteps=200):
    """Run MCMC sampling with emcee."""
    # Initialize walkers close to prior means
    p0 = [
        [prior["inc"][0], prior["beta"][0], prior["mbh"][0], prior["ml"][0]]
        + 1e-2 * np.random.randn(ndim)
        for _ in range(nwalkers)
    ]

    print("Starting MCMC...")
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    print("MCMC completed.")
    # Save sampler output
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

    return sampler

# ------------------ MAIN SCRIPT -------------------
if __name__ == "__main__":
    # CONSTANTS
    ndim = 4
    nwalkers = 20  # Increased walkers for better sampling
    nsteps = 200   # Small for testing, increase for convergence

    # Paths
    data_path = "/home/osilcock/DM_data/kwargs.pkl"
    output_path = "/fred/oz059/olivia/NGC5102_samples_fixed.pkl"

    # Load input data
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # Run MCMC
    sampler = run_mcmc(output_path, ndim, nwalkers, nsteps)

    # Summary of results
    print("MCMC completed. Saving results...")
    flat_samples = sampler.get_chain(discard=20, thin=5, flat=True)
    medians = np.median(flat_samples, axis=0)
    print(f"Median parameters: inc={medians[0]:.2f}, beta={medians[1]:.2f}, mbh={medians[2]:.2f}, ml={medians[3]:.2f}")
