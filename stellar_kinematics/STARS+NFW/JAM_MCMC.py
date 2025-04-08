# Purpose: NGC5102_JAM with Mitzkus Code Integration and 1D NFW Model
# Author: Olivia Silcock
# Date: Mar 2025

# IMPORTS ===================
import numpy as np
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
from mgefit.mge_fit_1d import mge_fit_1d
from schwimmbad import MPIPool
import sys

# FUNCTIONS =================
# Log-normal prior helper
def prior_log_normal(x, mu, sigma):
    if x <= 0:
        return -np.inf
    return -0.5 * ((np.log(x) - mu) / sigma) ** 2

def mge_pot(Rs, p0, arcsec_to_pc):
    r_max = max(500, 1.2 * Rs)  # Rs in arcsec
    r = np.logspace(np.log10(0.1), np.log10(r_max), 50)  # Log-spaced in arcsec

    r_parsec = r * arcsec_to_pc
    R = r / Rs  # Unitless

    intrinsic_density = p0 / (R * (1 + R)**2)

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

    surf = p.sol[0, :]
    sigma = p.sol[1, :] / arcsec_to_pc
    qobs = np.ones_like(surf)
    return surf, sigma, qobs

# JAM likelihood with NFW
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Bound checks
    if not (d['inc_bounds'][0] < inc < d['inc_bounds'][1]):
        return -np.inf
    if not (d['beta_bounds'][0] < beta < d['beta_bounds'][1]):
        return -np.inf
    if not (d['mbh_bounds'][0] < mbh < d['mbh_bounds'][1]):
        return -np.inf
    if not (d['ml_bounds'][0] < ml < d['ml_bounds'][1]):
        return -np.inf
    if not (d['Rs_bounds'][0] < Rs < d['Rs_bounds'][1]):
        return -np.inf
    if not (d['p0_bounds'][0] < p0 < d['p0_bounds'][1]):
        return -np.inf

    # # Log-normal priors for Rs and p0
    ln_prior = (
        prior_log_normal(Rs, mu=np.log(2000), sigma=0.5) +
        prior_log_normal(p0, mu=np.log(0.2), sigma=1.0)
    )

    if not np.isfinite(ln_prior):
        return -np.inf

    # DM potential component
    try:
        surf_dm, sigma_dm, qobs_dm = mge_pot(Rs, p0, d['arcsec_to_pc'])
        combined_surface_density = np.concatenate((d['surf_pot'], surf_dm))
        combined_sigma = np.concatenate((d['sigma_pot'], sigma_dm))
        combined_q = np.concatenate((d['qObs_pot'], qobs_dm))
    except Exception as e:
        print(f"Error in MGE construction: {e}")
        return -np.inf

    # Run JAM model
    jam_result = jam_axi_proj(
        d["surf_lum"],
        d["sigma_lum"],
        d["qObs_lum"],
        combined_surface_density,
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
        goodbins=d['goodbins'],
        beta=np.full_like(d["qObs_lum"], beta),
        data=d['rms'],
        errors=d['erms'],
        ml=ml
    )

    chi2 = -0.5 * jam_result.chi2 * len(d['rms'])
    return chi2  + ln_prior

# MCMC runner
def run_mcmc_nfw(output_path, ndim, nwalkers, nsteps):
    # Generate uniform initial positions for each parameter
    p0 = []
    for _ in range(nwalkers):
        inc  = np.random.uniform(*d['inc_bounds'])
        beta = np.random.uniform(*d['beta_bounds'])
        mbh  = np.random.uniform(*d['mbh_bounds'])
        ml   = np.random.uniform(*d['ml_bounds'])
        Rs   = np.random.uniform(*d['Rs_bounds'])
        p0dm = np.random.uniform(*d['p0_bounds'])  # renamed to avoid conflict with p0 (list of walkers)
        p0.append([inc, beta, mbh, ml, Rs, p0dm])

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print("Starting MCMC...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_nfw_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    print("Saving results...")
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# MAIN ======================
if __name__ == "__main__":
    output_path = "/fred/oz059/olivia/NFW_samples.pkl"
    ndim = 6
    nwalkers = 20        # Slightly more walkers = better exploration
    nsteps = 300


    with open("/home/osilcock/DM_NFW_data/kwargs.pkl", "rb") as f:
        d = pickle.load(f)

    print(f"Running EMCEE with {nsteps} steps, {nwalkers} walkers, {ndim} parameters.")
    run_mcmc_nfw(output_path, ndim, nwalkers, nsteps)
    print("MCMC sampling completed and saved!")
