# Purpose: NGC5102_JAM with Mitzkus Code Integration and 1D NFW Model
# Author: Olivia Silcock
# Date: Mar 2025

# IMPORTS ===================
import numpy as np
import emcee
import pickle
from jampy.jam_axi_proj import jam_axi_proj
from schwimmbad import MPIPool
import sys

# FUNCTIONS =================
# mge pot
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

# Define the likelihood function including the NFW component
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    # Check parameter bounds
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

    # Compute DM MGE component
    surf_dm, sigma_dm, qobs_dm = mge_pot(Rs, p0, arcsec_to_pc)

    # Combine with stellar components
    try:
        combined_surface_density = np.concatenate((d['surf_pot'], surf_dm))
        combined_sigma = np.concatenate((d['sigma_pot'], sigma_dm))
        combined_q = np.concatenate((d['qObs_pot'], qobs_dm))

    except NameError:
        print("Data variables not defined.")
        return np.inf

    # Run JAM model with ml set
    jam_result = jam_axi_proj(
       		d["surf_lum"], 
       		d["sigma_lum"], 
       		d["qObs_lum"], 
       		combined_surface_density * ml,
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
        	ml=1
    )
    
    chi2 = -0.5 * jam_result.chi2 * len(d['rms'])
    return chi2

# Function to run MCMC
def run_mcmc_nfw(output_path, ndim, nwalkers, nsteps):
    # Starting parameter guesses
    p0 = [
        [88, -0.1, 1.0, 3.3, 500, 0.5] + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)
    ]

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, jam_nfw_lnprob, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    # Save the results
    with open(output_path, "wb") as f:
        pickle.dump(sampler, f)

# Main execution block
if __name__ == "__main__":
    output_path = "/fred/oz059/olivia/samples.pkl"
    ndim = 6  # incl, beta, mbh, ml, Rs, p0
    nwalkers = 20
    nsteps = 100

    # Load your data
    with open("/home/osilcock/DM_NFW_data/kwargs.pkl", "rb") as f:
        d = pickle.load(f)
        
    # Run MCMC
    run_mcmc_nfw(output_path, ndim, nwalkers, nsteps)
    print("MCMC sampling completed and saved!")
