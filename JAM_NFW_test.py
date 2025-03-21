# TITLE: Jam: NFW & Stars
# AUTHOR: Olivia Silcock

# IMPORTS ===================
import numpy as np
from mgefit.mge_fit_1d import mge_fit_1d
from jam.jam_axi_proj import jam_axi_proj 


# FUNCTIONS =================
# Mge: incorporate the NFW into the gravitational potential
def mge_pot(Rs, p0, arcsec_to_pc):
    """
    Compute the dark matter NFW density profile and fit an MGE model.
    Applies Equation (6) from the README for normalization.
    """
    # Logarithmically spaced radius (in pc)
    r = np.geomspace(min(d['NFW_radius']), max(d['NFW_radius']), len(d['NFW_radius']))

    # Standard NFW density profile in (Msun/pc^3)
    DM_density = p0 / ((r / Rs) * (1 + r / Rs)**2)

    # Apply Equation (6) for projection
    DM_sigma_pc = np.sqrt(Rs * r)  
    DM_q = np.ones_like(DM_sigma_pc)  

    # Convert dispersions from pc to arcsec
    DM_sigma_arcsec = DM_sigma_pc / arcsec_to_pc

    # Compute projected surface density
    DM_surface_density = (DM_density * DM_q * DM_sigma_pc * np.sqrt(2 * np.pi))


    # Fit MGE model to DM surface density profile
    p = mge_fit_1d(
        r, DM_surface_density,
        negative=False,
        ngauss=12,
        rbounds=None,
        inner_slope=1,
        outer_slope=3,
        quiet=False,
        plot=True
    )

    # Extract fitted parameters
    DM_TotalCounts = p.sol[0, :]  # Amplitudes for the Gaussians (integrated)
    DM_sigma = p.sol[1, :]        # Dispersions (arcsec)

    return DM_TotalCounts, DM_sigma, DM_q

# Likelihood function for JAM + NFW model
def jam_nfw_lnprob(pars):
    inc, beta, mbh, ml, Rs, p0 = pars

    if not param(pars):
        return -np.inf

    # Compute DM MGE component
    DM_surface_density, DM_sigma_arcsec, DM_q = mge_pot(Rs, p0, arcsec_to_pc)

    # Concatenate stellar and DM MGE components
    combined_surface_density = np.concatenate((d['surf_pot'], DM_surface_density))
    combined_sigma = np.concatenate((d['sigma_pot'], DM_sigma_arcsec))
    combined_q = np.concatenate((d['qObs_pot'], DM_q))

    # Run JAM model with ml set
    jam = jam_axi_proj(
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
        goodbins=d['goodbins'],  # Ensure only good bins are used
        beta=np.full_like(d["qObs_lum"], beta),
        data=d['rms'],  # Only use good bins
        errors=d['erms'],  # Only use good bins
        ml=1 
    )

    chi2 = -0.5 * jam.chi2 * len(d['rms'])  # Use length of good bins
    return chi2

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
        [80.0, 0.0, 1.0, 2.0, radius, rho] + 0.01 * np.random.randn(ndim) for _ in range(nwalkers)
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
    nwalkers = 20
    nsteps = 100

    # Paths
    data_path = "/home/osilcock/DM_data/NFW_kwargs.pkl"
    output_path = "/fred/oz059/olivia/NFW_samples.pkl"

    # Load input data
    with open(data_path, "rb") as f:
        d = pickle.load(f)

    # Run MCMC
    run_mcmc(output_path, ndim, nwalkers, nsteps)













