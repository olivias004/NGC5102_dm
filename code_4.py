import numpy as np
import emcee
from time import time, localtime, strftime
from scipy.optimize import minimize
import pickle

# --------------------- Copied Functions from the Original Code -----------------------

# Parameter boundaries. [lower, upper]
boundary = {'cosinc': [0.0, 1.0], 'beta': [0.0, 0.4], 'ml': [0.5, 15]}

# Parameter Gaussian priors. [mean, sigma]
prior = {'cosinc': [0.5, 1e4], 'beta': [0.2, 1e4], 'ml': [1.0, 1e4]}

model = {'boundary': boundary, 'prior': prior}

def check_boundary(parsDic, boundary=None):
    """Check whether parameters are within the boundary limits."""
    for key in parsDic.keys():
        if boundary[key][0] < parsDic[key] < boundary[key][1]:
            pass
        else:
            return -np.inf
    return 0.0

def lnprior(parsDic, prior=None):
    """Calculate the Gaussian prior log-probability."""
    rst = 0.0
    for key in parsDic.keys():
        rst += -0.5 * (parsDic[key] - prior[key][0])**2 / prior[key][1]**2
    return rst

def flat_initp(keys, nwalkers):
    """Create initial positions for MCMC: Flat distribution within boundaries."""
    ndim = len(keys)
    p0 = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        p0[:, i] = np.random.uniform(low=boundary[keys[i]][0]+1e-4,
                                     high=boundary[keys[i]][1]-1e-4,
                                     size=nwalkers)
    return p0

def lnprob_massFollowLight(pars, returnType='lnprob', model=None):
    """Log-probability for the mass-follows-light model."""
    cosinc, beta, ml = pars
    parsDic = {'cosinc': cosinc, 'beta': beta, 'ml': ml}
    
    # Check boundaries
    if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
        return -np.inf
    
    # Compute prior probability
    lnpriorValue = lnprior(parsDic, prior=model['prior'])
    inc = np.arccos(cosinc)
    Beta = np.zeros(model['lum2d'].shape[0]) + beta
    
    # Run JAM model (replace this with your actual JAM call)
    rmsModel = model['JAM'].run(inc, Beta, ml=ml)
    
    # Compute chi-squared
    chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
             model['errRms'][model['goodbins']])**2).sum()
    
    if np.isnan(chi2):
        return -np.inf
    
    # Log-probability
    return -0.5 * chi2 + lnpriorValue

def _runEmcee(sampler, p0):
    """Run the MCMC sampler."""
    burninStep = model['burnin']
    runStep = model['runStep']
    
    # Burn-in phase
    startTime = time()
    pos, prob, state = sampler.run_mcmc(p0, burninStep)
    print(f"Time for burn-in: {time() - startTime:.2f}s")
    sampler.reset()
    
    # Main MCMC run
    sampler.run_mcmc(pos, runStep)
    return sampler

def analyzeRst(sampler, model):
    """Analyze the MCMC results."""
    rst = {}
    rst['chain'] = sampler.chain
    rst['lnprobability'] = sampler.lnprobability
    try:
        rst['acor'] = sampler.acor
    except:
        rst['acor'] = np.nan
    rst['acceptance_fraction'] = sampler.acceptance_fraction
    rst['goodchains'] = ((rst['acceptance_fraction'] > 0.15) *
                         (rst['acceptance_fraction'] < 0.75))
    
    flatchain = rst['chain'][rst['goodchains'], :, :].reshape((-1, model['ndim']))
    flatlnprob = rst['lnprobability'][rst['goodchains'], :].reshape(-1)
    
    # Compute parameter estimates (median, mean, etc.)
    medianPars = np.median(flatchain, axis=0)
    print(f"Median parameters: {medianPars}")
    
    return rst

# --------------------- Script for Model A ----------------------------------------

# Your data
path = '/Users/livisilcock/Documents/PROJECTS/NGC5102/files/JAM_Model/kwargs.pkl'
with open(path, "rb") as f:
    d = pickle.load(f)




rms = d['rms']        # Observed root-mean-square velocities
err_rms = d['erms']      # Errors in RMS velocities
mge_lum = np.column_stack((d['surf_lum'], d['sigma_lum'], d['qObs_lum']))     # MGE luminosity model
mge_pot = np.column_stack((d['surf_pot'], d['sigma_pot'], d['qObs_pot']))      # MGE potential (same as luminosity for mass-follows-light)

# Set up the JAM model
class JAM:
    def run(self, inc, beta, ml):
        """Run the JAM model with jam_axi_proj."""
        # Required parameters from the data and model
        surf_lum, sigma_lum, qObs_lum = model['lum2d'].T
        surf_pot, sigma_pot, qObs_pot = model['pot2d'].T
        pixsize = d['pixsize']  # Placeholder for pixel size, adjust as needed
        sigmapsf = d['sigmapsf']  # Placeholder for PSF sigma, set to match your data
        normpsf = d['normpsf']  # Placeholder for PSF normalization, set to match your data
        dist = d['dist']  # Example: distance in Mpc, update based on your object
        bh0 = 0  # Black hole mass (use 0 for no BH in Model A)
        rot_x, rot_y = d['rot_x'], d['rot_y']  # Rotation adjustments (set to 0 unless specified)

        # Call to jam_axi_proj
        return jam_axi_proj(
            surf_lum, sigma_lum, qObs_lum,
            surf_pot, sigma_pot, qObs_pot,
            inc, bh0, dist, rot_x, rot_y,
            align='cyl', moment='zz', plot=False
            )


# Populate the model dictionary
model['lum2d'] = mge_lum
model['pot2d'] = mge_pot
model['rms'] = rms
model['errRms'] = err_rms
model['goodbins'] = d['goodbins']  # Assume all bins are good
model['JAM'] = JAM()  # Initialize JAM class
model['burnin'] = 500
model['runStep'] = 1000
model['nwalkers'] = 30
model['ndim'] = 3  # cosinc, beta, ml

# Initialize MCMC
keys = ['cosinc', 'beta', 'ml']
p0 = flat_initp(keys, model['nwalkers'])
sampler = emcee.EnsembleSampler(model['nwalkers'], model['ndim'], lnprob_massFollowLight,
                                kwargs={'model': model})

# Run the sampler
sampler = _runEmcee(sampler, p0)

# Analyze results
results = analyzeRst(sampler, model)

# Save results
with open("/Users/livisilcock/Desktop/model_a_results.pkl", "wb") as f:
    pickle.dump(results, f)
