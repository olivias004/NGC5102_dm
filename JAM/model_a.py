import numpy as np
import pickle
from mcmc_pyjam import mcmc  # Importing the provided Mitzkus MCMC module




# ------------------ MAIN SCRIPT -------------------
if __name__ == "__main__":
    # Paths
    data_path = "/home/osilcock/DM_data/kwargs.pkl"  # Update with your data file path
    output_path = "/fred/oz059/olivia/NGC5102_samples_test.dat"  # Output file

    # Load the galaxy data
    with open(data_path, "rb") as f:
        galaxy_data = pickle.load(f)

    # Prepare the necessary inputs for the Mitzkus `mcmc` class
    galaxy = {
        "lum2d": np.column_stack((galaxy_data["surf_lum"], galaxy_data["sigma_lum"], galaxy_data["qObs_lum"])),
        "pot2d": np.column_stack((galaxy_data["surf_pot"], galaxy_data["sigma_pot"], galaxy_data["qObs_pot"])),  
        "distance": galaxy_data["dist"],   # Galaxy distance [Mpc]
        "rms": galaxy_data["rms"],         # RMS velocity
        "errRms": galaxy_data["erms"],     # RMS velocity errors
        "goodbins": galaxy_data["goodbins"],  # Boolean array for good bins
        "sigmapsf": galaxy_data["sigmapsf"],  # PSF sigma
        "pixsize": galaxy_data["pixsize"],    # Pixel size [arcsec]
        "xbin": galaxy_data["xbin"],         # x-coordinates of data points
        "ybin": galaxy_data["ybin"],         # y-coordinates of data points
        "name": "NGC5102_ModelA",            # Galaxy name (optional)
        "burnin": 100,                        # Number of burn-in steps
        "runStep": 200,                       # Number of steps in the final run
        "nwalkers": 10,                       # Number of walkers
        "clip": "noclip",                    # Clipping mode
        "outfolder": "./",                  # Output folder
        "fname": "NGC5102_samples_test.dat",   # Output file name
    }

    # Instantiate the Mitzkus `mcmc` class with the galaxy data
    model = mcmc(galaxy)

    # Set up the Mass-Follows-Light (MFL) model
    model.massFollowLight()

    # Outputs are saved automatically in the specified folder and file name.
    print(f"MCMC completed. Results saved to: {output_path}")
