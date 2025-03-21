


#IMPORTS=====================
from kinms import KinMS

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from kinms.utils.sauron_colormap import sauron



# Define the surface brightness profile
scalerad = 10  # Scale radius in arcseconds
radius = np.arange(0, 1000, 0.1)  # Radius vector in arcseconds
sbprof = np.exp(-radius / scalerad)  # Exponential profile

# Define the velocity profile
vel = (210) * (2 / np.pi) * np.arctan(radius)  # Max velocity of 210 km/s

print("Surface brightness and velocity profiles defined.")


# Define position angle and inclination
pos = 270  # Position angle in degrees
inc = 45  # Inclination angle in degrees

print("Galaxy orientation defined.")


# Define the data cube properties
xsize = 128  # X-axis size in arcseconds
ysize = 128  # Y-axis size in arcseconds
vsize = 700  # Velocity range in km/s
cellsize = 1  # Pixel size in arcsec/pixel
dv = 10  # Channel width in km/s
beamsize = [4, 4, 0]  # Beam size: major, minor (arcsec), and position angle (degrees)

print("Data cube properties defined.")



# Initialize the KinMS model with the total flux
flux = 1.6  # Total HI flux in Jy km/s (adjust as needed)

kin = KinMS(
    xsize, ysize, vsize, cellsize, dv, beamsize, verbose=True
)

# Assign the flux to the KinMS model
kin.flux_clouds = np.array([flux])
print("KINMS model initialized with flux.")



# Generate the model cube
cube = kin.model_cube(
    inc=inc, sbProf=sbprof, sbRad=radius, velProf=vel, posAng=pos
)

print("Model cube generated.")

