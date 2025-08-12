import numpy as np
from numpy import sin, cos, tan, pi

N = 51             # Number of trajectory points
nx = 6              # State Dimension
nu = 2              # Control Dimension
NUM = 50

params = {}
# Blunt Nose Cone Geometry (Stardust)
params['m'] = 46                      # Mass (kg)
params['rn'] = 0.2202                 # Nose Radius (m)
params['rc'] = 0.4064                 # Cone Radius (m)
params['dc'] = 60 * (pi/180)          # Cone Angle (rad)
params['S'] = pi*params['rc']**2      # surface area (m^2)
params['Cpmax'] = 2                   # Newton Cp max Coefficient

# Constants
params['g0'] = 9.798                  # Earth's gravity at surface (m/s^2)
params['rho0'] = 1.225                # Earth's sea-level density (kg/m^3)
params['Re'] = 6378.137 * 10 ** 3     # Earth's radius (m)
params['w'] = 7.2921159 * 10 ** -5    # Earth's angular velocity (rad/s)
params['H'] = 8.5 * 10 ** 3           # Earth's scale height (m)
