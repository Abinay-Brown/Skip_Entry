import sympy as sym
import matplotlib.pyplot as plt
from Constants import *
from numpy import sin, cos, tan, pi
from scipy.integrate import solve_ivp
from mpl_toolkits import mplot3d

''' State Variables '''
r       = sym.symbols('r')        # Radial Distance (m)
theta   = sym.symbols('theta')    # Longitude (rad)
phi     = sym.symbols('phi')      # Latitude (rad)
Vr      = sym.symbols('V_r')      # Velocity Reentry (m/s)
y       = sym.symbols('gamma')    # Flight Path Angle (rad)
psi     = sym.symbols('psi')      # Heading Angle (rad)

''' Control Variables '''
alpha = sym.symbols('alpha')      # Angle of Attack (rad)
sigma = sym.symbols('sigma')      # Bank Angle (rad)


''' Model Equations '''
# Aerodynamics
CN = params['Cpmax'] * (1 - ((params['rn']/params['rc'])**2)*(cos(params['dc'])**2))*((cos(params['dc'])**2)*sym.sin(alpha)*sym.cos(alpha));
CA = params['Cpmax'] * ((0.5*(1-(sin(params['dc'])**4))*(params['rn']/params['rc'])**2) +
                        ((sin(params['dc'])**2)*(sym.cos(alpha)**2) + 
                         0.5*(sym.sin(alpha)**2)*(cos(params['dc'])**2))*(1-((params['rn']/params['rc'])**2)*(cos(params['dc'])**2)));

CL = CN * sym.cos(alpha) - CA * sym.sin(alpha);
CD = CN * sym.sin(alpha) + CA * sym.cos(alpha);

rho = params['rho0'] * np.e**(-(r - params['Re']) / params['H'])

L = 0.5 * rho * Vr * Vr * CL * params['S']
D = 0.5 * rho * Vr * Vr * CD * params['S']

# Gravity
g = params['g0'] * (params['Re'] / r)**2

# Dynamics   
r_dot = Vr * sym.sin(y)
theta_dot = (Vr * sym.cos(y) * sym.cos(psi)) / (r * sym.cos(phi))
phi_dot = (Vr * sym.cos(y) * sym.sin(psi)) / r
Vr_dot = -D/params['m'] - g * sym.sin(y) + r*(params['w']**2)* sym.cos(phi) * (sym.cos(phi)*sym.sin(y) - sym.sin(phi)*sym.sin(psi)*sym.cos(y))
y_dot = (1/Vr)*((L/params['m'])*sym.cos(sigma) - g*sym.cos(y) + ((Vr**2)/r)*sym.cos(y) + 2*Vr*params['w']*sym.cos(phi)*sym.cos(psi) + 
                r*(params['w']**2)*sym.cos(phi)*(sym.cos(phi)*sym.cos(y) - sym.sin(phi)*sym.sin(psi)*sym.sin(y)))
psi_dot = (1/Vr)*(L*sym.sin(sigma)/(params['m']*sym.cos(y)) - (Vr*Vr/r * sym.cos(y) * sym.cos(psi) * sym.tan(phi)) +
                  2*Vr*params['w']*(sym.sin(psi)*sym.cos(phi)*sym.tan(y) - sym.sin(phi)) -
                  (r*(params['w']**2)/sym.cos(y)) * sym.sin(phi)*sym.cos(phi)*sym.cos(psi))

''' Numerical functions '''
Fsym = sym.Matrix([r_dot, theta_dot, phi_dot, Vr_dot, y_dot, psi_dot])
Asym = Fsym.jacobian([r, theta, phi, Vr, y, psi])
Bsym = Fsym.jacobian([alpha, sigma])
F = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), Fsym, 'numpy')
A = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), Asym, 'numpy') 
B = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), Bsym, 'numpy')

''' Hessians '''
H1sym = sym.hessian(Fsym[0], [r, theta, phi, Vr, y, psi])
H2sym = sym.hessian(Fsym[1], [r, theta, phi, Vr, y, psi])
H3sym = sym.hessian(Fsym[2], [r, theta, phi, Vr, y, psi])
H4sym = sym.hessian(Fsym[3], [r, theta, phi, Vr, y, psi])
H5sym = sym.hessian(Fsym[4], [r, theta, phi, Vr, y, psi])
H6sym = sym.hessian(Fsym[5], [r, theta, phi, Vr, y, psi])

H1 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H1sym, 'numpy')
H2 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H2sym, 'numpy')
H3 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H3sym, 'numpy')
H4 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H4sym, 'numpy')
H5 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H5sym, 'numpy')
H6 = sym.lambdify((r, theta, phi, Vr, y, psi, alpha, sigma), H6sym, 'numpy')

''' ODE function '''
def dynamics(x, t, input):
    alpha, sigma = input
    r, theta, phi, Vr, y, psi = x[0:]
    xdot = F(r, theta, phi, Vr, y, psi, alpha, sigma)
    
    return np.squeeze(xdot)

''' Trajectory Generator '''
def trajectory_generator(X0, U0, sim_time, N):
    t_eval = np.linspace(0, sim_time, N)

    def dynamics_ivp(t, x):
        return dynamics(x, t, U0)

    sol = solve_ivp(fun=dynamics_ivp, t_span=(0.0, sim_time), y0=X0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    Xg = sol.y
    Ug = np.vstack([ np.ones((1, N-1)) * U0[0], np.ones((1, N-1)) * U0[1]])
    Tg = sim_time

    return Xg, Ug, Tg

'''
if __name__ == "__main__":
    X0 = [params['Re'] + 120e3, 0, 0, 7800, -5 * (np.pi/180.0), 0]

    U0 = [5 * (pi/180.0), 5 * (pi/180.0)]
    sim_time = 175
    Xg, Ug, Tg = trajectory_generator(X0, U0, sim_time, N)

    print(Xg[:, -1])
    print(Xg[0, -1] - params['Re'])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(
        Xg[1, :],       # θ
        Xg[2, :],       # ϕ
        Xg[0, :] - params['Re'],  # altitude
        c='green',
        s=5
    )
    ax.set_xlabel("Longitude θ (rad)")
    ax.set_ylabel("Latitude ϕ (rad)")
    ax.set_zlabel("Altitude = r–Re (m)")
    plt.show()
'''