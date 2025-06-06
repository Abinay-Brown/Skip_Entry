'''
Description: Multiple Shooting Optimization for Atmospheric Reentry Vehicles
Author: Abinay Brown
Date: 2025-06-05
'''
import casadi as cas
import matplotlib.pyplot as plt
from Constants import *
from numpy import sin, cos, tan, pi
from Dynamics import trajectory_generator



''' Generate Initial Guess '''
X0 = [params['Re'] + 125e3, 0, 0, 12791, -8 * (pi/180.0), 96*(pi/180)]
U0 = [0 * (pi/180.0), 0 * (pi/180.0)]
sim_time = 150
Xg, Ug, Tg = trajectory_generator(X0, U0, sim_time, N)
'''
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
print(Xg[0, -1]- params['Re'])
'''

# State Variables
r       = cas.MX.sym('r')           # Radial Distance (m)
theta   = cas.MX.sym('theta')       # Longitude (rad)
phi     = cas.MX.sym('phi')         # Latitude (rad)
Vr      = cas.MX.sym('Vr')          # Velocity Reentry (m/s)
y       = cas.MX.sym('y')           # Flight Path Angle (rad)
psi     = cas.MX.sym('psi')         # Heading Angle (rad)

x       = cas.vertcat(r, theta, phi, Vr, y, psi) 

# Control Variables
alpha = cas.MX.sym('alpha')         # Angle of Attack (rad)
sigma = cas.MX.sym('sigma')         # Bank Angle (rad)

u = cas.vertcat(alpha, sigma) 

# Model Equations 

# Aerodynamics
CN = params['Cpmax'] * (1 - ((params['rn']/params['rc'])**2)*(cos(params['dc'])**2))*((cos(params['dc'])**2)*cas.sin(alpha)*cas.cos(alpha));
CA = params['Cpmax'] * ((0.5*(1-(sin(params['dc'])**4))*(params['rn']/params['rc'])**2) +
                        ((sin(params['dc'])**2)*(cas.cos(alpha)**2) + 
                         0.5*(cas.sin(alpha)**2)*(cos(params['dc'])**2))*(1-((params['rn']/params['rc'])**2)*(cos(params['dc'])**2)));
CL = CN * cas.cos(alpha) - CA * cas.sin(alpha);
CD = CN * cas.sin(alpha) + CA * cas.cos(alpha);

rho = params['rho0'] * cas.exp(-(r - params['Re']) / params['H'])

L = 0.5 * rho * Vr * Vr * CL * params['S']
D = 0.5 * rho * Vr * Vr * CD * params['S']

# Gravity
g = params['g0'] * (params['Re'] / r)**2

# Dynamics   
r_dot = Vr * cas.sin(y)
theta_dot = (Vr * cas.cos(y) * cas.cos(psi)) / (r * cas.cos(phi))
phi_dot = (Vr * cas.cos(y) * cas.sin(psi)) / r
Vr_dot = -D/params['m'] - g * cas.sin(y) + r*(params['w']**2)* cas.cos(phi) * (cas.cos(phi)*cas.sin(y) - cas.sin(phi)*cas.sin(psi)*cas.cos(y))
y_dot = (1/Vr)*((L/params['m'])*cas.cos(sigma) - g*cas.cos(y) + (Vr**2/r)*cas.cos(y) + 2*Vr*params['w']*cas.cos(phi)*cas.cos(psi) + 
                r*(params['w']**2)*cas.cos(phi)*(cas.cos(phi)*cas.cos(y) - cas.sin(phi)*cas.sin(psi)*cas.sin(y)))
psi_dot = (1/Vr)*(L*cas.sin(sigma)/(params['m']*cas.cos(y)) - (Vr*Vr/r * cas.cos(y) * cas.cos(psi) * cas.tan(phi)) +
                  2*Vr*params['w']*(cas.sin(psi)*cas.cos(phi)*cas.tan(y) - cas.sin(phi)) - (r*(params['w']**2)/cas.cos(y)) * cas.sin(phi)*cas.cos(phi)*cas.cos(psi))

xdot = cas.vertcat(r_dot, theta_dot, phi_dot, Vr_dot, y_dot, psi_dot)

# Model Function
dynamics = cas.Function('dynamics', [x, u], [xdot]) 

# Decision Variables
X = cas.MX.sym('X', nx, N)       
U = cas.MX.sym('U', nu, N-1)     
T = cas.MX.sym('T') 

# Integrator & Shooting

# Runge-Kutta 4th Order Integrator
def rk4_step(x, u, h):
    k1 = dynamics(x, u)
    k2 = dynamics(x + (h/2) * k1, u)
    k3 = dynamics(x + (h/2) * k2, u)
    k4 = dynamics(x +  h    * k3, u)
    return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Shooting Function
def shoot(x, u, dt, n_sub=1):
    h = dt / n_sub
    x_temp = x
    for _ in range(n_sub):
        x_temp = rk4_step(x_temp, u, h)
    return x_temp
    
# Constraints Setup 

# Dynamics Constraints
g_list = []    
for k in range(N-1):
    xk = X[:, k]       
    uk = U[:, k]       
    dt = T/(N-1)  
    xkp1 = shoot(xk, uk, dt)  
    x_next = X[:, k+1]  
    res_k = x_next - xkp1 
    g_list.append(res_k)

g_cont = cas.vertcat(*g_list)


# Delta Control Constraints
lim = 10 * pi/180

delta_list = []
dt   = T/(N-1)
for k in range(1, N-1):
    delta_list.append( (U[0, k] - U[0, k-1])/dt )
    delta_list.append( (U[1, k] - U[1, k-1])/dt )
g_deltas = cas.vertcat(*delta_list)

 
# Boundary Conditions
g1 = X[0, 0] - (params['Re'] + 125e3)
g2 = X[1, 0] - (-123.35*(pi/180))
g3 = X[2, 0] - (41.59*(pi/180))
g4 = X[3, 0] - 12791
g5 = X[4, 0] - (-8*(pi/180))
g6 = X[5, 0] - 96*(pi/180)

g7 = X[0, -1] - (params['Re'] + 32e3)
g8 = X[1, -1] - ((-123.35 - 0.65)*(pi/180))
g9 = X[2, -1] - ((41.59 + 5.41)*(pi/180))
g_bc = cas.vertcat(g1, g2, g3, g4, g5, g6, g7, g8, g9)

g_eq = cas.vertcat(g_cont, g_bc, g_deltas)

# Lower and Upper Bound Constraints

x_lb = cas.DM([params['Re'] + 10000, -pi, -pi/2, 20, -pi/2, -pi]) 
x_ub = cas.DM([params['Re'] + 200000, pi,  pi/2, 20000,  pi/2, pi])

u_lb = cas.DM([ -2*pi/180,  -45*pi/180 ]) 
u_ub = cas.DM([  10*pi/180,   45*pi/180 ])


lbx_X = cas.repmat(x_lb, N, 1)   
ubx_X = cas.repmat(x_ub, N, 1)

lbx_U = cas.repmat(u_lb, N-1, 1)  
ubx_U = cas.repmat(u_ub, N-1, 1)


lb_T = cas.DM([50])
ub_T = cas.DM([2000])

lbx = cas.vertcat(lbx_X, lbx_U, lb_T)
ubx = cas.vertcat(ubx_X, ubx_U, ub_T)

# Define the objective function
#dt   = T/(N-1)
#decs = [(X[3, k] - X[3, k+1]) / dt for k in range(N-1)]  
#peak = decs[0]
#for k in range(1, N-1):
#    peak = cas.fmax(peak, decs[k])  
#obj = peak

obj = cas.sumsqr(cas.vec(U))
#obj = T
# Define the problem

w = cas.vertcat(cas.reshape(X, nx * N, 1), cas.reshape(U, nu * (N - 1), 1), T)
n_cont   = g_cont.size1()    # = nx*(N−1)
n_bc     = g_bc.size1()      # = 9
n_delta  = g_deltas.size1()  # = 2*(N−2)

# build lbg/ubg
lbg = cas.vertcat(
    cas.DM.zeros(n_cont + n_bc, 1),
    cas.DM(-lim * np.ones((n_delta, 1)))
)
ubg = cas.vertcat(
    cas.DM.zeros(n_cont + n_bc, 1),
    cas.DM(lim * np.ones((n_delta, 1)))
)


X_guess = cas.DM(Xg)   
U_guess = cas.DM(Ug)   

X_guess_flat = cas.reshape(X_guess, nx * N, 1)       
U_guess_flat = cas.reshape(U_guess, nu * (N - 1), 1) 

T_guess = cas.DM([Tg])  
w0 = cas.vertcat( X_guess_flat, U_guess_flat, T_guess)

nlp = {'x': w, 'f': obj, 'g': g_eq}

opts = { 'ipopt.print_level': 5,
        'print_time': True, 
        'ipopt.max_iter': 10000,
        'ipopt.tol': 1e-4, 
        'ipopt.nlp_scaling_method': 'gradient-based',
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.dual_inf_tol': 1e-4,
        'ipopt.constr_viol_tol': 1e-4}

solver = cas.nlpsol('solver', 'ipopt', nlp, opts)


sol = solver(x0  = w0, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg)

w_opt = sol['x']          
w_np = np.array(w_opt).flatten()

X_flat = w_np[0 : nx * N]

start_u = nx * N
end_u   = start_u + nu * (N - 1)
U_flat  = w_np[start_u : end_u]
T_opt = float(w_np[-1])

X_opt = X_flat.reshape((nx, N), order='F').T   

U_opt = U_flat.reshape((nu, N - 1), order='F').T   
U_opt_padded = np.vstack([U_opt, U_opt[-1, :]])   

theta_vals = X_opt[:, 1]   
phi_vals   = X_opt[:, 2]   
r_vals     = X_opt[:, 0] - params['Re'] 

print(X_opt[0, :])
print(X_opt[-1, :])
print(T_opt)

fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

ax.plot(
    theta_vals,    
    phi_vals,      
    r_vals,        
    marker='o',
    linestyle='-',
    color='b',
    markersize=3
)

ax.set_xlabel(r'$\theta$ [rad]', labelpad=10)
ax.set_ylabel(r'$\phi$ [rad]', labelpad=10)
ax.set_zlabel(r'$r$ [m]',   labelpad=10)
ax.set_title("3D Trajectory: $(\\theta,\\phi,r)$")

plt.tight_layout()
plt.show()