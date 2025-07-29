import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from spaces import GRF

def alpha_x(x):
    np.random.seed(42)  
    a = 0.1
    l = 0.01
    space = GRF(1, length_scale=l, N=201, interp="cubic")
    features = space.random(1)
    alpha_0 = np.exp(-((x - 0.5)**2) / (2 * 0.2**2))
    sample = np.log(1+np.exp(space.eval_u(features, x))) 
    return alpha_0 + (np.ravel(a * sample))

def alpha_x0(x):
    a = 0
    l = 0.01
    space = GRF(1, length_scale=l, N=201, interp="cubic")
    features = space.random(1)
    alpha_0 = np.exp(-((x - 0.5)**2) / (2 * 0.2**2))
    sample = np.log(1+np.exp(space.eval_u(features, x))) 
    return alpha_0 + (np.ravel(a * sample))

def pde_ode_system(t, y, N, dx, beta, gamma, D_values):
    """
    The PDE system:

      S_t = - beta [ D(x)* S * I_xx + S*I ]
      I_t =   beta [ D(x)* S * I_xx + S*I ] - gamma I
      R_t =  γ I

    with Dirichlet BC: S(0,t)=S(L,t)=0, I(0,t)=I(L,t)=0. S(x,0)=S_0, I(x,0)=I_0
    """
    S = y[0:N]
    I = y[N:2*N]
    R = y[2*N:3*N]
    
    # Zero Dirichlet BCs
    I_bc = np.zeros(N+2)
    I_bc[1:-1] = I
    # I_bc[0] = I[0]
    # I_bc[-1] = I[-1]
    # I_bc = np.pad(I, (1, 1), 'edge')
    
    S_bc = np.zeros(N+2)
    S_bc[1:-1] = S
    # S_bc[0]   = 1.0
    # S_bc[-1]  = 1.0
    # S_bc = np.pad(S, (1, 1), 'edge')
    
    I_xx = (I_bc[2:] - 2.0 * I_bc[1:-1] + I_bc[:-2]) / (dx**2)
    diff_term = D_values * S_bc[1:-1] * I_xx
    reaction  = S_bc[1:-1] * I_bc[1:-1]

    dSdt = -beta * (diff_term + reaction)
    dIdt =  beta * (diff_term + reaction) - gamma * I_bc[1:-1]
    dRdt =  gamma * I_bc[1:-1]
    
    return np.concatenate([dSdt, dIdt, dRdt])

def S_0_func(x):
    return (1.0 - 0.5*np.cos(2 * np.pi * x)) #1 + 0.5*np.cos(x * np.pi *3) #1.3 + 1.0 * np.cos(x * np.pi * 2)

def I_0_func(x,L):
    return 0.01*np.exp(-((x - L/3)**2) / (2 * 0.1**2))#0.01 * np.exp(-1000*x)

def solve_SIR_diffusion(N=200, Nt=200, L=1.0, T=1.0, beta=1.0, gamma=1.0, dx_2 = 0.001, alpha_func=None):
    if alpha_func is None:
        alpha_func = lambda x: 1.0
    dx = L/(N-1)
    xgrid = np.linspace(0, L, N)
    D_values = alpha_func(xgrid)*dx_2

    S_init = S_0_func(xgrid)
    I_init = I_0_func(xgrid, L)
    R_init = gamma * I_init
    y0 = np.concatenate([S_init, I_init, R_init])
    t_eval = np.linspace(0, T, Nt)
    #implicit solver
    sol = solve_ivp(
        fun=lambda t, y: pde_ode_system(t, y, N, dx, beta, gamma, D_values),
        t_span=(0, T),
        y0=y0,
        method='BDF',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-11,
        dense_output=True
    )
    
    # sol.t, sol.y => shape(2*N, len(t))
    return xgrid, sol.t, sol.y

def solver(N, Nt, L, T, beta, gamma, dx_2, f):
    x, t, Y = solve_SIR_diffusion(N=N, Nt=Nt, L=L, T=T, beta=beta, gamma=gamma, dx_2 = dx_2, alpha_func=f)
    S = Y[0:N, :]
    I = Y[N:2*N, :]
    return x, t, S, I

def time_derivative(S, dt):
    Nx, Nt = S.shape
    S_t = np.zeros_like(S)

    # Interior points: central difference
    # For j = 1..Nt-2
    S_t[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2 * dt)

    # First time index: forward difference
    S_t[:, 0] = (S[:, 1] - S[:, 0]) / dt

    # Last time index: backward difference
    S_t[:, -1] = (S[:, -1] - S[:, -2]) / dt

    return S_t

def second_derivative_x(S, dx):
    Nx, Nt = S.shape
    S_xx = np.zeros_like(S)

    # Interior points: central difference
    # S_xx[i, j] = (S[i+1, j] - 2*S[i, j] + S[i-1, j]) / dx^2
    S_xx[1:-1, :] = (S[2:, :] - 2.0 * S[1:-1, :] + S[:-2, :]) / (dx**2)

    # Boundary treatment (simple "copy from next" or custom one-sided difference)
    # Option A: Copy from the nearest interior node:
    S_xx[0, :]  = S_xx[1, :]
    S_xx[-1, :] = S_xx[-2, :]
    return S_xx

def main():
    N = 1001
    Nt = 3001
    L = 1.0
    T = 30.0
    beta = 0.9
    gamma = 0.5
    dx_2 = 0.001
    x, t_vals, Y = solve_SIR_diffusion(N=N, Nt=Nt, L=L, T=T, beta=beta, gamma=gamma, dx_2 = dx_2, alpha_func=alpha_x)
    print("The shape of solution:", Y.shape, t_vals.shape, x.shape)
    
    N = len(x)
    S_all0 = Y[0:len(x), :]
    I_all0 = Y[len(x):2*len(x), :]
    R_all0 = Y[2*len(x):3*len(x), :]
    
    S_t = time_derivative(S_all0, 30/(Nt-1))
    I_t = time_derivative(I_all0, 30/(Nt-1))
    R_t = time_derivative(R_all0, 30/(Nt-1))

    print("S_t.shape =", S_t.shape)
    print(np.max((S_t + I_t + R_t)))
    dx = x[1] - x[0]
    I_xx = second_derivative_x(I_all0, dx)
    # print("I_xx.shape =", I_xx.shape)
    D_values = dx_2*alpha_x(x)
    D_values = np.rot90(np.tile(D_values, (Nt, 1)), k=3)
    # D_values = 0.0005*np.ones_like(I_xx)
    print(np.max(D_values*beta*S_all0*I_xx + beta*S_all0*I_all0 + S_t))
    print(np.max(D_values*beta*S_all0*I_xx + beta*S_all0*I_all0 - I_t - gamma*I_all0))
    print(np.max(R_t - gamma*I_all0))



if __name__ == "__main__":
    main()
