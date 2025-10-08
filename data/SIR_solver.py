import numpy as np
import matplotlib.pyplot as plt
from spaces import GRF

def alpha_x(x):
    a = 0.5
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

def S_0_func(x):
    return (1.0 - 0.5*np.cos(4 * np.pi * x)) #1 + 0.5*np.cos(x * np.pi *3) #1.3 + 1.0 * np.cos(x * np.pi * 2)

def I_0_func(x):
    return 0.3*np.exp(-((x - 2/3)**2) / (2 * 0.15**2))#0.01 * np.exp(-1000*x)

def solver(N, dt, L, T, beta, gamma, dx_2, f):
    x, t, S, I, R = solve_SIR(N=N, dt=dt, L=L, T=T, beta=beta, gamma=gamma, dx_2 = dx_2, alpha_func=f)
    return x, t, S, I

def solver_all(N, dt, L, T, beta, gamma, dx_2, f):
    x, t, S, I, R = solve_SIR(N=N, dt=dt, L=L, T=T, beta=beta, gamma=gamma, dx_2 = dx_2, alpha_func=f)
    return x, t, S, I, R

def solve_SIR(N, dt, L, T, beta, gamma, dx_2, alpha_func):
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    D = dx_2 * alpha_func(x)
    
    # Initial conditions
    I = I_0_func(x)
    S = S_0_func(x)
    R = gamma*I
    
    Lap = -2.0 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    Lap[0, 1] = 2.0
    Lap[-1, -2] = 2.0
    Lap = Lap / (dx**2)
    ID = np.eye(N)

    Ssave = []
    Isave = []
    Rsave = []

    t_vals = np.arange(0, T + dt/2, dt)
    for t in t_vals:
        L1 = np.zeros((N, N))
        for i in range(N):
            L1[i, :] = Lap[i, :] * (S[i] * beta * D[i])
            L1[i, i] += S[i] * beta
        B = (1.0 + dt * gamma) * ID - dt * L1

        # Solve B * I_hat = I
        I_hat = np.linalg.solve(B, I)

        # S_hat = S - dt * L1 * I_hat
        S_hat = S - dt * (L1 @ I_hat)
        
        # R_t = gamma*I (no diffusion in R)
        R_hat = R + dt * gamma * I_hat

        # Update S, I
        I = I_hat
        S = S_hat
        R = R_hat

        Isave.append(I.copy())
        Ssave.append(S.copy())
        Rsave.append(R.copy())

    Ssave = np.array(Ssave).T
    Isave = np.array(Isave).T
    Rsave = np.array(Rsave).T
    return x, t_vals, Ssave, Isave, Rsave


def main():
    N = 201
    L = 1.0
    T = 10.0
    dt = 0.02
    gamma = 0.2
    beta = 0.8
    dx_2 = 0.001
    # x, t_vals, Ssave, Isave, Rsave = solve_SIR(N=N, dt=dt, L=L, T=T, beta=beta, dx_2 = dx_2, gamma=gamma, alpha_func=alpha_x0)
    
    dx_2 = 0.001
    x, t_vals, S_all, I_all, R_all = solve_SIR(N=N, dt=dt, L=L, T=T, beta=beta, dx_2 = dx_2, gamma=gamma, alpha_func=alpha_x)
    # print(np.mean(S_all), np.mean(Ssave))
    # print(np.mean(I_all), np.mean(Isave))
    # print(np.mean(R_all), np.mean(Rsave))

    import deepxde as dde
    # print(dx_2, dde.metrics.l2_relative_error(S_all, Ssave), 
    #       dde.metrics.l2_relative_error(R_all, Rsave),
    #       dde.metrics.l2_relative_error(I_all, Isave))
    
    print(S_all.shape, I_all.shape, R_all.shape)

    S_final = S_all[:, -1]
    I_final = I_all[:, -1]
    R_final = R_all[:, -1]
    
    D_values = dx_2*alpha_x(x)
    plt.figure()
    plt.plot(x, D_values, label='D(x)', linewidth=2, color='purple')
    plt.xlabel('x')
    plt.ylabel('D(x)')
    #plt.title('Spatial Variation of D(x)')
    #plt.legend()
    plt.show()
    
    S_values = S_0_func(x)
    plt.figure()
    plt.plot(x, S_values, label='S(x)', linewidth=2, color='purple')
    plt.xlabel('x')
    plt.ylabel('S_0')
    #plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(x, beta*S_values, label=r'$\beta*S_0$', linewidth=2, color='black')
    plt.plot(x, gamma*np.ones_like(S_values), label=r'$\gamma$', linewidth=2, color='red') 
    plt.xlabel('x')
    plt.ylabel('beta*S_0')
    plt.legend()
    plt.show()
    
    I_values = I_0_func(x)
    plt.figure()
    plt.plot(x, I_values, label='I(x)', linewidth=2, color='purple')
    plt.xlabel('x')
    plt.ylabel('I_0')
    #plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(x, S_final, label='S(x, T)', linewidth=2, color='blue')
    plt.plot(x, I_final, label='I(x, T)', linewidth=2, color='red')
    plt.plot(x, R_final, label='R(x, T)', linewidth=2, color='green')
    plt.xlabel('x')
    plt.ylabel('Solution at T=final')
    #plt.title('Final States')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(x, S_all[:, 0], label='S(x, T)', linewidth=2, color='blue')
    plt.plot(x, I_all[:, 0], label='I(x, T)', linewidth=2, color='red')
    plt.plot(x, R_all[:, 0], label='R(x, T)', linewidth=2, color='green')
    plt.xlabel('x')
    plt.ylabel('Solution at T=0')
    #plt.title('Final States')
    plt.legend()
    plt.show()

    x_edges = np.linspace(x.min(), x.max() + (x[1] - x[0]), len(x) + 1)
    t_edges = np.append(t_vals, t_vals[-1] + (t_vals[-1] - t_vals[-2]))
    
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x_edges, t_edges, S_all.T, cmap='rainbow')
    plt.colorbar(label='S(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('S(x,t)')
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x_edges, t_edges, I_all.T, cmap='rainbow')
    plt.colorbar(label='I(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('I(x,t)')
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x_edges, t_edges, R_all.T, cmap='rainbow')
    plt.colorbar(label='R(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('R(x,t)')
    plt.show()



if __name__ == "__main__":
    main()