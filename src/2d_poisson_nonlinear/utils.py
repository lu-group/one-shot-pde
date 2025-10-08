import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


def plot_data(f_name, u_name, d_num):
    f = np.loadtxt("data{}/{}.dat".format(d_num,f_name))
    u = np.loadtxt("data{}/{}.dat".format(d_num, u_name))
    f = np.rot90(f)
    u = np.rot90(u)
    print("({}, {})".format(f_name, u_name))
    plt.figure()
    plt.rcParams.update({'font.size': 20,"savefig.dpi": 200, "figure.figsize": (8, 6)})
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(f, cmap = "gnuplot",  origin='upper', extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/{}.png".format(d_num,f_name))
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.imshow(u,  origin='upper', cmap = "gnuplot", extent=(0,1,0,1), aspect='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.savefig("data{}/{}.png".format(d_num,u_name))
    plt.show()

def construct_more_data5(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], f[i, j]
    M = len(f) # 1001
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    #print(xstep, tstep) #10, 10
    outputs = u[xstep: -xstep, tstep:-tstep].reshape((-1, 1))
    #print(outputs.shape) #(972171, 1)
    #print(u[:-2*xstep, tstep:].shape, u[xstep: -xstep, :-tstep].shape, u[2*xstep: , tstep:].shape, f[xstep: -xstep, tstep].shape)
    inputs = np.hstack((u[:-2*xstep, tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, :-2*tstep].reshape((-1, 1)), 
                        u[2*xstep: , tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, 2*tstep:].reshape((-1, 1)),
                        f[xstep: -xstep, tstep:-tstep].reshape((-1, 1))))
    #print(inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data9(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1], u[i-1, j-1],  
    # u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], f[i, j]
    M = len(f) # 1001
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    #print(xstep, tstep) #10, 10
    outputs = u[xstep: -xstep, tstep:-tstep].reshape((-1, 1))
    inputs = np.hstack((u[:-2*xstep, tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, :-2*tstep].reshape((-1, 1)), 
                        u[2*xstep: , tstep:-tstep].reshape((-1, 1)), u[xstep: -xstep, 2*tstep:].reshape((-1, 1)),
                        u[:-2*xstep, :-2*tstep].reshape((-1, 1)),u[:-2*xstep, 2*tstep:].reshape((-1, 1)),
                        u[2*xstep: , :-2*tstep].reshape((-1, 1)), u[2*xstep: , 2*tstep:].reshape((-1, 1)), 
                        f[xstep: -xstep, tstep:-tstep].reshape((-1, 1))))
    #print(inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data13(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1],
    # u[i-1, j-1], u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], 
    # u[i-2, j], u[i, j-2], u[i+2, j], u[i, j+2], 
    # f[i, j]
    M = len(f) # 201
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    #print(xstep, tstep)
    outputs = u[2*xstep: -2*xstep, 2*tstep:-2*tstep].reshape((-1, 1))
    inputs = np.hstack((u[xstep:-3*xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, tstep:-3*tstep].reshape((-1, 1)), 
                        u[3*xstep:-xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, 3*tstep:-tstep].reshape((-1, 1)),
                        u[xstep:-3*xstep, tstep:-3*tstep].reshape((-1, 1)), u[xstep:-3*xstep, 3*tstep:-tstep].reshape((-1, 1)),
                        u[3*xstep:-xstep, tstep:-3*tstep].reshape((-1, 1)), u[3*xstep:-xstep, 3*tstep:-tstep].reshape((-1, 1)),                         
                        u[:-4*xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, :-4*tstep].reshape((-1, 1)), 
                        u[4*xstep: , 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, 4*tstep:].reshape((-1, 1)),
                        f[2*xstep: -2*xstep, 2*tstep:-2*tstep].reshape((-1, 1))))
    return inputs, outputs

def construct_more_data13_5(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1],
    # u[i-1, j-1], u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], 
    # u[i-2, j], u[i, j-2], u[i+2, j], u[i, j+2], 
    # f[i, j]
    M = len(f) # 201
    xstep = int((M-1)/(Nx-1))
    tstep = int((M-1)/(Ny-1))
    #print(xstep, tstep)
    outputs = np.hstack((u[xstep:-3*xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, tstep:-3*tstep].reshape((-1, 1)), 
                        u[3*xstep:-xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, 3*tstep:-tstep].reshape((-1, 1)),
                        u[2*xstep: -2*xstep, 2*tstep:-2*tstep].reshape((-1, 1))))
    inputs = np.hstack((f[xstep:-3*xstep, 2*tstep:-2*tstep].reshape((-1, 1)), f[2*xstep: -2*xstep, tstep:-3*tstep].reshape((-1, 1)), 
                        f[3*xstep:-xstep, 2*tstep:-2*tstep].reshape((-1, 1)), f[2*xstep: -2*xstep, 3*tstep:-tstep].reshape((-1, 1)),
                        u[xstep:-3*xstep, tstep:-3*tstep].reshape((-1, 1)), u[xstep:-3*xstep, 3*tstep:-tstep].reshape((-1, 1)),
                        u[3*xstep:-xstep, tstep:-3*tstep].reshape((-1, 1)), u[3*xstep:-xstep, 3*tstep:-tstep].reshape((-1, 1)),                         
                        u[:-4*xstep, 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, :-4*tstep].reshape((-1, 1)), 
                        u[4*xstep: , 2*tstep:-2*tstep].reshape((-1, 1)), u[2*xstep: -2*xstep, 4*tstep:].reshape((-1, 1)),
                        f[2*xstep: -2*xstep, 2*tstep:-2*tstep].reshape((-1, 1))))
    #print("13_5:", inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data25(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1],
    # u[i-1, j-1], u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], 
    # u[i-2, j], u[i, j-2], u[i+2, j], u[i, j+2], 
    # u[i-2, j-1], u[i-2, j+1], u[i+2, j-1], u[i+2, j+1], 
    # u[i-1, j-2], u[i+1, j-2], u[i-1, j+2], u[i+1, j+2], 
    # u[i-2, j-2], u[i-2, j+2], u[i+2, j-2], u[i+2, j+2], 
    # f[i, j]
    M = len(f) # 201
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    #print(xstep, ystep)
    outputs = u[2*xstep:-2*xstep, 2*ystep:-2*ystep].reshape((-1, 1))
    inputs = np.hstack([
        u[2*xstep:-2*xstep, ystep:-3*ystep].reshape((-1, 1)),  # u[i, j-1]
        u[3*xstep:-xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i+1, j]
        u[2*xstep:-2*xstep, 3*ystep:-ystep].reshape((-1, 1)),  # u[i, j+1]
        u[xstep:-3*xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i-1, j]
        
        u[xstep:-3*xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i-1, j-1]
        u[xstep:-3*xstep, 3*ystep:-ystep].reshape((-1, 1)),    # u[i-1, j+1]
        u[3*xstep:-xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i+1, j-1]
        u[3*xstep:-xstep, 3*ystep:-ystep].reshape((-1, 1)),    # u[i+1, j+1]
        
        u[:-4*xstep, 2*ystep:-2*ystep].reshape((-1, 1)),       # u[i-2, j]
        u[2*xstep:-2*xstep, :-4*ystep].reshape((-1, 1)),       # u[i, j-2]
        u[4*xstep:, 2*ystep:-2*ystep].reshape((-1, 1)),        # u[i+2, j]
        u[2*xstep:-2*xstep, 4*ystep:].reshape((-1, 1)),        # u[i, j+2]
        
        u[:-4*xstep, ystep:-3*ystep].reshape((-1, 1)),         # u[i-2, j-1]
        u[:-4*xstep, 3*ystep:-ystep].reshape((-1, 1)),         # u[i-2, j+1]
        u[4*xstep:, ystep:-3*ystep].reshape((-1, 1)),          # u[i+2, j-1]
        u[4*xstep:, 3*ystep:-ystep].reshape((-1, 1)),          # u[i+2, j+1]
        
        u[xstep:-3*xstep, :-4*ystep].reshape((-1, 1)),         # u[i-1, j-2]
        u[3*xstep:-xstep, :-4*ystep].reshape((-1, 1)),         # u[i+1, j-2]
        u[xstep:-3*xstep, 4*ystep:].reshape((-1, 1)),          # u[i-1, j+2]
        u[3*xstep:-xstep, 4*ystep:].reshape((-1, 1)),          # u[i+1, j+2]
        
        u[:-4*xstep, :-4*ystep].reshape((-1, 1)),              # u[i-2, j-2]
        u[:-4*xstep, 4*ystep:].reshape((-1, 1)),              # u[i-2, j+2]
        u[4*xstep:, :-4*ystep].reshape((-1, 1)),              # u[i+2, j-2]
        u[4*xstep:, 4*ystep:].reshape((-1, 1)),               # u[i+2, j+2]
        
        f[2*xstep:-2*xstep, 2*ystep:-2*ystep].reshape((-1, 1))  # f[i, j]
    ])
    #print(inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data25_9(Nx, Ny, f, u):
    # u[i-1, j], u[i, j-1], u[i+1, j], u[i, j+1],
    # u[i-1, j-1], u[i-1, j+1], u[i+1, j-1], u[i+1, j+1], 
    # u[i-2, j], u[i, j-2], u[i+2, j], u[i, j+2], 
    # u[i-2, j-1], u[i-2, j+1], u[i+2, j-1], u[i+2, j+1], 
    # u[i-1, j-2], u[i+1, j-2], u[i-1, j+2], u[i+1, j+2], 
    # u[i-2, j-2], u[i-2, j+2], u[i+2, j-2], u[i+2, j+2], 
    # f[i, j]
    M = len(f) # 201
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    #print(xstep, ystep)
    outputs = np.hstack([
        u[2*xstep:-2*xstep, ystep:-3*ystep].reshape((-1, 1)),  # u[i, j-1]
        u[3*xstep:-xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i+1, j]
        u[2*xstep:-2*xstep, 3*ystep:-ystep].reshape((-1, 1)),  # u[i, j+1]
        u[xstep:-3*xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i-1, j]
        
        u[xstep:-3*xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i-1, j-1]
        u[xstep:-3*xstep, 3*ystep:-ystep].reshape((-1, 1)),    # u[i-1, j+1]
        u[3*xstep:-xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i+1, j-1]
        u[3*xstep:-xstep, 3*ystep:-ystep].reshape((-1, 1)),
        u[2*xstep:-2*xstep, 2*ystep:-2*ystep].reshape((-1, 1))])
    inputs = np.hstack([
        f[2*xstep:-2*xstep, ystep:-3*ystep].reshape((-1, 1)),  # u[i, j-1]
        f[3*xstep:-xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i+1, j]
        f[2*xstep:-2*xstep, 3*ystep:-ystep].reshape((-1, 1)),  # u[i, j+1]
        f[xstep:-3*xstep, 2*ystep:-2*ystep].reshape((-1, 1)),  # u[i-1, j]
        
        f[xstep:-3*xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i-1, j-1]
        f[xstep:-3*xstep, 3*ystep:-ystep].reshape((-1, 1)),    # u[i-1, j+1]
        f[3*xstep:-xstep, ystep:-3*ystep].reshape((-1, 1)),    # u[i+1, j-1]
        f[3*xstep:-xstep, 3*ystep:-ystep].reshape((-1, 1)),    # u[i+1, j+1]
        
        u[:-4*xstep, 2*ystep:-2*ystep].reshape((-1, 1)),       # u[i-2, j]
        u[2*xstep:-2*xstep, :-4*ystep].reshape((-1, 1)),       # u[i, j-2]
        u[4*xstep:, 2*ystep:-2*ystep].reshape((-1, 1)),        # u[i+2, j]
        u[2*xstep:-2*xstep, 4*ystep:].reshape((-1, 1)),        # u[i, j+2]
        
        u[:-4*xstep, ystep:-3*ystep].reshape((-1, 1)),         # u[i-2, j-1]
        u[:-4*xstep, 3*ystep:-ystep].reshape((-1, 1)),         # u[i-2, j+1]
        u[4*xstep:, ystep:-3*ystep].reshape((-1, 1)),          # u[i+2, j-1]
        u[4*xstep:, 3*ystep:-ystep].reshape((-1, 1)),          # u[i+2, j+1]
        
        u[xstep:-3*xstep, :-4*ystep].reshape((-1, 1)),         # u[i-1, j-2]
        u[3*xstep:-xstep, :-4*ystep].reshape((-1, 1)),         # u[i+1, j-2]
        u[xstep:-3*xstep, 4*ystep:].reshape((-1, 1)),          # u[i-1, j+2]
        u[3*xstep:-xstep, 4*ystep:].reshape((-1, 1)),          # u[i+1, j+2]
        
        u[:-4*xstep, :-4*ystep].reshape((-1, 1)),              # u[i-2, j-2]
        u[:-4*xstep, 4*ystep:].reshape((-1, 1)),              # u[i-2, j+2]
        u[4*xstep:, :-4*ystep].reshape((-1, 1)),              # u[i+2, j-2]
        u[4*xstep:, 4*ystep:].reshape((-1, 1)),               # u[i+2, j+2]
        
        f[2*xstep:-2*xstep, 2*ystep:-2*ystep].reshape((-1, 1))  # f[i, j]
    ])
    # print("25_9:", inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data49(Nx, Ny, f, u):
    # u[i-3, j-3], u[i-3, j-2], u[i-3, j-1], u[i-3, j], u[i-3, j+1], u[i-3, j+2], u[i-3, j+3]
    # u[i-2, j-3], u[i-2, j-2], u[i-2, j-1], u[i-2, j], u[i-2, j+1], u[i-2, j+2], u[i-2, j+3]
    # u[i-1, j-3], u[i-1, j-2], u[i-1, j-1], u[i-1, j], u[i-1, j+1], u[i-1, j+2], u[i-1, j+3]
    # u[i  , j-3], u[i  , j-2], u[i  , j-1], u[i  , j+1], u[i  , j+2], u[i  , j+3]
    # u[i+1, j-3], u[i+1, j-2], u[i+1, j-1], u[i+1, j], u[i+1, j+1], u[i+1, j+2], u[i+1, j+3]
    # u[i+2, j-3], u[i+2, j-2], u[i+2, j-1], u[i+2, j], u[i+2, j+1], u[i+2, j+2], u[i+2, j+3]
    # u[i+3, j-3], u[i+3, j-2], u[i+3, j-1], u[i+3, j], u[i+3, j+1], u[i+3, j+2], u[i+3, j+3]
    # f[i, j]
    M = len(f) 
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    outputs = u[3*xstep:-3*xstep, 3*ystep:-3*ystep].reshape((-1, 1))
    inputs = np.hstack([
        u[3*xstep-dx:-(3*xstep)-dx, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dx in range(-2*xstep, 4*xstep, xstep)
        for dy in range(-2*ystep, 4*ystep, ystep)
        if not (dx == 0 and dy == 0) 
    ])
    edge1 = np.hstack([
        u[6*xstep:, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dy in range(-2*ystep, 4*ystep, ystep)
    ])
    edge2 = np.hstack([
        u[3*xstep-dx:-(3*xstep)-dx, 6*ystep:].reshape((-1, 1))
        for dx in range(-2*xstep, 4*xstep, xstep)
    ])
    central = f[3*xstep:-3*xstep, 3*ystep:-3*ystep].reshape((-1, 1))
    inputs = np.hstack((inputs, edge1, edge2, u[6*xstep:, 6*ystep:].reshape((-1, 1)), central))
    #print(inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data49_25(Nx, Ny, f, u):
    # u[i-3, j-3], u[i-3, j-2], u[i-3, j-1], u[i-3, j], u[i-3, j+1], u[i-3, j+2], u[i-3, j+3]
    # u[i-2, j-3], u[i-2, j-2], u[i-2, j-1], u[i-2, j], u[i-2, j+1], u[i-2, j+2], u[i-2, j+3]
    # u[i-1, j-3], u[i-1, j-2], u[i-1, j-1], u[i-1, j], u[i-1, j+1], u[i-1, j+2], u[i-1, j+3]
    # u[i  , j-3], u[i  , j-2], u[i  , j-1], u[i  , j+1], u[i  , j+2], u[i  , j+3]
    # u[i+1, j-3], u[i+1, j-2], u[i+1, j-1], u[i+1, j], u[i+1, j+1], u[i+1, j+2], u[i+1, j+3]
    # u[i+2, j-3], u[i+2, j-2], u[i+2, j-1], u[i+2, j], u[i+2, j+1], u[i+2, j+2], u[i+2, j+3]
    # u[i+3, j-3], u[i+3, j-2], u[i+3, j-1], u[i+3, j], u[i+3, j+1], u[i+3, j+2], u[i+3, j+3]
    # f[i, j]
    M = len(f) 
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    outputs = np.hstack([
        u[3*xstep-dx:-(3*xstep)-dx, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dx in range(-2*xstep, 3*xstep, xstep)
        for dy in range(-2*ystep, 3*ystep, ystep)
    ])

    inputs = np.hstack([
        f[3*xstep-dx:-(3*xstep)-dx, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dx in range(-2*xstep, 3*xstep, xstep)
        for dy in range(-2*ystep, 3*ystep, ystep)
    ])
    edge1 = np.hstack([
        u[6*xstep:, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dy in range(-2*ystep, 4*ystep, ystep)
    ])
    edge2 = np.hstack([
        u[3*xstep-dx:-(3*xstep)-dx, 6*ystep:].reshape((-1, 1))
        for dx in range(-2*xstep, 4*xstep, xstep)
    ])
    edge3 = np.hstack([
        u[:-6*xstep, 3*ystep-dy:-(3*ystep)-dy].reshape((-1, 1))
        for dy in range(-2*ystep, 4*ystep, ystep)
    ])
    edge4 = np.hstack([
        u[3*xstep-dx:-(3*xstep)-dx, :-6*ystep].reshape((-1, 1))
        for dx in range(-2*xstep, 3*xstep, xstep)
    ])
    corner = np.hstack([u[6*xstep:, 6*ystep:].reshape((-1, 1))])

    inputs = np.hstack((inputs, edge1, edge2, edge3, edge4, corner))
    #print("49_25:", inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data81(Nx, Ny, f, u):
    M = len(f) 
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    outputs = u[4*xstep:-4*xstep, 4*ystep:-4*ystep].reshape((-1, 1))
    inputs = np.hstack([
        u[4*xstep-dx:-(4*xstep)-dx, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dx in range(-3*xstep, 5*xstep, xstep)
        for dy in range(-3*ystep, 5*ystep, ystep)
        if not (dx == 0 and dy == 0)
    ])
    edge1 = np.hstack([
        u[8*xstep:, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dy in range(-3*ystep, 5*ystep, ystep)
    ])
    edge2 = np.hstack([
        u[4*xstep-dx:-(4*xstep)-dx, 8*ystep:].reshape((-1, 1))
        for dx in range(-3*xstep, 5*xstep, xstep)
    ])
    central = f[4*xstep:-4*xstep, 4*ystep:-4*ystep].reshape((-1, 1))
    inputs = np.hstack((inputs, edge1, edge2, u[8*xstep:, 8*ystep:].reshape((-1, 1)), central))
    #print(inputs.shape, outputs.shape)
    return inputs, outputs

def construct_more_data81_49(Nx, Ny, f, u):
    M = len(f) 
    xstep = (M-1) // (Nx-1)
    ystep = (M-1) // (Ny-1)
    outputs = np.hstack([
        u[4*xstep-dx:-(4*xstep)-dx, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dx in range(-3*xstep, 4*xstep, xstep)
        for dy in range(-3*ystep, 4*ystep, ystep)
    ])
    inputs = np.hstack([
        f[4*xstep-dx:-(4*xstep)-dx, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dx in range(-3*xstep, 4*xstep, xstep)
        for dy in range(-3*ystep, 4*ystep, ystep)
    ])
    edge1 = np.hstack([
        u[8*xstep:, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dy in range(-3*ystep, 5*ystep, ystep)
    ])
    edge2 = np.hstack([
        u[4*xstep-dx:-(4*xstep)-dx, 8*ystep:].reshape((-1, 1))
        for dx in range(-3*xstep, 5*xstep, xstep)
    ])
    edge3 = np.hstack([
        u[:-8*xstep, 4*ystep-dy:-(4*ystep)-dy].reshape((-1, 1))
        for dy in range(-3*ystep, 5*ystep, ystep)
    ])
    edge4 = np.hstack([
        u[4*xstep-dx:-(4*xstep)-dx, :-8*ystep].reshape((-1, 1))
        for dx in range(-3*xstep, 4*xstep, xstep)
    ])
    corner = np.hstack([u[8*xstep:, 8*ystep:].reshape((-1, 1))])
    inputs = np.hstack((inputs, edge1, edge2, edge3, edge4, corner))
    #print("81_49:", inputs.shape, outputs.shape)
    return inputs, outputs



def load_all_data(M, Nx, Ny, N_f, N_b, l, a, l_new, a_new, dname, size, gen = False, correction = False, grid = False, isplot = False):
    # if gen:
    #     print("Generate new dataset ... ")
    #     gen_data_GRF(M, Nx, Ny, l, a, dname, isplot)
    #     interp_u0 = gen_test_data(M, Nx, Ny, dname, isplot)
    #     gen_data_correction(interp_u0, dname, isplot)
    #     gen_new_data_GRF(M, Nx, Ny, N_f, N_b, dname, a_new, l_new, isplot)

    #plot_data("f_T", "u_T", d_num)
    #plot_data("f_0", "u_0", d_num)
    #plot_data("f_new_grid", "u_new_grid", d_num)
    
    if size == 5:
        construct_more_data = construct_more_data5
    elif size == 9:
        construct_more_data = construct_more_data9
    elif size == 13:
        construct_more_data = construct_more_data13_5
    elif size == 25:
        construct_more_data = construct_more_data25_9
    elif size == 49:
        construct_more_data = construct_more_data49_25
    elif size == 81:
        construct_more_data = construct_more_data81_49

    # training data
    f_T = np.loadtxt(f"{dname}/f_T.dat")
    u_T = np.loadtxt(f"{dname}/u_T.dat")
    print(f"Loaded f_T {f_T.shape} and u_T {u_T.shape} for training the local solution operator.")
    d_T = construct_more_data(Nx, Ny, f_T, u_T)
    
    # test data
    f_0 = np.loadtxt(f"{dname}/f_0_grid.dat")
    u_0 = np.loadtxt(f"{dname}/u_0_grid.dat")
    print(f"Loaded f_0_grid {f_0.shape} and u_0_grid {u_0.shape} for testing the local solution operator.")
    d_0 = construct_more_data(Nx, Ny, f_0, u_0)
    
    data_G = dde.data.DataSet(X_train=d_T[0], y_train=d_T[1], X_test=d_0[0], y_test=d_0[1])
    
    if not grid:
        x_train = np.loadtxt(f"{dname}/u_new.dat")[:, 0:2]
        x = np.loadtxt(f"{dname}/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{dname}/y_grid.dat").reshape((-1, 1))
        x_test = np.concatenate((x, t), axis = 1)
        print(f"Loaded u_new {x_train.shape} and u_new_grid {x_test.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))
        
        if correction:
            # For cLOINN-random
            u_new = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            u_init = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1, 1))
            y_test = u_new - u_init
            print("Dataset generated for cLOINN-random (x_train, y_train, x_test, y_test).")
        else:
            # LOINN-random
            y_test = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            print("Dataset generated for LOINN-random (x_train, y_train, x_test, y_test).")
    else:
        x = np.loadtxt(f"{dname}/x_grid.dat").reshape((-1, 1))
        t = np.loadtxt(f"{dname}/y_grid.dat").reshape((-1, 1))
        x_train = np.concatenate((x, t), axis = 1)
        x_test = x_train
        print(f"Loaded u_new_grid {x_train.shape} for x_train and x_test.")
        y_train = np.concatenate(([[0] * len(x_train)])).reshape((-1, 1))

        if correction:
            # For cLOINN-grid
            u_new = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            u_init = np.loadtxt(f"{dname}/u_0_grid.dat").reshape((-1, 1))
            y_test = u_new - u_init
            print("Dataset generated for cLOINN-grid (x_train, y_train, x_test, y_test).")
        else:
            # For FPI and LOINN-grid
            y_test = np.loadtxt(f"{dname}/u_new_grid.dat").reshape((-1, 1))
            print("Dataset generated for FPI/LOINN-grid (x_train, y_train, x_test, y_test).")

    data = dde.data.DataSet(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
    print(x_train.shape, y_test.shape)
    return data_G, data

