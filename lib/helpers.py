import numpy as np

def create_data_cube(data,input_dim=24, output_dim = 12, timesteps=720):
    m = len(data)
    
    A = np.empty((m,input_dim))
    B = np.empty((m, output_dim))
    
    try:
        for i in range(m):
            window = data[i+input_dim:i:-1]
            A[i,:] = window
            B[i,:] = data[i+input_dim:i+input_dim+output_dim]
    except:
        A = A[:i,:]
    X = np.empty((i-timesteps, timesteps, input_dim))
    y = np.empty((i-timesteps, output_dim))
    for j in range(timesteps):
        X[:,j,:] = A[j:i-(timesteps-j),:]
        
    for a in range(i-(timesteps)):
        y[a,:] = B[a+timesteps,:]
    A = None
    B = None
    return X,y
