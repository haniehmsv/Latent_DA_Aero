import numpy as np

# Load the stored Cx and Cy history

tsnap = [100,120,140,160]       # choose snapshots at which to perform SVD and analyze the Gramians
Vx, Vy, Lambda_x, Lambda_y = [], [], [], []
for t in tsnap:
    u_x, s_x, _ = np.linalg.svd(Cx_history[case_name][t])  
    Lambda_x.append(s_x.copy())
    Vx.append(u_x.copy())                   # State space basis vectors (left singular vectors of Cx)
    u_y, s_y, _ = np.linalg.svd(Cy_history[case_name][t])  
    Lambda_y.append(s_y.copy())
    Vy.append(u_y.copy())        # Observation space basis vectors (left singular vectors of Cy)

Vx = np.array(Vx)
Vy = np.array(Vy)
Lambda_x = np.array(Lambda_x)
Lambda_y = np.array(Lambda_y)