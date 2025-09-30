import tensorflow as tf
from keras import layers, models
import numpy as np
import pandas as pd
import h5py

import os
os.chdir('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/dynamics/NeuralODE_lat_dim_7_long_time')
os.getcwd()

# load reduced-order data, together with the same training and testing indices used for training the autoencoder
x_lat = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/x_lat.npy')
idx_train = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/idx_train.npy')
idx_test = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/idx_test.npy')

def load_gust_parameters(AoA, cases, nsnap):
    """
    Load gust disturbance parameters for specified angles of attack (AoA) and gust cases.

    This function reads gust-related parameters stored in `.jld2` files for each 
    (AoA, gust case) combination and aggregates them into arrays for analysis or 
    training. The parameters characterize the spatial and temporal properties of 
    the incoming gusts.

    Args:
        AoA (array-like): List or array of angles of attack to include (e.g., [20, 30, 40, 50, 60]).
        cases (array-like): List or array of gust disturbance identifiers.
        nsnap (int): Number of snapshots per case (not used directly in this function, 
                     but often relevant for downstream alignment with flow data).

    Returns:
        Dy (np.ndarray): Array of gust strength in the y-direction across cases.
        sigma (np.ndarray): Array of gust width (σx) across cases.
        y0 (np.ndarray): Array of initial vertical gust positions across cases.
        t0 (np.ndarray): Array of gust initiation times in the shedding cycle across cases.

    Notes:
        - Each file is expected at path:
          `/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA{num}/AoA{num}_RD{j}.jld2`
        - Parameters are read directly from dataset keys: `Dy`, `σx`, `y0`, and `t0`.
        - Output arrays are ordered consistently with the nested loop over AoA and cases.
    """
    Dy = []
    sigma = []
    y0 = [] 
    t0 = []
    # gust cases
    for num in AoA:
        for j in cases:
            path = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA{num}/AoA{num}_RD{j}.jld2'
            with h5py.File(path, "r") as output:
                Dy.append(output['Dy'][()])
                sigma.append(output['σx'][()])
                y0.append(output['y0'][()])
                t0.append(output['t0'][()])
    Dy = np.array(Dy)
    sigma = np.array(sigma)
    y0 = np.array(y0)
    t0 = np.array(t0)
    return Dy, sigma, y0, t0


def prepare_data(x_lat_cases, AoA:np.ndarray, n_cases, t0, dt):
    """
    Prepare latent trajectory data by extracting fixed-length segments 
    from base and gust-disturbed aerodynamic cases.

    For base cases (first `len(AoA)` entries), the function takes the first 
    200 snapshots directly. For gust-disturbed cases, it extracts 200 snapshots 
    starting at an offset determined by the gust onset time `t0` relative to 
    the time step `dt` (with an additional shift of +10 steps).

    Args:
        x_lat_cases (np.ndarray): Latent variable trajectories for all cases, 
            shape (n_cases, nsnap, n), where nsnap is the number of snapshots 
            and n is the latent dimension. n_cases is the total number of independent cases.
        AoA (np.ndarray): Array of base angles of attack.
        n_cases (int): Total number of cases (base + gust).
        t0 (np.ndarray): Gust onset times for gust cases, shape (n_gust_cases,).
        dt (float): Time step size between snapshots.

    Returns:
        np.ndarray: Array of processed latent trajectories, shape 
        (n_cases, 200, n), where each case is truncated or shifted 
        to a fixed 200-snapshot segment.
    """
    idx_start = (t0 / dt).astype(int) + 10
    idx_end = idx_start + 200
    n_AoA = len(AoA)
    x_lat = []
    for i in range(len(AoA)):
        x_lat.append(x_lat_cases[i,0:200,:])
    for i in range(5,n_cases):
        x_lat.append(x_lat_cases[i,idx_start[i-n_AoA]:idx_end[i-n_AoA],:])
    return np.array(x_lat)

AoA = np.array([20, 30, 40, 50, 60])
n_cases = len(AoA)*61           # 5 base cases + 5*60 gust cases
cases = np.arange(1,61)         # 60 independent gust cases for each AoA
nsnap = 500                     # 500 nsnapshots for each case
Dy, sigma, y0, t0 = load_gust_parameters(AoA, cases, nsnap)
with h5py.File('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA20/AoA20_RD1.jld2', "r") as output:
    dt = output['Δt'][()]
dt *= 20    # time step between snapshots in the latent space (every 20 CFD time steps)

lat_dim = x_lat.shape[-1]
x_lat_cases = x_lat.reshape(n_cases, nsnap, lat_dim)    # shape: (n_cases, nsnap, lat_dim)
x_lat_data = prepare_data(x_lat_cases, AoA, n_cases, t0, dt)    # shape: (n_cases, 200, lat_dim)
nsnap = x_lat_data.shape[1]

# Augment latent states with one-hot encoded AoA to distinguish different AoA conditions
AoA_cases = np.repeat(AoA, n_cases-len(AoA))
AoA_cases = np.concatenate([AoA, AoA_cases])    # repeat each AoA value enough times to cover the gust cases, and then concatenate with the original AoA.
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
AoA_onehot = onehot_encoder.fit_transform(AoA_cases.reshape(-1, 1)) # Each AoA value is transformed into a one-hot vector. Shape: (n_cases, len(AoA))
idx_case = np.arange(n_cases)
AoA_cases = []
for idx in idx_case:
    AoA_cases.append(np.tile(AoA_onehot[idx], (nsnap, 1)))  # Expand one-hot AoA across time snapshots
AoA_cases = np.array(AoA_cases) # shape: (n_cases, nsnap, len(AoA))

x_lat_data = np.concatenate([x_lat_data, AoA_cases], axis=-1)   # Augment latent states with one-hot AoA. New shape: (n_cases, nsnap, lat_dim + len(AoA))

x_lat_data_train = x_lat_data[idx_train]
x_lat_data_test = x_lat_data[idx_test]
initial_lag = 0
x_lat_data_train = x_lat_data_train[:,initial_lag:,:]
x_lat_data_test = x_lat_data_test[:,initial_lag:,:]
nsnap_lagged = nsnap - initial_lag


# Training Neural ODE to learn latent dynamics
class LatentODEFunc(tf.keras.Model):
    def __init__(self, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim//2, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim//2, activation='tanh'),
            tf.keras.layers.Dense(lat_dim)  # Output dx/dt
        ])

    def call(self, x):
        return self.net(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def ode_integrate(f, x0, t, dt=1.0):
    xs = [x0[:, :lat_dim]]
    x = x0
    for _ in range(t-1):
        x_lat = x[:, :lat_dim]
        x_AoA = x[:, lat_dim:]
        dx = f(x)
        x_lat = x_lat + dt * dx
        x = tf.concat([x_lat, x_AoA], axis=-1)
        xs.append(x_lat)
    return tf.stack(xs, axis=1)  # shape: (batch, nsnap, lat_dim)

def calculate_loss(x_true, x_pred, time_weight=False):
    nsnap = x_pred.shape[1]
    if time_weight:
        time_weights = tf.linspace(1.0, 0.5, nsnap)
        time_weights = tf.reshape(time_weights, (1, nsnap, 1))
    else:
        time_weights = 1.0
    loss = tf.square(x_true - x_pred)
    return tf.reduce_mean(time_weights * loss)

epochs = 5000
history_data = {'epoch': [], 'train_loss': [], 'val_loss': []}
model_path = './forward_model.keras'  # Path to save and load the model
patience = 500
train_loss_log = []
val_loss_log = []

def train_latent_ode(x_lat_train, x_lat_test, model, optimizer, epochs=500, alpha=1.0, beta=1.0):
    best_val_loss = float('inf')  # Track best validation loss
    n_cases, nsnap, _ = x_lat_train.shape
    _, nsnap_test, _ = x_lat_test.shape
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            x0 = x_lat_train[:, 0, :]  # shape: (n_cases, lat_dim+len(AoA))
            x_pred = ode_integrate(model, x0, nsnap, dt=dt)  # shape: (n_cases, nsnap, lat_dim) long-term prediction from initial state
            
            # Full rollout reconstruction losses
            loss_reconstruction = calculate_loss(x_lat_train[:,:,:lat_dim], x_pred, time_weight=True)  # MSE loss over rollout
            
            x_pred_1 = [x_lat_train[:,0,:lat_dim]]
            for t in range(nsnap-1):
                x = ode_integrate(model, x_lat_train[:,t,:], 2, dt=dt)  # one-step prediction from each state
                x_pred_1.append(x[:,-1,:])
            x_pred_1 = tf.stack(x_pred_1, axis=1)
            loss_reconstruction_1 = calculate_loss(x_lat_train[:,:,:lat_dim], x_pred_1)  # MSE loss over one step
            
            train_loss = loss_reconstruction + beta * loss_reconstruction_1

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        x0 = x_lat_test[:, 0, :]  # shape: (n_cases, lat_dim+len(AoA))
        x_pred = ode_integrate(model, x0, nsnap_test, dt=dt)  # shape: (n_cases, nsnap, lat_dim)
        
        # Full rollout reconstruction losses
        loss_reconstruction_test = calculate_loss(x_lat_test[:,:,:lat_dim], x_pred, time_weight=True)  # MSE loss over rollout
        
        x_pred_1 = [x_lat_test[:,0,:lat_dim]]
        for t in range(nsnap_test-1):
            x = ode_integrate(model, x_lat_test[:,t,:], 2, dt=dt)
            x_pred_1.append(x[:,-1,:])
        x_pred_1 = tf.stack(x_pred_1, axis=1)
        loss_reconstruction_1_test = calculate_loss(x_lat_test[:,:,:lat_dim], x_pred_1)  # MSE loss over one step
            
        val_loss = loss_reconstruction_test + beta * loss_reconstruction_1_test
        print(f"Epoch {epoch} | Loss: {train_loss.numpy():.6f} | reconst_loss: {loss_reconstruction.numpy():0.6f} | one_step_loss: {beta*loss_reconstruction_1.numpy():0.6f} | Val_loss: {val_loss.numpy():0.6f}")
        
        history_data['epoch'].append(epoch)
        history_data['train_loss'].append(train_loss)
        history_data['val_loss'].append(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            print(f"Validation loss dropped from {best_val_loss:.6f} to {val_loss:.6f}. Saving the model.", flush=True)
            best_val_loss = val_loss
            patience_counter = 0
            model.save(model_path)  # Save best model
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.6f}", flush=True)
            break
    df_results = pd.DataFrame(history_data)
    df_results.to_csv('./history.csv', index=False)

alpha = 1.0
beta = 150.0
x_lat_train = tf.convert_to_tensor(x_lat_data_train, dtype=tf.float32)
x_lat_test = tf.convert_to_tensor(x_lat_data_test, dtype=tf.float32)
latent_dynamics_model = LatentODEFunc()
optimizer = tf.keras.optimizers.Adam(amsgrad=True)
train_latent_ode(x_lat_train, x_lat_test, latent_dynamics_model, optimizer, epochs=epochs, alpha=alpha, beta=beta)


# Evaluation on test data
forward_model = tf.keras.models.load_model('forward_model.keras', custom_objects={'LatentODEFunc': LatentODEFunc})

def autoregressive_rollout(model, x0, nsnap, dt, lat_dim=7):
    """
    Autoregressively predict latent states one step at a time.
    
    Args:
        model: Trained LatentODEFunc model
        x0: Initial input [xi_0, AoA] of shape (lat_dim + AoA_dim,)
        nsnap: Number of time steps to predict
        dt: Time step size
        lat_dim: Dimension of latent space (default 3)

    Returns:
        latent trajectory: shape (batch_size, nsnap, lat_dim)
    """
    x0 = tf.expand_dims(x0, axis=0)
    xs = [tf.identity(x0[:, :lat_dim])]  # Start with the initial latent only
    x = x0  # Full input = [xi_0; AoA]
    
    for _ in range(nsnap - 1):
        dx = model(x)  # shape: (batch_size, lat_dim)
        x_lat = x[:, :lat_dim] + dt * dx  # Update only the latent variables
        x = tf.concat([x_lat, x[:, lat_dim:]], axis=-1)  # Reassemble with AoA
        xs.append(x_lat)
    
    return tf.stack(xs, axis=1)  # shape: (batch, nsnap, lat_dim)

def one_step_prediction(model, x_true, dt, lat_dim=7):
    """
    Autoregressively predict latent states one step at a time.
    
    Args:
        model: Trained LatentODEFunc model
        x0: Initial input [xi_0, AoA] of shape (lat_dim + AoA_dim,)
        nsnap: Number of time steps to predict
        dt: Time step size
        lat_dim: Dimension of latent space (default 3)

    Returns:
        latent trajectory: shape (batch_size, nsnap, lat_dim)
    """
    nsnap = x_true.shape[0]
    x_true_ = tf.expand_dims(x_true, axis=0)
    x_pred_1 = [x_true_[:,0,:lat_dim]]
    x = x_true_[:,0,:]
    for t in range(nsnap-1):
        dx = model(x_true_[:,t,:])
        x_lat = x[:,:lat_dim] + dt * dx
        x = x_true_[:,t+1,:]
        x_pred_1.append(x_lat)
    
    return tf.stack(x_pred_1, axis=1)  # shape: (batch, nsnap, lat_dim)

x_pred_cases = []
for i in range(len(idx_test)):
    x0 = x_lat_test[i,0]
    x_lat_pred_ = autoregressive_rollout(forward_model, x0, nsnap_lagged, dt, lat_dim=lat_dim)
    x_pred_cases.append(x_lat_pred_.numpy().copy())
x_pred_cases = np.array(x_pred_cases)
x_pred_cases = np.squeeze(x_pred_cases, axis=1)

# computing empirical residual covariance 7*7
error = x_pred_cases - x_lat_test[:,:,:lat_dim]
error = tf.reshape(error,(-1,error.shape[-1]))
error_centered = error - tf.reduce_mean(error, axis=0, keepdims=True)
N = tf.shape(error)[0]
Q = tf.matmul(error_centered, error_centered, transpose_a=True) / tf.cast(N - 1, tf.float32)
np.save('Q.npy', Q)