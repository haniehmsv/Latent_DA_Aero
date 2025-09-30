import numpy as np
import tensorflow as tf
from typing import Callable, Any
from dataclasses import dataclass
from tqdm import tqdm  # For progress tracking
import h5py
import os
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

import sys
sys.path.append('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/LREnKF/LREnKF_lat_dim_7_long_time/process_noise_Q')
from LREnKF import *

def load_pressure(AoA:np.ndarray, ranget, rangep, cases:np.ndarray):
    # base cases
    y_pres_list = []
    for num in AoA:
        presdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/pressure/a{num}/Pressure_AoA{num}_base.jld'
        output = h5py.File(presdir, "r")
        output = output['pres_box']
        output = output[:]
        y_pres_list.append(output[ranget,rangep])
        
    # Gust cases
    for num in AoA:
        for j in cases:
            presdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/pressure/a{num}/Pressure_AoA{num}_RD{j}.jld'
            output = h5py.File(presdir, "r")
            output = output['pres_box']
            output = output[:]
            y_pres_list.append(output[ranget,rangep])
    
    y_pres = np.concatenate(y_pres_list, axis=0)
    return y_pres

def load_lift(AoA:np.ndarray, ranget, cases:np.ndarray):
    # base cases
    y_CL_list = []
    for num in AoA:
        liftdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/lift/a{num}/Lift_AoA{num}_base.jld'
        output = h5py.File(liftdir, "r")
        output = output['lift_box']
        output = output[:]
        y_CL_list.append((output.transpose(1, 0))[ranget])
        
    # Gust cases
    for num in AoA:
        for j in cases:        
            liftdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/lift/a{num}/Lift_AoA{num}_RD{j}.jld'
            output = h5py.File(liftdir, "r")
            output = output['lift_box']
            output = output[:]
            y_CL_list.append((output.transpose(1, 0))[ranget])
    
    y_CL = np.concatenate(y_CL_list, axis=0)
    return y_CL

idx_test = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/idx_test.npy')
x_lat_true = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/x_lat.npy')

nsnap = 500
x_lat = x_lat_true.copy()
lat_dim = x_lat.shape[-1]
n_cases = x_lat.shape[0] // nsnap   # Total number of independent gust cases
x_lat_cases = x_lat.reshape(n_cases, nsnap, x_lat.shape[1]) # shape (n_cases, nsnap, lat_dim)
cases = np.arange(1,61)             # 60 independent gust cases for each AoA
ranget = slice(1,501,1)             # 500 nsnapshots for each case
rangep = slice(4,69,6)              # 11 pressure sensors
AoA = np.array([20, 30, 40, 50, 60])
CL = load_lift(AoA, ranget, cases)
CL_cases = CL.reshape(n_cases, nsnap, 1)
Y_pres = load_pressure(AoA, ranget, rangep, cases)
Y_pres_cases = Y_pres.reshape(n_cases, nsnap, Y_pres.shape[-1])
initial_lag = 0
Nsens = Y_pres_cases.shape[-1]

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

forecast = tf.keras.models.load_model('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/dynamics/NeuralODE_lat_dim_7_long_time/model.keras', custom_objects={'LatentODEFunc': LatentODEFunc})
model_decod = tf.keras.models.load_model('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/model.keras')
observation = Model(inputs=model_decod.get_layer('dense_2').output, outputs=model_decod.get_layer('dense_6').output)
decoder_CL = Model(inputs=model_decod.get_layer('dense_2').output, outputs=model_decod.get_layer('dense_7').output)


augmentation_status = True
if augmentation_status:
    # Augmenting the encoded AOA as an auxiliary variable
    AoA_cases = np.repeat(AoA, 60)
    AoA_cases = np.concatenate([AoA, AoA_cases])
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    AoA_onehot = onehot_encoder.fit_transform(AoA_cases.reshape(-1, 1))
    idx_case = np.arange(n_cases)
    AoA_cases = []
    for idx in idx_case:
        AoA_cases.append(np.tile(AoA_onehot[idx], (nsnap, 1)))
    AoA_cases = np.array(AoA_cases)

idx_test = np.sort(idx_test)
test_idx = [3, 23, 46, 62, 84]  # sample test cases at AoA = 20, 30, 40, 50, 60 respectively

## construct state ensemble
idx_case = idx_test[test_idx[4]]    # test case at AoA = 60
C_L = CL_cases[idx_case,initial_lag:,0]
set_x_lat_dim(x_lat_true.shape[-1])
Ne = 200                            # ensemble size 
x0 = x_lat_cases[idx_case,initial_lag]
AoA_enc = AoA_cases[idx_case,0]
x0 = np.concatenate([x0,AoA_enc], axis=0)
x_true = x_lat_cases[idx_case,initial_lag:]
Sigma_x = np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/dynamics/NeuralODE_lat_dim_7_long_time/Q.npy')    # Process noise covariance
X0 = create_initial_ensemble(x0, Ne, 1e-2)   #shape (Ne,1,lat_dim+len(AoA))
fdata = ForecastData(x_lat_true.shape[-1], Sigma_x)

## Construct observation ensemble
sigma_eps = (np.load('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time/std_meas.npy')).item() # Observation noise standard deviation
y_truth = Y_pres_cases[idx_case,initial_lag:]
xsens0 = []
ysens0 = []
sens = Sensor(xsens0, ysens0, Nsens)
odata = ObservationData(sens,sigma_eps,y_truth, observation)

with h5py.File('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA20/AoA20_RD1.jld2', "r") as output:
    dt = output['Δt'][()]
dt *= 20    # Adjusting time step according to the snapshot frequency

## Sequential estimation
dt_dyn = dt
dt_obs = dt_dyn
tspan = (0,nsnap-initial_lag)
lowrankEnKF = LREnKFParameters(forecast, observation, fdata, odata, dt_dyn, dt_obs, tspan, Ne=Ne)
Xf, Xa, Cx_history, Cy_history, rxhist, ryhist = lrenkf(lowrankEnKF, X0.copy(), beta=1.0)

np.savez('AoA60_test.npz', Xa=Xa, Xf=Xf, Cx_history=Cx_history, Cy_history=Cy_history, rxhist=rxhist, ryhist=ryhist)


## Then, we only need to lift the latent state analysis ensemble (Xa) back to the full state space thriugh the pretrained decoder.