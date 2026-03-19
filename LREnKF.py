import numpy as np
import tensorflow as tf
from typing import Callable, Any
from dataclasses import dataclass
from tqdm import tqdm  # For progress tracking
import h5py
from keras.models import Model

x_lat_dim = None

RXDEFAULT = 100
RYDEFAULT = 100
DEFAULT_ADAPTIVE_RATIO = 0.99
NEDEFAULT = 100

model_decod = tf.keras.models.load_model('autoencoder_model.keras')
decoder = Model(inputs=model_decod.get_layer('dense_2').output, outputs=model_decod.get_layer('conv2d_16').output)
decoder_CL = Model(inputs=model_decod.get_layer('dense_2').output, outputs=model_decod.get_layer('dense_7').output)

def set_x_lat_dim(value):
    """Setter function to allow external modification of x_lat_dim."""
    global x_lat_dim
    x_lat_dim = value

@dataclass
class StochEnKFParameters:
    """
    A structure for parameters of the stochastic ensemble Kalman filter (sEnKF).

    Attributes:
    - forecast: Forecast operator
    - observation: Observation operator
    - fdata: Forecast data
    - odata: Observation data
    - dt_dyn: Forecast time step
    - dt_obs: Observation time step
    - tspan: (t0,t_f)
    - Lyy: Sensor localization distance (default: 1e10)
    - Lxy: State-to-sensor localization distance (default: 1e10)
    - islocal: True if localization is desired (default: False)
    """
    forecast: Callable
    observation: Callable
    fdata: Any
    odata: Any
    dt_dyn: float
    dt_obs: float
    tspan: Any
    Lyy: float = 1e10
    Lxy: float = 1e10
    islocal: bool = False
    Ne: int = NEDEFAULT

    def __str__(self):
        return f"Stochastic EnKF with localization = {self.islocal}"

@dataclass
class LREnKFParameters:
    """
    A structure for parameters of the stochastic ensemble Kalman filter (sEnKF).

    Attributes:
    - forecast: Forecast operator
    - observation: Observation operator
    - fdata: Forecast data
    - odata: Observation data
    - dt_dyn: Forecast time step
    - dt_obs: Observation time step
    - tspan: (t0,t_f)
    - rxdefault: Truncated dimension of the informative subspace of the state space
    - rydefault: Truncated dimension of the informative subspace of the observations space
    - ratio: Ratio of cumulative energy of the state and observation Gramians to retain
    - Ne: Number of ensemble particles to calculate Gramians
    """
    forecast: Callable
    observation: Callable
    fdata: Any
    odata: Any
    dt_dyn: float
    dt_obs: float
    tspan: Any
    rxdefault:int = RXDEFAULT
    rydefault:int = RYDEFAULT
    ratio:float = DEFAULT_ADAPTIVE_RATIO
    Ne: int = NEDEFAULT
    
    
class ForecastData:
    """A structure for parameters of forecast data

    Args:
        Nx (int): number of states
        sigma_x (float): standard deviation of state for inflation and model noise addition purposes
    """
    def __init__(self, Nx:int, Sigma_x):
        super().__init__()  # Call the constructor of the superclass
        self.Nx = Nx
        self.Sigma_x = Sigma_x

class Sensor:
    """A class for sensor measurements

    Args:
        x (np.ndarray): x location of sensors
        y (np.ndarray): y location of sensors
        Nsens (int): number of measurements
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, Nsens: int):
        self.x = x
        self.y = y
        self.Nsens = Nsens

def generate_synthetic_data(y_truth, sigma_eps: float) -> np.ndarray:
    noise = np.random.normal(loc=0, scale=sigma_eps, size=y_truth.shape)  #shape (Nt,Ny)
    return y_truth + noise


class ObservationData:
    """A class for parameters of observation data

    Args:
        sens (Sensor): coordinates and number of sensor measurements
        sigma_eps (float): standard deviation of measurement noise
        obs_data (np.ndarray): truth measured data using synthetic experiment
        net: neural network for mapping from latent space to observation space
    """
    def __init__(self, sens: Sensor, sigma_eps: float, y_truth: np.ndarray, net):
        self.sens = sens
        self.sigma_eps = sigma_eps
        self.obs_data = generate_synthetic_data(y_truth, sigma_eps)
        self.net = net
    
    def calculate_jacobian(self, x):
        global x_lat_dim
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        Ny = self.sens.Nsens
        Nx = x_lat_dim
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.net(x)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
        jac = tape.jacobian(y,x)
        return (tf.einsum('bxby->bxy', jac).numpy())[:,:Ny,:Nx]

def measurement_length(odata:ObservationData) -> int:
    return odata.sens.Nsens

def create_initial_ensemble(x0, Ne, sigma_x):
    global x_lat_dim
    if len(x0) == x_lat_dim:
        return x0 + np.random.normal(loc=0, scale=sigma_x, size=(Ne,x_lat_dim))
    else:
        return x0 + np.concatenate([np.random.normal(loc=0, scale=sigma_x, size=(Ne,x_lat_dim)), np.zeros((Ne,len(x0)-x_lat_dim))], axis=1)

def create_ensemble(Ne, mean, Sigma_x):
    return np.random.multivariate_normal(mean=mean, cov=Sigma_x, size=Ne)

def create_ensemble_diag(Ne, mean, sigma_diag):
    std = np.sqrt(sigma_diag)
    return mean[None, :] + np.random.randn(Ne, mean.size) * std[None, :]
    
def additive_inflation(X: np.ndarray, Sigma_x: np.ndarray):
    """
    additive_inflation!(X,sigma_x)
    Add to `X` random noise drawn from a Gaussian distribution with
    zero mean and variance given by `Sigma_x`.
    """
    Ne, Nx = X.shape
    noise = np.random.multivariate_normal(mean=np.zeros(Nx), cov=Sigma_x, size=Ne)  # shape (Ne, Nx)
    return X + noise

def ensemble_perturb(X: np.ndarray):
    return X - np.mean(X,axis=0)

def multiplicative_inflation(X: np.ndarray, β: float):
    """
    multiplicative_inflation!(X,β)
    Carry out the operation ``\\bar{x} + \\beta(x^j - \\bar{x})`` for every ensemble
    member in `X`.
    """
    X = np.mean(X,axis=0) + β*ensemble_perturb(X)
    return X

def norm(X:np.ndarray,Sigma:np.ndarray):
    out = 0.0
    Sigma_inv = np.linalg.inv(Sigma)
    for x in X:
        x = x.reshape(-1, 1)    #convert to a column vector
        norm_value = (x.T @ Sigma_inv @ x).item()
        out += norm_value
    return np.sqrt(out / X.shape[0])

def cross_covariance(X:np.ndarray,Y:np.ndarray):
    Ne = X.shape[0]
    X_perturb = ensemble_perturb(X)
    Y_perturb = ensemble_perturb(Y)
    return (X_perturb.T @ Y_perturb) / (Ne-1)   #shape (Nx,Ny)

def whiten(X:np.ndarray,Sigma:np.ndarray):
    """
    whiten(X:ndarray,Sigma)

    Remove the mean from the ensemble data in `X` and left-multiply each member by ``Sigma^{-1/2}``
    """
    return (np.linalg.inv(np.sqrt(Sigma))@(ensemble_perturb(X).T)).T

def whiten_diag(X:np.ndarray, diag_Sigma:np.ndarray):
    """
    whiten(X:ndarray, diag_Sigma)

    Remove the mean from the ensemble data in `X` and left-multiply each member by ``Sigma^{-1/2}``
    """
    inv_sqrt_diag = 1.0 / np.sqrt(diag_Sigma)   # [Ny]
    return ensemble_perturb(X) * inv_sqrt_diag[None, :]      # [Ne, Ny]

def allocate_jacobian(Nx,Ny,algo:LREnKFParameters):
    return np.zeros((Ny,Nx))
def allocate_state_gramian(Nx,algo:LREnKFParameters):
    return np.zeros((Nx,Nx))
def allocate_observation_gramian(Ny,algo:LREnKFParameters):
    return np.zeros((Ny,Ny))

def jacob(X,odata):
    return odata.calculate_jacobian(X)

def gramians(Cx, Cy, H, odata, Sigma_eps, X, Sigma_x, Ne):
    Cx_new = np.zeros_like(Cx)
    Cy_new = np.zeros_like(Cy)
    
    invD_eps = np.linalg.inv(np.sqrt(Sigma_eps))
    Dx = np.sqrt(Sigma_x)
    fact = min(1.0,1.0/(Ne-1))
    Jac = jacob(X,odata)
    
    for j in range(Ne):
        H = Jac[j]
        H = invD_eps @ H @ Dx
        
        Cx_new += H.T @ H
        Cy_new += H @ H.T
        
    Cx_new *= fact
    Cy_new *= fact
    return Cx_new, Cy_new

def lrenkf_kalman_update(algo,X_ens, Y_ens, ystar, Sigma_x, Sigma_eps, fdata, odata, Cx_history, Cy_history, rxhist, ryhist, Cx, Cy, Jac,eps):
    global x_lat_dim
    rxdefault, rydefault, ratio, Ne = algo.rxdefault, algo.rydefault, algo.ratio, algo.Ne
    Nx = fdata.Nx
    Ny = measurement_length(odata)
    Cx, Cy = gramians(Cx, Cy, Jac, odata, Sigma_eps, X_ens, Sigma_x, Ne)

    Cx_sym = (Cx + Cx.T) / 2  # Ensure Cx is symmetric
    Cy_sym = (Cy + Cy.T) / 2  # Ensure Cy is symmetric
    V, Lambda_x, _ = np.linalg.svd(Cx_sym)  #V.shape=(Nx,Nx)
    U, Lambda_y, _ = np.linalg.svd(Cy_sym)  #U.shape=(Ny,Ny)
    
    ry = min(Ny, rydefault)
    rx = min(Nx, rxdefault)
    
    if ratio < 1.0:
        # Find the first index where the cumulative sum exceeds the ratio (similar to findfirst in Julia)
        cum_x = np.cumsum(Lambda_x) / np.sum(Lambda_x)
        cum_y = np.cumsum(Lambda_y) / np.sum(Lambda_y)
        rx = next((i+1 for i, x in enumerate(cum_x) if x >= ratio), rxdefault)
        ry = next((i+1 for i, x in enumerate(cum_y) if x >= ratio), rydefault)
    rx = 1 if rx is None else rx
    ry = 1 if ry is None else ry
    rxhist.append(rx)
    ryhist.append(ry)
    
    # rank reduction
    Vr = V[:, :rx]  #shape (Nx,rx)
    Ur = U[:, :ry]  #shape (Ny,ry)
    X_lat_ens = X_ens[:,:x_lat_dim]
    
    innov = ystar[None,:] - Y_ens + eps   #shape (Ne,Ny)
    Y_hat = (Ur.T @ np.linalg.inv(np.sqrt(Sigma_eps)) @ innov.T).T   # shape (Ne,ry)
    Xp = (Vr.T @ whiten(X_lat_ens,Sigma_x).T).T     #shape (Ne,rx)
    HXp = (Ur.T @ whiten(Y_ens,Sigma_eps).T).T      #shape (Ne,ry)
    eps_p = (Ur.T @ whiten(eps,Sigma_eps).T).T      #shape (Ne,ry)
    
    CHH = cross_covariance(HXp, HXp)  #shape (ry,ry)
    Cee = cross_covariance(eps_p, eps_p)  #shape (ry,ry)
    
    Sigma_Y_tilde = CHH + Cee   #shape (ry,ry)
    Sigma_XY_tilde = cross_covariance(Xp, HXp) #shape (rx,ry)
    sqrt_Sigma_x = np.sqrt(Sigma_x)     #shape (Nx,Nx)
    
    X_lat_ens += (sqrt_Sigma_x @ Vr @ Sigma_XY_tilde @ np.linalg.solve(Sigma_Y_tilde,Y_hat.T)).T    #shape (Ne,Nx)
    
    Cx_history.append(Cx.copy())
    Cy_history.append(Cy.copy())
    return X_lat_ens, Cx_history, Cy_history, rxhist, ryhist



def senkf_kalman_update(algo,X_ens, Y_ens, ystar, sigma_x_diag, sigma_eps_diag, fdata, odata, eps):
    global x_lat_dim
    sqrt_Sigma_x = np.sqrt(sigma_x_diag)          # shape (Nx,)
    inv_sqrt_Sigma_eps = 1.0 / np.sqrt(sigma_eps_diag)   # shape (Ny,)
    
    innov = ystar[None,:] - Y_ens + eps  #shape (Ne,Ny)
    Y_hat_T = inv_sqrt_Sigma_eps[:, None] * innov.T    # shape (Ny,Ne)
    
    Xp = whiten_diag(X_ens, sigma_x_diag)     #shape (Ne,Nx)
    HXp = whiten_diag(Y_ens,sigma_eps_diag)      #shape (Ne,Ny)
    # eps_p = whiten_diag(eps,sigma_eps_diag)      #shape (Ne,Ny)
    
    CHH = cross_covariance(HXp, HXp)  #shape (Ny,Ny)
    # Cee = cross_covariance(eps_p, eps_p)  #shape (Ny,Ny)
    Ny  = Y_ens.shape[1]
    I = np.eye(Ny, dtype=Y_ens.dtype)
    Sigma_Y_tilde  = CHH + I
    # Sigma_Y_tilde = CHH + Cee   #shape (Ny,Ny)
    
    Sigma_XY_tilde = cross_covariance(Xp, HXp) #shape (Nx,Ny)
    
    X_ens += ((sqrt_Sigma_x[:, None] * Sigma_XY_tilde) @ np.linalg.solve(Sigma_Y_tilde, Y_hat_T)).T      #shape (Ne,Nx)
    
    return X_ens

def enkf(algo:StochEnKFParameters, X, inflate=True, beta=1.0):
    return _enkf(algo, X, inflate, beta)


def _enkf(algo:LREnKFParameters, X, inflate, beta):
    """
    Perform Low-rank Ensemble Kalman Filter (EnKF) assimilation of pressure observations.

    Parameters:
    - algo: StochEnKFParameters
    - X: ndarray, initial ensemble tensor of state distribution augmented by the encoded AoA. Shape: (Ne, Nx+len(AoA))
    - inflate: bool, whether to apply inflation
    - beta: float, multiplicative inflation parameter

    Returns:
    - Xf: list, forecasted ensemble matrices
    - Cx_history: list, history of state Gramians
    - Cy_history: list, history of observation Gramians
    - rxhist: list, history of truncated dimensions of the state Gramians
    - ryhist: list, history of truncated dimensions of the observation Gramians
    """
    global x_lat_dim
    forecast, observation, fdata, odata, dt_dyn, dt_obs, tspan = algo.forecast, algo.observation, algo.fdata, algo.odata, algo.dt_dyn, algo.dt_obs, algo.tspan
    Nx = fdata.Nx
    Ny = measurement_length(odata)
    ytrue = odata.obs_data
    Xf, Xa = [X[:,:x_lat_dim].copy()], [X[:,:x_lat_dim].copy()]
    ystar = np.zeros((Ny))
    t0, t_f = tspan
    step = int(np.ceil(dt_obs / dt_dyn))
    n0 = int(np.ceil(t0 / dt_obs))
    n_dyn = int((t_f - t0)) - 1
    Dcycle = range(n0,n_dyn)
    
    AoA_enc = X[0,x_lat_dim:]
    sigma_eps_diag = np.full(Ny, odata.sigma_eps**2)   # shape (Ny,)
    
    # Run the EnKF
    for i in tqdm(Dcycle, desc="EnKF Progress"):
        # Forecast step
        if i==0 or (i % step == 0):
            dX_lat = forecast(X, training=False).numpy()    # X.shape: (Ne,Nx+len(AoA))
            X_lat = X[:, :x_lat_dim] + dt_dyn * dX_lat  # Update only the latent variables
            X_lat = additive_inflation(X_lat, fdata.Sigma_x)
            Xf.append(X_lat.copy())
            ystar = ytrue[i+1]
            
            if inflate:
                X_lat = multiplicative_inflation(X_lat, beta)
            
            # Compute marginally the variance of the state ensemble
            std_dev = np.std(X_lat, axis=0)
            # Sigma_x = np.diag(std_dev ** 2)
            sigma_x_diag = std_dev ** 2
            
            # Observation update
            Y = observation(X_lat, training=False).numpy() #shape (Ne,Ny)
            # Sigma_eps = np.diag(np.ones(Ny)*odata.sigma_eps**2)  #shape (Ny,Ny)
            eps = create_ensemble_diag(algo.Ne,np.zeros(Ny),sigma_eps_diag)   #shape (Ne,Ny)
            X_lat =  senkf_kalman_update(algo,X_lat,Y,ystar,sigma_x_diag,sigma_eps_diag,fdata,odata,eps)
            
            Xa.append(X_lat.copy())
            X[:, :x_lat_dim] = X_lat
        else:
            dX_lat = forecast(X, training=False).numpy()    # X.shape: (Ne,Nx+len(AoA))
            X_lat = X[:, :x_lat_dim] + dt_dyn * dX_lat  # Update only the latent variables
            X_lat = additive_inflation(X_lat, fdata.Sigma_x)
            Xf.append(X_lat.copy())
            X[:, :x_lat_dim] = X_lat
    return Xf, Xa


def lrenkf(algo:LREnKFParameters, X, inflate=True, beta=1.0):
    return _lrenkf_CL_data(algo, X, inflate, beta)

def _lrenkf_CL_data(algo:LREnKFParameters, X, inflate, beta):
    """
    Perform Low-rank Ensemble Kalman Filter (EnKF) assimilation of pressure observations.

    Parameters:
    - algo: StochEnKFParameters
    - X: ndarray, initial ensemble tensor of state distribution augmented by the encoded AoA. Shape: (Ne, Nx+len(AoA))
    - inflate: bool, whether to apply inflation
    - beta: float, multiplicative inflation parameter

    Returns:
    - Xf: list, forecasted ensemble matrices
    - Cx_history: list, history of state Gramians
    - Cy_history: list, history of observation Gramians
    - rxhist: list, history of truncated dimensions of the state Gramians
    - ryhist: list, history of truncated dimensions of the observation Gramians
    """
    global x_lat_dim
    forecast, observation, fdata, odata, dt_dyn, dt_obs, tspan = algo.forecast, algo.observation, algo.fdata, algo.odata, algo.dt_dyn, algo.dt_obs, algo.tspan
    Nx = fdata.Nx
    Ny = measurement_length(odata)
    ytrue = odata.obs_data
    Xf, Xa = [X[:,:x_lat_dim].copy()], [X[:,:x_lat_dim].copy()]
    ystar = np.zeros((Ny))
    t0, t_f = tspan
    step = int(np.ceil(dt_obs / dt_dyn))
    n0 = int(np.ceil(t0 / dt_obs))
    n_dyn = int((t_f - t0)) - 1
    Dcycle = range(n0,n_dyn)
    
    AoA_enc = X[0,x_lat_dim:]
    
    # Pre-allocate the Jacobian, state and observation Gramians
    Jac = allocate_jacobian(fdata.Nx,Ny,algo)   #shape (Ny,Nx)
    Cx = allocate_state_gramian(fdata.Nx,algo)  #shape (Nx,Nx)
    Cy = allocate_observation_gramian(Ny,algo)  #shape (Ny,Ny)
    rxhist = []
    ryhist = []
    Cx_history = []
    Cy_history = []
    
    # Run the EnKF
    for i in tqdm(Dcycle, desc="LREnKF Progress"):
        # Forecast step
        if i==0 or (i % step == 0):
            dX_lat = forecast(X, training=False).numpy()    # X.shape: (Ne,Nx+len(AoA))
            X_lat = X[:, :x_lat_dim] + dt_dyn * dX_lat  # Update only the latent variables
            X_lat = additive_inflation(X_lat, fdata.Sigma_x)
            Xf.append(X_lat.copy())
            ystar = ytrue[i+1]
            
            if inflate:
                X_lat = multiplicative_inflation(X_lat, beta)
            
            # Compute marginally the variance of the state ensemble
            std_dev = np.std(X_lat, axis=0)
            Sigma_x = np.diag(std_dev ** 2)
            
            # Observation update
            Y = np.zeros((algo.Ne, Ny))
            Y = observation(X_lat, training=False).numpy() #shape (Ne,Ny)
            Sigma_eps = np.diag(np.ones(Ny)*odata.sigma_eps**2)  #shape (Ny,Ny)
            eps = create_ensemble(algo.Ne,np.zeros(Ny),Sigma_eps)   #shape (Ne,Ny)
            X_lat, Cx_history, Cy_history, rxhist, ryhist =  lrenkf_kalman_update(algo,X_lat,Y,ystar,Sigma_x,Sigma_eps,fdata,odata,Cx_history,Cy_history,rxhist,ryhist,Cx,Cy,Jac,eps)
            
            Xa.append(X_lat.copy())
            X = np.concatenate([X_lat, X[:, x_lat_dim:]], axis=-1)
        else:
            dX_lat = forecast(X, training=False).numpy()    # X.shape: (Ne,Nx+len(AoA))
            X_lat = X[:, :x_lat_dim] + dt_dyn * dX_lat  # Update only the latent variables
            X_lat = additive_inflation(X_lat, fdata.Sigma_x)
            Xf.append(X_lat.copy())
            X = np.concatenate([X_lat, X[:, x_lat_dim:]], axis=-1)
    return Xf, Xa, Cx_history, Cy_history, rxhist, ryhist
