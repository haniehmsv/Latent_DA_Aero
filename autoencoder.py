import tensorflow as tf
from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose, Dropout, SpectralNormalization
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
from scipy.io import loadmat
import pickle
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

import os
os.chdir('/u/project/sofia/hanieh/EnKFthroughLearnedOperators/gaussian_force/autoencoder/autoencoder_temporal_loss/autoencoder_temporal_loss_7_long_time')
os.getcwd()

def load_vorticity_pressure_lift(AoA:np.ndarray, ranget, rangep, cases:np.ndarray):
    """
    Load vorticity, lift coefficient, and surface pressure data for specified angles of attack (AoA)
    and gust disturbance cases.

    This function reads base flow cases and gust-disturbed cases from `.jld` files, extracts 
    the requested time and pressure ranges, and concatenates them into unified arrays for 
    downstream training or analysis.

    Args:
        AoA (np.ndarray): Array of angles of attack to load (e.g., [20, 30, 40, 50, 60]).
        ranget (slice or array-like): Range of time indices to extract from each case.
        rangep (slice or array-like): Range of pressure sensor indices to extract.
        cases (np.ndarray): Array of gust disturbance identifiers.

    Returns:
        y_1 (np.ndarray): Concatenated vorticity fields of shape (N, H, W, 1), 
                          where N is the total number of selected snapshots across AoA and aerodynamic cases.
        y_CL (np.ndarray): Concatenated lift coefficient histories of shape (N, 1).
        y_pres (np.ndarray): Concatenated surface pressure measurements of shape (N, n_sensors),
                             where n_sensors = len(rangep).

    Notes:
        - Base cases are always included for each AoA.
        - Gust cases are additionally loaded according to the provided `cases` array.
        - Vorticity fields are transposed to (time, height, width).
        - Lift histories are transposed to (time, 1).
        - Pressure is indexed directly with (time, sensors).
    """
    
    # base cases
    y_1_list = []
    y_CL_list = []
    y_pres_list = []
    for num in AoA:
        vordir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/data/vort_a{num}/AoA{num}_base.jld'
        output = h5py.File(vordir, "r")
        output = output['omg_box']
        output = output[:]
        y_1_list.append((output.transpose(2, 1, 0))[ranget])
    
        liftdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/lift/a{num}/Lift_AoA{num}_base.jld'
        output = h5py.File(liftdir, "r")
        output = output['lift_box']
        output = output[:]
        y_CL_list.append((output.transpose(1, 0))[ranget])
        
        presdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/pressure/a{num}/Pressure_AoA{num}_base.jld'
        output = h5py.File(presdir, "r")
        output = output['pres_box']
        output = output[:]
        y_pres_list.append(output[ranget,rangep])
        
    # Gust cases
    for num in AoA:
        for j in cases:
            vordir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/data/vort_a{num}/AoA{num}_RD{j}.jld'
            output = h5py.File(vordir, "r")
            output = output['omg_box']
            output = output[:]
            y_1_list.append((output.transpose(2, 1, 0))[ranget])
        
            liftdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/lift/a{num}/Lift_AoA{num}_RD{j}.jld'
            output = h5py.File(liftdir, "r")
            output = output['lift_box']
            output = output[:]
            y_CL_list.append((output.transpose(1, 0))[ranget])
        
            presdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/pressure/a{num}/Pressure_AoA{num}_RD{j}.jld'
            output = h5py.File(presdir, "r")
            output = output['pres_box']
            output = output[:]
            y_pres_list.append(output[ranget,rangep])
    
    y_1 = np.concatenate(y_1_list, axis=0)
    y_CL = np.concatenate(y_CL_list, axis=0)
    y_pres = np.concatenate(y_pres_list, axis=0)
    y_1 = np.expand_dims(y_1, axis=-1)
    return y_1, y_CL, y_pres

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

def construct_grid(xrange, yrange, nx:int, ny:int):
    """
    Construct a 2D Cartesian grid from given ranges and resolution.

    Args:
        xrange (tuple): (xmin, xmax), the range of x-coordinates.
        yrange (tuple): (ymin, ymax), the range of y-coordinates.
        nx (int): Number of grid points along the x-axis.
        ny (int): Number of grid points along the y-axis.

    Returns:
        (X, Y) (tuple of np.ndarray): Meshgrid arrays of shape (ny, nx) representing 
                                      the 2D grid coordinates.
    """
    xmin, xmax = xrange
    ymin, ymax = yrange
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

def load_sensor_positions(AoA:np.ndarray, rangep, n:int, cases:np.ndarray):
    """
    Load and replicate sensor coordinates (x, y) for base and gust cases.

    This function reads sensor positions from `.jld2` files corresponding to 
    base flow and gust-disturbed cases at specified angles of attack (AoA). 
    The selected sensor indices are repeated across time snapshots to align 
    with other time-dependent data (e.g., pressure or vorticity).

    Args:
        AoA (np.ndarray): Array of angles of attack to process (e.g., [20, 30, 40, 50, 60]).
        rangep (slice or array-like): Indices of sensors to include (subset of all available sensors).
        n (int): Number of time snapshots per case; used to tile sensor positions along time.
        cases (np.ndarray): Array of gust disturbance identifiers.

    Returns:
        xsens (np.ndarray): Concatenated x-coordinates of selected sensors, 
                            shape (N_cases * n, n_sensors).
        ysens (np.ndarray): Concatenated y-coordinates of selected sensors, 
                            shape (N_cases * n, n_sensors).
    """
    
    xsens_list = []
    ysens_list = []
    
    # base cases
    for num in AoA:
        posdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA{num}/AoA{num}_base.jld2'
        output = h5py.File(posdir, "r")
        output = output['xsens']
        output = output[:][rangep]
        output = np.tile(output, (n, 1)) 
        xsens_list.append(output)
        
        output = h5py.File(posdir, "r")
        output = output['ysens']
        output = output[:][rangep]
        output = np.tile(output, (n, 1)) 
        ysens_list.append(output)
    
    # Gust cases
    for num in AoA:
        for j in cases:
            posdir = f'/u/project/sofia/hanieh/EnKFthroughLearnedOperators/data_generation_gaussian_forcing/AoA{num}/AoA{num}_RD{j}.jld2'
            output = h5py.File(posdir, "r")
            output = output['xsens']
            output = output[:][rangep]
            output = np.tile(output, (n, 1)) 
            xsens_list.append(output)
            
            output = h5py.File(posdir, "r")
            output = output['ysens']
            output = output[:][rangep]
            output = np.tile(output, (n, 1)) 
            ysens_list.append(output)
            
    xsens = np.concatenate(xsens_list, axis=0)
    ysens = np.concatenate(ysens_list, axis=0)
        
    return xsens, ysens

def split_train_test_dataset(y_1, y_CL, y_pres, test_size, n_cases, nsnap, random_state=42):
    """
    Split vorticity, lift, and pressure datasets into training and testing subsets.

    The split is performed at the case level so that all 
    snapshots from a given case belong entirely to either training or testing.

    Args:
        y_1 (np.ndarray): Vorticity field data of shape (n_cases, nsnap, H, W, 1).
        y_CL (np.ndarray): Lift coefficient histories of shape (n_cases, nsnap, 1).
        y_pres (np.ndarray): Pressure sensor measurements of shape (n_cases, nsnap, n_sensors).
        test_size (float): Proportion (0–1).
        n_cases (int): Total number of independent flow cases, including both undisturbed and gust-disturbed.
        nsnap (int): Number of time snapshots per case.
        random_state (int, default=42): Random seed for reproducibility.

    Returns:
        X_train (np.ndarray): Training vorticity data.
        X_test (np.ndarray): Testing vorticity data.
        X_train_CL (np.ndarray): Training lift coefficient data.
        X_test_CL (np.ndarray): Testing lift coefficient data.
        X_train_pres (np.ndarray): Training pressure data.
        X_test_pres (np.ndarray): Testing pressure data.
        idx_train (np.ndarray): Indices of training cases.
        idx_test (np.ndarray): Indices of testing cases.
    """
    idx_case = np.arange(n_cases)
    X_train, X_test, X_train_CL, X_test_CL, X_train_pres, X_test_pres, idx_train, idx_test = train_test_split(y_1, y_CL, y_pres, idx_case, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_train_CL, X_test_CL, X_train_pres, X_test_pres, idx_train, idx_test

def flatten_data(data):
    data = tf.reshape(data, (-1, *data.shape[2:]))
    return data

def temporal_smoothness_loss(latents):
    """
    Compute a second-order temporal smoothness loss for latent variables.

    This loss penalizes rapid changes in the latent trajectory by encouraging 
    approximate second-order continuity (i.e., discouraging sudden accelerations). 

    Args:
        latents (tf.Tensor): Sequence of latent variables of shape (B, n), 
                             where B is the number of time steps in a batch and n is 
                             the latent dimension.

    Returns:
        tf.Tensor: Scalar tensor representing the mean squared second-order 
                   difference across all time steps and latent dimensions.
    """
    z_prev = latents[:-2]
    z_curr = latents[1:-1]
    z_next = latents[2:]
    second_diffs = z_next - 2*z_curr + z_prev
    return tf.reduce_mean(tf.square(second_diffs))

AoA = np.array([20, 30, 40, 50, 60])
cases = np.arange(1,61) # 60 independent gust cases for each AoA

ranget = slice(1,501,1) # 500 nsnapshots for each case
rangep = slice(4,69,6)  # 11 pressure sensors
y_1, y_CL, y_pres = load_vorticity_pressure_lift(AoA, ranget, rangep, cases)
nsnap = 500 # 500 nsnapshots for each case

n_cases = len(AoA)*61   # 5 base cases + 5*60 gust cases
y_1_cases = y_1.reshape(n_cases, nsnap, y_1.shape[1], y_1.shape[2], y_1.shape[3])   # Shape (n_cases, nsnap, H, W, C=1)
y_CL_cases = y_CL.reshape(n_cases, nsnap, y_CL.shape[-1])   # Shape (n_cases, nsnap, C=1)
y_pres_cases = y_pres.reshape(n_cases, nsnap, y_pres.shape[-1])  # Shape (n_cases, nsnap, n_sensors)
X_train, X_test, X_train_CL, X_test_CL, X_train_pres, X_test_pres, idx_train, idx_test = split_train_test_dataset(y_1_cases, y_CL_cases, y_pres_cases, 0.3, n_cases, nsnap)
np.save('idx_train.npy', idx_train)
np.save('idx_test.npy', idx_test)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
X_train_CL = tf.convert_to_tensor(X_train_CL, dtype=tf.float32)
X_test_CL = tf.convert_to_tensor(X_test_CL, dtype=tf.float32)
X_train_pres = tf.convert_to_tensor(X_train_pres, dtype=tf.float32)
X_test_pres = tf.convert_to_tensor(X_test_pres, dtype=tf.float32)

num_chunks = 4
batch_size = nsnap // num_chunks

X_train = flatten_data(X_train) # Shape (N_train*nsnap, H, W, 1)
X_train_CL = flatten_data(X_train_CL)   # Shape (N_train*nsnap, 1)
X_train_pres = flatten_data(X_train_pres)   # Shape (N_train*nsnap, n_sensors)
X_test = flatten_data(X_test)   # Shape (N_test*nsnap, H, W, 1)
X_test_CL = flatten_data(X_test_CL)  # Shape (N_test*nsnap, 1)
X_test_pres = flatten_data(X_test_pres) # Shape (N_test*nsnap, n_sensors)


SEED = 42
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train_CL, X_train_pres))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True).shuffle(buffer_size=len(X_train), seed=SEED, reshuffle_each_iteration=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test_CL, X_test_pres))
test_dataset = test_dataset.batch(batch_size, drop_remainder=True).repeat().prefetch(tf.data.experimental.AUTOTUNE)

continue_state = True   # Set this to True to continue training from a saved model

if continue_state:
    # Load the previously saved model
    if os.path.exists('./model.keras'):
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model('./model.keras')
    else:
        print("Pre-trained model not found. Starting fresh training...")
        continue_state = False
else:
    ## Encoder
    act = 'tanh'
    input_img = Input(shape=(120,240,1))
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((2,2),padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((2,2),padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((5,5),padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Reshape([12*6*4])(x1)
    x1 = Dense(256,activation=act)(x1)
    x1 = Dense(128,activation=act)(x1)
    x_lat = Dense(7)(x1)

    ## pressure and lift augmentation
    x_pres = Dense(32,activation=act)(x_lat)
    x_pres = Dense(64,activation=act)(x_pres)
    x_pres = Dense(32,activation=act)(x_pres)
    x_pres_final = Dense(11)(x_pres)
    x_CL_final = Dense(1)(x_pres)

    ## Decoder
    x1 = Dense(128,activation=act)(x_lat)
    x1 = Dense(256,activation=act)(x1)
    x1 = Dense(288,activation=act)(x1)
    x1 = Reshape([6,12,4])(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((5,5))(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((2,2))(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((2,2))(x1)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x_final = Conv2D(1, (3,3),padding='same')(x1)
    
    model = Model(input_img, [x_final, x_CL_final, x_pres_final, x_lat])
    
optimizer = tf.keras.optimizers.Adam(amsgrad=True)

epochs = 5000
beta_latent = 5000.0     # temporal loss weight
beta_pres = 100.0         # pressure loss weight
beta_lift = 1.0      # lift loss weight
best_val_loss = float('inf')  # Track best validation loss
history_data = {'epoch': [], 'train_loss': [], 'val_loss': []}
model_path = './model.keras'  # Path to save and load the model
patience = 500
train_loss_log = []
val_loss_log = []
steps_per_epoch = len(X_train) // batch_size
val_steps_per_epoch = len(X_test) // batch_size

@tf.function
def train_step(y_batch, y_CL_true, y_pres_true):
    with tf.GradientTape() as tape:
        y_vort_pred, y_CL_pred, y_pres_pred, x_lat_pred = model(y_batch, training=True)
        loss_vort = tf.reduce_mean(tf.square(y_batch - y_vort_pred))
        loss_CL = tf.reduce_mean(tf.square(y_CL_true - y_CL_pred))
        loss_pres = tf.reduce_mean(tf.square(y_pres_true - y_pres_pred))
        loss_temp = temporal_smoothness_loss(x_lat_pred)
        total_loss = loss_vort + beta_lift * loss_CL + beta_pres * loss_pres + beta_latent * loss_temp 
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return total_loss, loss_vort, loss_CL, loss_pres, loss_temp

@tf.function
def val_step(y_batch, y_CL_true, y_pres_true):
    y_vort_pred, y_CL_pred, y_pres_pred, x_lat_pred = model(y_batch, training=False)
    val_loss_vort = tf.reduce_mean(tf.square(y_batch - y_vort_pred))
    val_loss_CL = tf.reduce_mean(tf.square(y_CL_true - y_CL_pred))
    val_loss_pres = tf.reduce_mean(tf.square(y_pres_true - y_pres_pred))
    val_loss_temporal = temporal_smoothness_loss(x_lat_pred)
    val_total_loss = val_loss_vort + beta_lift * val_loss_CL + beta_pres * val_loss_pres + beta_latent * val_loss_temporal
    return val_total_loss

# Training loop with early stopping
for epoch in range(epochs):
    train_loss = 0.0
    train_vort = 0.0
    train_CL = 0.0
    train_pres = 0.0
    train_temp = 0.0
    
    for step, batch in enumerate(train_dataset):
        if step >= steps_per_epoch:
            break
        y_batch, y_CL_true, y_pres_true = batch
        total_loss, loss_vort, loss_CL, loss_pres, loss_temp = train_step(y_batch, y_CL_true, y_pres_true)
        train_loss += total_loss
        train_vort += loss_vort
        train_CL += loss_CL
        train_pres += loss_pres
        train_temp += loss_temp
    
    train_loss /= (step+1)
    train_vort /= (step+1)
    train_CL /= (step+1)
    train_pres /= (step+1)
    train_temp /= (step+1)
    
    val_loss = 0.0
    for step, batch in enumerate(test_dataset):
        if step >= val_steps_per_epoch:
            break
        y_batch, y_CL_true, y_pres_true = batch
        val_total_loss = val_step(y_batch, y_CL_true, y_pres_true)
        val_loss += val_total_loss
    val_loss /= (step + 1)
    
    print(f"[Epoch {epoch+1:03d}] Loss: {train_loss:.4e} | Vort: {train_vort:.4e} | "
          f"CL: {beta_lift*train_CL:.4e} | Pres: {beta_pres*train_pres:.4e} | Temp: {beta_latent*train_temp:.4e} | Val Loss: {val_loss:.4e}")

    # Store loss history
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


# Encoding data
model = tf.keras.models.load_model("model.keras")
encoder = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
decoder_CL = Model(inputs=model.get_layer('dense_2').output, outputs=model.get_layer('dense_7').output)
decoder_pres = Model(inputs=model.get_layer('dense_2').output, outputs=model.get_layer('dense_6').output)
decoder = Model(inputs=model.get_layer('dense_2').output, outputs=model.get_layer('conv2d_16').output)

x_lat_data = encoder.predict(y_1)
np.save('x_lat.npy', x_lat_data)    # latent variables for all the data