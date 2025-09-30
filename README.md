# Latent_DA_Aero
![Demo of Latent_DA_Aero](vorticity_animation.gif)

A machine-learning tool to sequentially estimate the strongly disturbed aerodynamic states using sparse noisy pressure measurements. This code levereges data assimilation in a reduced state space to make estimation fast and efficient for real-time applications. Although the attached code is tailored to our paper [Sequential estimation of disturbed aerodynamic flows from sparse measurements via a reduced latent space](https://arxiv.org/abs/2509.03795)---where vorticity snapshots and lift coefficients serve as the flow states and pressure measurements as the observations, with an autoencoder used for dimensionality reduction---it can be readily applied to data assimilation in any reduced space, provided that a mapping from the reduced space to the original physical space is available. Only minor modifications are required in such cases.

For filtering problems, we specifically need a **forecast operator** that evolves the underlying flow states as well as an **observation operator** to map states to the corresponding observations. In a reduced space, these two models need to be learned. 

## Steps to execute the workflow
Follow the steps below in the specified order to execute the workflow:
1. Extract/learn a reduced state space (here by training a physics-augmented autoencoder):
Run the file `autoencoder.py` to train an autoencoder that extracts the underlying latent variables from data. These latent variables represent the reduced-order features of the vorticity field and lift coefficient while containing information about pressure observations as well. The observation operator is simultaneouly learned within the autoencoder.
2. Learn a dynamical model in the reduced latent space:
Execute `dynamics.py` to train a Multi-Layer Perceptron (MLP) network. This network maps sparse, clean surface pressure measurements to the extracted latent variables deterministically. The trained model will later be used for Gramian calculation.
3. Calculate Dominant Directions:
Run `noise_in_dominant_direction.py` to identify and store the dominant directions of both the measurement and latent variable spaces at each time step. Perturb the measurements along the dominant eigenvector of the measurement space Gramian, C_x.
4. Train the Bayesian Neural Network Using MC Dropout:
Use `probabilisticPressureNetwork.py` to train a Bayesian Neural Network. This model estimates the statistics of the latent variables by predicting the mean and covariance matrix of a multivariate normal distribution in the latent space. The training process minimizes the negative log-likelihood.
5. Flow Reconstruction and Uncertainty Quantification:
Finally, execute `flowReconstructionAndUQ.py` to:
- Estimate the latent variables while quantifying aleatoric (data-driven) and epistemic (model-driven) uncertainties.
- Map the estimated latent space samples back to the original high-dimensional space to reconstruct the vorticity field and lift coefficient.