#ls recovery for 4 GP models
#distance(euc, manh) x kernel (gaussian, linear)

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.stats import multivariate_normal
import numpyro.distributions as dist
import itertools
from scipy.stats import pearsonr
import ast 
import arviz as az
import os
import json
from pathlib import Path
from gp_functions import *
from numpyro.diagnostics import summary

n_samp = 2000
n_warm = 1000
n_chain = 3


peaks = [60,70,80,90]
for peak in peaks:
    ###~~~~~~~~~~~~~~~~~~~~~~~~~Euclidean RBF ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("euc. rbf")

    # Define Grid
    GRID_SIZE = 11
    cs = 60 #cs position idx
    #val = 70 #cs value
    row, col = divmod(cs, 11) #position coordinates
    cs_position = [row,col]
    N_SUBS = 1
    N_BLOCKS = 1
    N_TRIALS = 1 #in this case cs+ shown once 
    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))#3D

    #get euc distances to sample at different distances from CS+
    distance_euc = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            distance_euc[i, j] = np.sqrt((i - cs_position[0]) ** 2 + (j - cs_position[1]) ** 2)

    #sample at distances... 
    opts = [1,2,3,5]

    #simulate data
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid
    distances = jnp.sqrt(square_dist(X, 1)) #precompute distances between values in X
    y[0,0,0] = peak #cs value
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))
    mean = 0

    #range ls parameter
    lss = np.arange(0.01, 6, 0.05) 
    grids = np.zeros((len(lss), 121))
    for cs in [cs]: # example locations on grid
        for i in range(len(lss)):
            ls = lss[i]
            cell = np.array(cs).reshape(1, 1)
            cell.astype(np.int32)
            
            #get mu and var
            mu, var = gp_predict(cell, y_flat, distances, ls, mean)
            mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
            mu_flat = mu.squeeze()
            grids[i, :] = mu_flat #collect all grids
        

    target_n = len(opts)
    model_outputsRBF = {}
    target_distances = opts
    sampled_coords = sample_coordinates_by_distance(distance_euc, target_distances)
    df = extract_sampled_values(grids, sampled_coords)
    data = df
    unique_ids = data["ID"].unique()


    model_outputsRBF = {}
    for id_val in unique_ids:
        print(id_val)
        # Subset data for current ID
        subset = data[data["ID"] == id_val]
        N_TRIALS = len(subset["rating"])

        # Initialize arrays for shown and rated tiles
        prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
        csval = np.zeros((N_SUBS, N_BLOCKS, 1))

        # Fill with data
        prompt[0, 0, :] = subset["promptIDX"].values
        rating[0, 0, :] = subset["rating"].values
        csloc[0, 0, :] = cs
        csval[0, 0, :] = peak

        # Reshape data
        prompt_flat = np.reshape(prompt, (N_SUBS * N_BLOCKS, N_TRIALS))
        rating_flat = np.reshape(rating, (N_SUBS * N_BLOCKS, N_TRIALS))
        csloc_flat = np.reshape(csloc, (N_SUBS * N_BLOCKS, 1))
        csval_flat = np.reshape(csval, (N_SUBS * N_BLOCKS, 1))

        # Convert to JAX arrays
        prompt_flat = jax.device_put(prompt_flat.astype(int))
        rating_flat = jax.device_put(rating_flat)
        csloc_flat = jax.device_put(csloc_flat.astype(int))
        csval_flat = jax.device_put(csval_flat.astype(int))

        # Run model and fit across trials
        model_output = sampleLUI(
            grid_model_SIM_ALT,
            prompt_flat,
            rating_flat,
            csloc_flat,
            csval_flat,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samp,
            n_warmup=n_warm,
            num_chains=n_chain,
        )

        posterior_samples = model_output.get_samples()
        az_data = az.from_numpyro(model_output)
        model_output.print_summary()

        # Save
        model_outputsRBF[id_val] = {
            "model_output": model_output,
            "posterior_samples": posterior_samples,
            "az_data": az_data
        }

    #make pretty to save 
    all_rbf = []  
    for id_val in unique_ids:
        ls = lss[id_val]
        pid = model_outputsRBF[id_val]["posterior_samples"]
        pidplot = pid['ls_subject_transformed'].flatten()
        pidplot2 = pid['peak_subject_transformed'].flatten() 

        # Compute the mean of the posterior samples for this combination of ID and target_n
        mean_val = jnp.mean(pidplot)
        mean_val2 = jnp.mean(pidplot2)
        all_rbf.append([mean_val, ls, mean_val2, peak])


    all_means_np = np.array(all_rbf)
    df_euc_rbf = pd.DataFrame(all_means_np, columns=['recovered', 'ls', 'recovered_peak', 'peak'])
    df_euc_rbf['model'] = "euc_rbf"

    #save
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    directory = os.path.join(parent_dir, "/fitted/ls_parameterrecovery_peak")
    output_path = os.path.join(directory, f"ls_recovery_eucgauss_{peak}.csv")
    df_euc_rbf.to_csv(output_path, index=False)


    ###~~~~~~~~~~~~~~~~~~~~~~~~~Euclidean Linear ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("euc. lin")

    GRID_SIZE = 11
    cs = 60 #cs position idx
    #val = 70 #cs value
    row, col = divmod(cs, 11) #position coordinates
    cs_position = [row,col]

    N_SUBS = 1
    N_BLOCKS = 1
    N_TRIALS = 1 #in this case cs+ shown once 
    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))#3D
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances
    distances = jnp.sqrt(square_dist(X, 1)) #precompute distances between values in X
    y[0,0,0] = peak #cs value
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))
    mean = 0
    lss = np.arange(0.01, 6, 0.05) 
    grids = np.zeros((len(lss), 121))

    for cs in [cs]:
        for i in range(len(lss)):
            ls = lss[i]
            cell = np.array(cs).reshape(1, 1)
            cell.astype(np.int32)
            
            #get mu and var
            mu, var = gp_predict_lin(cell, y_flat, distances, ls, mean)
            mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
            mu_flat = mu.squeeze()
            grids[i, :] = mu_flat #collect all grids
        

    model_outputsLIN = {}
    target_distances = opts
    sampled_coords = sample_coordinates_by_distance(distance_euc, target_distances)
    df = extract_sampled_values(grids, sampled_coords)
    data = df
    unique_ids = data["ID"].unique()


    model_outputsLIN = {}
    for id_val in unique_ids:
        print(id_val)    
        # Subset data for current ID
        subset = data[data["ID"] == id_val]
        N_TRIALS = len(subset["rating"])

        # Initialize arrays for shown and rated tiles
        prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
        csval = np.zeros((N_SUBS, N_BLOCKS, 1))

        # Fill with data
        prompt[0, 0, :] = subset["promptIDX"].values
        rating[0, 0, :] = subset["rating"].values
        csloc[0, 0, :] = cs
        csval[0, 0, :] = peak

        # Reshape data
        prompt_flat = np.reshape(prompt, (N_SUBS * N_BLOCKS, N_TRIALS))
        rating_flat = np.reshape(rating, (N_SUBS * N_BLOCKS, N_TRIALS))
        csloc_flat = np.reshape(csloc, (N_SUBS * N_BLOCKS, 1))
        csval_flat = np.reshape(csval, (N_SUBS * N_BLOCKS, 1))

        # Convert to JAX arrays
        prompt_flat = jax.device_put(prompt_flat.astype(int))
        rating_flat = jax.device_put(rating_flat)
        csloc_flat = jax.device_put(csloc_flat.astype(int))
        csval_flat = jax.device_put(csval_flat.astype(int))

        # Run model and fit across trials
        model_output = sampleLUI(
            grid_model_LIN_ALT,
            prompt_flat,
            rating_flat,
            csloc_flat,
            csval_flat,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samp,
            n_warmup=n_warm,
            num_chains=n_chain,
        )    
            
        posterior_samples = model_output.get_samples()
        model_outputsLIN[id_val] = {
            "model_output": model_output,
            "posterior_samples": posterior_samples
        }
        

    all_lin = []
    for id_val in unique_ids:
        ls = lss[id_val]

        pid = model_outputsLIN[id_val]["posterior_samples"]
        pidplot = pid['ls_subject_transformed'].flatten()
        pidplot2 = pid['peak_subject_transformed'].flatten()

        mean_val = jnp.mean(pidplot)
        mean_val2 = jnp.mean(pidplot2)
        all_lin.append([mean_val, ls, mean_val2, peak])

    all_means_np = np.array(all_lin)
    df_euc_lin = pd.DataFrame(all_means_np, columns=['recovered', 'ls', 'recovered_peak', 'peak'])
    df_euc_lin['model'] = "euc_lin"

    #save
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    directory = os.path.join(parent_dir, "fitted/ls_parameterrecovery_peak")
    output_path = os.path.join(directory, f"ls_recovery_euclin_{peak}.csv")
    df_euc_lin.to_csv(output_path, index=False)


    ###~~~~~~~~~~~~~~~~~~~~~~~~~Manhattan RBF ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("manh. rbf")

    GRID_SIZE = 11
    cs = 60 #cs position idx
    #val = 70 #cs value
    row, col = divmod(cs, 11) #position coordinates
    cs_position = [row,col]
    N_SUBS = 1
    N_BLOCKS = 1
    N_TRIALS = 1

    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    #Manhattan distance; precompute distances between values in X
    distances = jnp.sqrt(manhattan_dist(X, 1))  
    y[0,0,0] = peak #cs value
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))
    mean = 0
    lss = np.arange(0.01, 6, 0.05) 
    grids = np.zeros((len(lss), 121))

    for cs in [cs]:
        for i in range(len(lss)):
            ls = lss[i]
            cell = np.array(cs).reshape(1, 1)
            cell.astype(np.int32)
            
            #get mu and var
            mu, var = gp_predict(cell, y_flat, distances, ls, mean)
            mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
            mu_flat = mu.squeeze()
            grids[i, :] = mu_flat
        

    distances = jnp.sqrt(manhattan_dist(X, 1)) 
    model_outputsMHRBF = {}
    target_distances = opts
    sampled_coords = sample_coordinates_by_distance(distance_euc, target_distances)
    df = extract_sampled_values(grids, sampled_coords)
    data = df
    unique_ids = data["ID"].unique()


    model_outputsMHRBF = {}
    for id_val in unique_ids:
        print(id_val)
        # Subset data for current ID
        subset = data[data["ID"] == id_val]
        N_TRIALS = len(subset["rating"])

        # Initialize arrays for shown and rated tiles
        prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
        csval = np.zeros((N_SUBS, N_BLOCKS, 1))

        # Fill with data
        prompt[0, 0, :] = subset["promptIDX"].values
        rating[0, 0, :] = subset["rating"].values
        csloc[0, 0, :] = cs
        csval[0, 0, :] = peak

        # Reshape data
        prompt_flat = np.reshape(prompt, (N_SUBS * N_BLOCKS, N_TRIALS))
        rating_flat = np.reshape(rating, (N_SUBS * N_BLOCKS, N_TRIALS))
        csloc_flat = np.reshape(csloc, (N_SUBS * N_BLOCKS, 1))
        csval_flat = np.reshape(csval, (N_SUBS * N_BLOCKS, 1))

        # Convert to JAX arrays
        prompt_flat = jax.device_put(prompt_flat.astype(int))
        rating_flat = jax.device_put(rating_flat)
        csloc_flat = jax.device_put(csloc_flat.astype(int))
        csval_flat = jax.device_put(csval_flat.astype(int))

        # Run model and fit across trials
        model_output = sampleLUI(
            grid_model_SIM_ALT,
            prompt_flat,
            rating_flat,
            csloc_flat,
            csval_flat,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samp,
            n_warmup=n_warm,
            num_chains=n_chain,
        )
            
        posterior_samples = model_output.get_samples()
        model_outputsMHRBF[id_val] = {
            "model_output": model_output,
            "posterior_samples": posterior_samples
        }
        
    all_mhrbf = []
    for id_val in unique_ids:
        ls = lss[id_val]
        pid = model_outputsMHRBF[id_val]["posterior_samples"]
        pidplot = pid['ls_subject_transformed'].flatten()
        pidplot2 = pid['peak_subject_transformed'].flatten()
        mean_val = jnp.mean(pidplot)
        mean_val2 = jnp.mean(pidplot2)
        all_mhrbf.append([mean_val, ls, mean_val2, peak])

    all_means_np = np.array(all_mhrbf)
    df_manh_rbf = pd.DataFrame(all_means_np, columns=['recovered', 'ls', 'recovered_peak', 'peak'])
    df_manh_rbf['model'] = "manh_rbf"

    #save
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    directory = os.path.join(parent_dir, "fitted/ls_parameterrecovery_peak")
    output_path = os.path.join(directory, f"ls_recovery_manhrbf_{peak}.csv")
    df_manh_rbf.to_csv(output_path, index=False)


    ###~~~~~~~~~~~~~~~~~~~~~~~~~Manhattan Linear ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("manh. lin")

    GRID_SIZE = 11
    cs = 60 #cs position idx
    #val = 70 #cs value
    row, col = divmod(cs, 11) #position coordinates
    cs_position = [row,col]

    N_SUBS = 1
    N_BLOCKS = 1
    N_TRIALS = 1 #in this case cs+ shown once 
    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))#3D
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances (Manhattan)
    distances = jnp.sqrt(manhattan_dist(X, 1))
    y[0,0,0] = peak #cs value
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))
    mean = 0
    lss = np.arange(0.01, 6, 0.05) 
    grids = np.zeros((len(lss), 121))

    for cs in [cs]:
        for i in range(len(lss)):
            ls = lss[i]
            cell = np.array(cs).reshape(1, 1)
            cell.astype(np.int32)
            
            #get mu and var
            mu, var = gp_predict_lin(cell, y_flat, distances, ls, mean)
            mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
            mu_flat = mu.squeeze()
            grids[i, :] = mu_flat

    distances = jnp.sqrt(manhattan_dist(X, 1)) 
    model_outputsMHLIN = {}
    target_distances = opts
    sampled_coords = sample_coordinates_by_distance(distance_euc, target_distances)
    df = extract_sampled_values(grids, sampled_coords)
    data = df
    unique_ids = data["ID"].unique()


    model_outputsMHLIN = {}
    for id_val in unique_ids:
        print(id_val)
        # Subset data for current ID
        subset = data[data["ID"] == id_val]
        N_TRIALS = len(subset["rating"])

        # Initialize arrays for shown and rated tiles
        prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
        csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
        csval = np.zeros((N_SUBS, N_BLOCKS, 1))

        # Fill with data
        prompt[0, 0, :] = subset["promptIDX"].values
        rating[0, 0, :] = subset["rating"].values
        csloc[0, 0, :] = cs
        csval[0, 0, :] = peak

        # Reshape data
        prompt_flat = np.reshape(prompt, (N_SUBS * N_BLOCKS, N_TRIALS))
        rating_flat = np.reshape(rating, (N_SUBS * N_BLOCKS, N_TRIALS))
        csloc_flat = np.reshape(csloc, (N_SUBS * N_BLOCKS, 1))
        csval_flat = np.reshape(csval, (N_SUBS * N_BLOCKS, 1))

        # Convert to JAX arrays
        prompt_flat = jax.device_put(prompt_flat.astype(int))
        rating_flat = jax.device_put(rating_flat)
        csloc_flat = jax.device_put(csloc_flat.astype(int))
        csval_flat = jax.device_put(csval_flat.astype(int))

        # Run model and fit across trials
        model_output = sampleLUI(
            grid_model_LIN_ALT,
            prompt_flat,
            rating_flat,
            csloc_flat,
            csval_flat,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samp,
            n_warmup=n_warm,
            num_chains=n_chain,
        )
            
        posterior_samples = model_output.get_samples()
        model_outputsMHLIN[id_val] = {
            "model_output": model_output,
            "posterior_samples": posterior_samples
        }


    all_mhlin = []
    for id_val in unique_ids:
        ls = lss[id_val]
        pid = model_outputsMHLIN[id_val]["posterior_samples"]
        pidplot = pid['ls_subject_transformed'].flatten()
        pidplot2 = pid['peak_subject_transformed'].flatten()
        mean_val = jnp.mean(pidplot)
        mean_val2 = jnp.mean(pidplot2)
        all_mhlin.append([mean_val, ls, mean_val2, peak])

    all_means_np = np.array(all_mhlin)
    df_manh_lin = pd.DataFrame(all_means_np, columns=['recovered', 'ls', 'recovered_peak', 'peak'])
    df_manh_lin['model'] = "manh_lin"

    #save
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    directory = os.path.join(parent_dir, "fitted/ls_parameterrecovery_peak")
    output_path = os.path.join(directory, f"ls_recovery_manhlin_{peak}.csv")
    df_manh_lin.to_csv(output_path, index=False)


