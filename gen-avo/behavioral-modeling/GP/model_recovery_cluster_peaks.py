#model recovery for 4 GP models
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


###~~~~~~~~~~~~~~~~~~~~~~~~~Prep~~~~~~~~~~~~~~~~~~~~~~~~~###

# Define Grid
GRID_SIZE = 11
cs = 60 #cs position idx
val = 70 #cs value
row, col = divmod(cs, 11) #position coordinates
cs_position = [row,col]
N_SUBS = 1
N_BLOCKS = 1
N_TRIALS = 1 #in this case cs+ shown once 
#y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))#3D

#get euc distances to sample at different distances from CS+
distance_euc = np.zeros((GRID_SIZE, GRID_SIZE))
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        distance_euc[i, j] = np.sqrt((i - cs_position[0]) ** 2 + (j - cs_position[1]) ** 2)


n_samp = 2000
n_warm = 1000
n_chain = 3

###~~~~~~~~~~~~~~~~~~~~~~~~~Simulate data~~~~~~~~~~~~~~~~~~~~~~~~~###

#peaks = np.arange(50, 101, 10)
peaks = [50,60,70,80,90]
for peak in peaks: # loop over peaks? 
    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid
    distances = jnp.sqrt(square_dist(X, 1)) #precompute distances between values in X
    y[0,0,0] = peak #cs value
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))
    mean = 0

    #range ls parameter
    lss = np.linspace(0.01, 6, 20) #100
    grids = np.zeros((len(lss)*4, 121))

    #SIMUALTE EUCLIDEAN 
    for i in range(len(lss)):
        ls = lss[i]
        cell = np.array(cs).reshape(1, 1)
        cell.astype(np.int32)
        
        #get mu and var
        mu, var = gp_predict(cell, y_flat, distances, ls, mean)
        mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
        mu_flat = mu.squeeze()
        grids[i, :] = mu_flat #collect all grids

    for i in range(len(lss)):
        ls = lss[i]
        cell = np.array(cs).reshape(1, 1)
        cell.astype(np.int32)
        
        #simulate some lin data at different ls (and save at i+len subs)
        mu, var = gp_predict_lin(cell, y_flat, distances, ls, mean)
        mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
        mu_flat = mu.squeeze()
        grids[i+len(lss), :] = mu_flat #collect all grids


    # SIMULATE MANHATTAN
    distances = jnp.sqrt(manhattan_dist(X, 1))

    for i in range(len(lss)):
        ls = lss[i]
        cell = np.array(cs).reshape(1, 1)
        cell.astype(np.int32)
        
        #simulate some rbf data at different ls
        mu, var = gp_predict(cell, y_flat, distances, ls, mean)
        mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
        mu_flat = mu.squeeze()
        grids[i+len(lss)*2, :] = mu_flat #collect all grids
    
    
    for i in range(len(lss)):
        ls = lss[i]
        cell = np.array(cs).reshape(1, 1)
        cell.astype(np.int32)
        
        #simulate some lin data at different ls (and save at i+len subs)
        mu, var = gp_predict_lin(cell, y_flat, distances, ls, mean)
        mu_grid = mu.reshape(GRID_SIZE, GRID_SIZE)
        mu_flat = mu.squeeze()
        grids[i+len(lss)*3, :] = mu_flat #collect all grids 
    

    #Sample from grids at target distances 
    N_SUBS = 1
    N_BLOCKS = 1

    model_outputs = {}
    model_outputs['euc_rbf'] = {}
    model_outputs['euc_lin'] = {}
    model_outputs['man_rbf'] = {}
    model_outputs['man_lin'] = {}


    #sample at distances... 
    target_distances = [1,2,3,5]
    sampled_coords = sample_coordinates_by_distance(distance_euc, target_distances)
    #print("Sampled coordinates:", sampled_coords)

    # Extract sampled values from the grid based on sampled coordinates
    df = extract_sampled_values(grids, sampled_coords)
    df["rating"] = np.clip(df["rating"], 0, 100)
    df["rating"] = np.round(df["rating"]).astype(int)

    ###~~~~~~~~~~~~~~~~~~~~~~~~~Fit to data~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("peak")
    print(peak)
    # Loop over ls (here ID!)
    data = df
    unique_ids = data["ID"].unique()
    #print(unique_ids)
    for id_val in unique_ids:
        results_df = []
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
        csval[0, 0, :] = val

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

        #EUCLIDEAN models 
        distances = jnp.sqrt(square_dist(X, 1))

        #fit model to data: RBF
        mod = 'euc_rbf_peak'
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
        model_output.print_summary()

        #save all relevant bits
        waic_data = az.from_numpyro(model_output)
        waic_result = az.waic(waic_data)
        loo_result = az.loo(waic_data)
        print("elpd_WAIC:", waic_result.elpd_waic, "elpd_LOO:", loo_result.elpd_loo)

        sim_md = 'euc_rbf_peak' if id_val < len(lss) else 'euc_lin_peak' if len(lss) <= id_val < len(lss)*2 else 'man_rbf_peak' if len(lss)*2 <= id_val < len(lss)*3 else 'man_lin_peak' if len(lss)*3 <= id_val < len(lss)*4 else 'NA'
        print(sim_md)

        results_df.append({
            'distances': target_distances,
            'ID': id_val,
            'EWAIC': waic_result.elpd_waic,
            'LOO': loo_result.elpd_loo,
            'SIM': sim_md,
            'FIT': mod,
            'LS': id_val,
            'PEAK': peak,
            #'model_output': model_output,
            #'posterior_samples': posterior_samples
            
        })
        
        #fit model to data: LIN
        mod = 'euc_lin_peak'
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
        model_output.print_summary()

        #WAIC - assess model fit
        waic_data = az.from_numpyro(model_output)
        waic_result = az.waic(waic_data)
        loo_result = az.loo(waic_data)
        print("elpd_WAIC:", waic_result.elpd_waic, "elpd_LOO:",loo_result.elpd_loo)

        sim_md = 'euc_rbf_peak' if id_val < len(lss) else 'euc_lin_peak' if len(lss) <= id_val < len(lss)*2 else 'man_rbf_peak' if len(lss)*2 <= id_val < len(lss)*3 else 'man_lin_peak' if len(lss)*3 <= id_val < len(lss)*4 else 'NA'
        results_df.append({
            'distances': target_distances,
            'ID': id_val,
            'EWAIC': waic_result.elpd_waic,
            'LOO': loo_result.elpd_loo,
            'SIM': sim_md,
            'FIT': mod,
            'LS': id_val,
            'PEAK': peak,
            #'model_output': model_output,
            #'posterior_samples': posterior_samples
        })



        #manh models 
        distances = jnp.sqrt(manhattan_dist(X, 1))  

        #fit model to data: RBF
        mod = 'man_rbf_peak'
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
        model_output.print_summary()

        #save all relevant bits
        waic_data = az.from_numpyro(model_output)
        waic_result = az.waic(waic_data)
        loo_result = az.loo(waic_data)
        print("elpd_WAIC:", waic_result.elpd_waic, "elpd_LOO:", loo_result.elpd_loo)
        sim_md = 'euc_rbf_peak' if id_val < len(lss) else 'euc_lin_peak' if len(lss) <= id_val < len(lss)*2 else 'man_rbf_peak' if len(lss)*2 <= id_val < len(lss)*3 else 'man_lin_peak' if len(lss)*3 <= id_val < len(lss)*4 else 'NA'
        
        results_df.append({
            'distances': target_distances,
            'ID': id_val,
            'EWAIC': waic_result.elpd_waic,
            'LOO': loo_result.elpd_loo,
            'SIM': sim_md,
            'FIT': mod,
            'LS': id_val,
            'PEAK': peak
            #'model_output': model_output,
            #'posterior_samples': posterior_samples
            
        })
        
        #fit model to data: LIN
        mod = 'man_lin_peak'
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
        model_output.print_summary()

        #WAIC - assess model fit
        waic_data = az.from_numpyro(model_output)
        waic_result = az.waic(waic_data)
        loo_result = az.loo(waic_data)
        print("elpd_WAIC:", waic_result.elpd_waic, "elpd_LOO:",loo_result.elpd_loo)
        sim_md = 'euc_rbf_peak' if id_val < len(lss) else 'euc_lin_peak' if len(lss) <= id_val < len(lss)*2 else 'man_rbf_peak' if len(lss)*2 <= id_val < len(lss)*3 else 'man_lin_peak' if len(lss)*3 <= id_val < len(lss)*4 else 'NA'
        
        results_df.append({
            'distances': target_distances,
            'ID': id_val,
            'EWAIC': waic_result.elpd_waic,
            'LOO': loo_result.elpd_loo,
            #'SIM':'euc_rbf' if id_val < 10 else 'euc_lin' if 10 <= id_val < 20 else 'man_rbf' if 20 <= id_val < 30 else 'man_lin' if 30 <= id_val < 40 else 'NA',
            'SIM': sim_md,
            'FIT': mod,
            'LS': id_val,
            'PEAK': peak
            #'model_output': model_output,
            #'posterior_samples': posterior_samples
        
        })

        results_df = pd.DataFrame(results_df)
        #save
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
        directory = os.path.join(parent_dir, "fitted/ls_modelrecovery_peak")
        output_path = os.path.join(directory, f"modelrecovery_{id_val}_{int(float(peak))}_{sim_md}.csv")
        results_df.to_csv(output_path, index=False)

