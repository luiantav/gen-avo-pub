#get all things needed 
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
import argparse

#requires 
#jax==0.4.31
#arviz==0.20.0
#numpy==1.24.2
#pandas==2.23


#print(az.__version__)
numpyro.set_host_device_count(4) 
jax.local_device_count()


n_samp = 2000
n_warm = 1000
n_chain = 3
############### Prep saving ####################
    
def save_mcmc_json(samples, sub, run_val, directory="fitted/testPRE"):
    """Save multiple runs in a single JSON file per subject."""

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    directory = os.path.join(parent_dir, directory)
    filename = os.path.join(directory, f"PRE_{sub}.json")

    # NumPy arrays to lists for JSON compatibility
    samples_json = {key: value.tolist() for key, value in samples.items()}

    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[run_val] = {"posterior_samples": samples_json}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved MCMC samples for run {run_val} to: {filename}")



def save_diagnostics_df(diagnostics_df, id_val, run_val, model_name, directory):
    """Save the diagnostics DataFrame for a specific subject and run value."""

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../output"))
    directory = os.path.join(parent_dir, directory)

    if not os.path.exists(directory): # Create the directory if it doesn't exist
        os.makedirs(directory)

    filename = os.path.join(directory, f"fitted{id_val}_run_{run_val}_{model_name}.csv")
    diagnostics_df.to_csv(filename, index=True)
    print(f"Saved diagnostics for id {id_val} and run {run_val} to: {filename}")



############### Get data & Set paths ###########
here = Path(__file__).resolve().parent
file_path = here / "data-for-models/alldata.csv"
print(file_path)
dffull = pd.read_csv(file_path)

dffull['ID'] = dffull['ID'].str.replace('ID', '', regex=False)
print(len(dffull["ppt"].unique()))



################ PREP Cluster ###################
run_on_cluster = 1

if run_on_cluster == 1:
    # Parse Sub name 
    parser = argparse.ArgumentParser(description='Process subject ID.')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    args = parser.parse_args()
    subject = [args.subject]
else:
    subject = ["294353176824689","268113725819931"]


print("subject = ", subject)


################## Ratings PRE ####################
phase = ["ratingPRE", "ratingPOST"]

for ph in phase:
    df_phase = dffull[dffull["state"] == ph]

    # Split the 'ratingCoordinate' into 'row' and 'col'
    df_phase['ratingCoordinate'] = df_phase['ratingCoordinate'].apply(ast.literal_eval)
    df_phase[['row', 'col']] = pd.DataFrame(df_phase['ratingCoordinate'].tolist(), index=df_phase.index)

    # Calculate the corresponding index (rateIDX) on the 11x11 grid (0-indexed)
    GRID_SIZE = 11
    df_phase['promptIDX'] = df_phase['row'] * GRID_SIZE + df_phase['col']

    # Same for cs coordinate
    df_phase['csPosition'] = df_phase['csPosition'].apply(ast.literal_eval)
    df_phase[['rowCS', 'colCS']] = pd.DataFrame(df_phase['csPosition'].tolist(), index=df_phase.index)
    df_phase['csIDX'] = df_phase['rowCS'] * GRID_SIZE + df_phase['colCS']

    #rating counts 
    rating_counts = df_phase.groupby(['ID', 'run', 'state']).size().reset_index(name='RatingCount')

    print('Now fitting rating PRE ratings:')

###~~~~~~~~~~~~~~~~~~~~~~~~~Manhattan RBF (INCLUDING PEAK) ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("Now fitting peak manh rbf")
    # Make cartesian grid
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances: euc
    distances = jnp.sqrt(manhattan_dist(X, 1))

    N_SUBS = 1  # Fitting x subject
    N_BLOCKS = 1  # Single block
    model_outputsPRE = {}
    data = df_phase
    unique_runs = [0,1,2,3,4,5,6,7,8]

    for id_val in subject:
        print("running: ", id_val)
        for run_val in unique_runs:
            
            subset = data[(data["ID"] == id_val) & (data["run"] == run_val)]

            if subset.empty:
                print(f"Skipping ID {id_val}, run {run_val} (no data)")
                continue
            
            N_TRIALS = len(subset["rating"])

            cs = subset["csIDX"].unique() #cs position idx
            val = subset["csValue"].unique() #cs value

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

            # save diagnostics to df
            posterior_samples = model_output.get_samples()
            samples = model_output.get_samples(group_by_chain=True)
            diagnostics = summary(samples, prob=0.9)
            df = pd.DataFrame.from_dict(diagnostics, orient='index')
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

            df["run"] = run_val
            df["ID"] = id_val
            df["model"] = "manh_rbf_peak"
            df["ratings"] = ph
            
            az_data = az.from_numpyro(model_output)
            waic_res = az.waic(az_data)
            loo_res = az.loo(az_data)
            df["WAIC"] = waic_res.elpd_waic
            df["WAIC_P"] = waic_res.p_waic
            df["LOO"] = loo_res.elpd_loo
            df["LOO_P"] = loo_res.p_loo
            #print(df)
            save_diagnostics_df(df, id_val, run_val, "manh_rbf_peak", f"gp_fitted/{ph}")
            model_output.print_summary()


###~~~~~~~~~~~~~~~~~~~~~~~~~Euc Linear PEAK~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("Now fitting euc lin (incl peak)")
    # Make cartesian grid
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances (euc)
    distances = jnp.sqrt(square_dist(X, 1)) 

    N_SUBS = 1  # Fitting x subject
    N_BLOCKS = 1  # Single block
    model_outputsPRE = {}
    data = df_phase
    unique_runs = [0,1,2,3,4,5,6,7,8]

    #loop over runs 
    for id_val in subject:
        print("running: ", id_val)
        for run_val in unique_runs:
            
            # Subset data for current ID
            subset = data[(data["ID"] == id_val) & (data["run"] == run_val)]

            if subset.empty:
                print(f"Skipping ID {id_val}, run {run_val} (no data)")
                continue
            
            N_TRIALS = len(subset["rating"])

            cs = subset["csIDX"].unique() #cs position idx
            val = subset["csValue"].unique() #cs value

            # Initialize arrays for shown and rated tiles
            prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
            csval = np.zeros((N_SUBS, N_BLOCKS, 1))

            # Fill with data
            prompt[0, 0, :] = subset["promptIDX"].values
            rating[0, 0, :] = subset["rating"].values
            csloc[0, 0, :] = cs
            csval[0, 0, :] = val  # placeholder

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

            #save posterior samples
            posterior_samples = model_output.get_samples()

            # save diagnostics to separate df (basically a shorter version)
            samples = model_output.get_samples(group_by_chain=True)
            diagnostics = summary(samples, prob=0.9)
            df = pd.DataFrame.from_dict(diagnostics, orient='index')
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

            df["run"] = run_val
            df["ID"] = id_val
            df["model"] = "euc_lin_peak"
            df["ratings"] = ph
            
            az_data = az.from_numpyro(model_output)
            waic_res = az.waic(az_data)
            loo_res = az.loo(az_data)
            df["WAIC"] = waic_res.elpd_waic
            df["WAIC_P"] = waic_res.p_waic
            df["LOO"] = loo_res.elpd_loo
            df["LOO_P"] = loo_res.p_loo

            #print(df)
            save_diagnostics_df(df, id_val, run_val, "euc_lin_peak", f"gp_fitted/{ph}")
            model_output.print_summary()



    ###~~~~~~~~~~~~~~~~~~~~~~~~~Manhattan Linear PEAK~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("Now fitting manh lin (incl peak)")
    # Make cartesian grid
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances (euc or manh)
    distances = jnp.sqrt(manhattan_dist(X, 1))

    N_SUBS = 1  # Fitting x subject
    N_BLOCKS = 1  # Single block
    model_outputsPRE = {}
    data = df_phase
    unique_runs = [0,1,2,3,4,5,6,7,8]

    #loop over runs 
    for id_val in subject:
        print("running: ", id_val)
        for run_val in unique_runs:
            
            # Subset data for current ID
            subset = data[(data["ID"] == id_val) & (data["run"] == run_val)]

            if subset.empty:
                print(f"Skipping ID {id_val}, run {run_val} (no data)")
                continue
            
            N_TRIALS = len(subset["rating"])

            cs = subset["csIDX"].unique() #cs position idx
            val = subset["csValue"].unique() #cs value

            # Initialize arrays for shown and rated tiles
            prompt = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            rating = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            csloc = np.zeros((N_SUBS, N_BLOCKS, 1))
            csval = np.zeros((N_SUBS, N_BLOCKS, 1))

            # Fill with data
            prompt[0, 0, :] = subset["promptIDX"].values
            rating[0, 0, :] = subset["rating"].values
            csloc[0, 0, :] = cs
            csval[0, 0, :] = val  # placeholder

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

            #save posterior samples
            posterior_samples = model_output.get_samples()

            # save diagnostics to separate df (basically a shorter version)
            samples = model_output.get_samples(group_by_chain=True)
            diagnostics = summary(samples, prob=0.9)
            df = pd.DataFrame.from_dict(diagnostics, orient='index')
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

            df["run"] = run_val
            df["ID"] = id_val
            df["model"] = "manh_lin_peak"
            df["ratings"] = ph
            
            az_data = az.from_numpyro(model_output)
            waic_res = az.waic(az_data)
            loo_res = az.loo(az_data)
            df["WAIC"] = waic_res.elpd_waic
            df["WAIC_P"] = waic_res.p_waic
            df["LOO"] = loo_res.elpd_loo
            df["LOO_P"] = loo_res.p_loo

            #print(df)
            save_diagnostics_df(df, id_val, run_val, "manh_lin_peak", f"gp_fitted/{ph}")
            model_output.print_summary()



    ###~~~~~~~~~~~~~~~~~~~~~~~~~Euclidean RBF (INCLUDING PEAK) ~~~~~~~~~~~~~~~~~~~~~~~~~###
    print("Now fitting peak euc rbf")
    # Make cartesian grid
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2) #all coordinates on grid

    # Get distances: euc
    distances = jnp.sqrt(square_dist(X, 1)) #precompute distances between values in X (rows single position vectors w corresponding distances), ls not needed

    N_SUBS = 1  # Fitting x subject
    N_BLOCKS = 1  # Single block
    model_outputsPRE = {}
    data = df_phase
    unique_runs = [0,1,2,3,4,5,6,7,8]

    #fit x run
    for id_val in subject:
        print("running: ", id_val)
        for run_val in unique_runs:
            
            # Subset data for current ID
            subset = data[(data["ID"] == id_val) & (data["run"] == run_val)]

            if subset.empty:
                print(f"Skipping ID {id_val}, run {run_val} (no data)")
                continue
            
            N_TRIALS = len(subset["rating"])

            cs = subset["csIDX"].unique() #cs position idx
            val = subset["csValue"].unique() #cs value

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

            # save diagnostics to df
            posterior_samples = model_output.get_samples()
            samples = model_output.get_samples(group_by_chain=True)
            diagnostics = summary(samples, prob=0.9)
            df = pd.DataFrame.from_dict(diagnostics, orient='index')
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

            df["run"] = run_val
            df["ID"] = id_val
            df["model"] = "euc_rbf_peak"
            df["ratings"] = ph
            
            az_data = az.from_numpyro(model_output)
            waic_res = az.waic(az_data)
            loo_res = az.loo(az_data)
            df["WAIC"] = waic_res.elpd_waic
            df["WAIC_P"] = waic_res.p_waic
            df["LOO"] = loo_res.elpd_loo
            df["LOO_P"] = loo_res.p_loo
            #print(df)
            save_diagnostics_df(df, id_val, run_val, "euc_rbf_peak", f"gp_fitted/{ph}")
            model_output.print_summary()

