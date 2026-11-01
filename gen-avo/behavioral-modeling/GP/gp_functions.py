import numpy as np
import numpyro.distributions as dist
from typing import Union, Tuple
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
import random
import pandas as pd
print('here')

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky
print(jax.__version__)


# Precompute Euc. distances
@jit
def square_dist(X: np.ndarray, ls: float, Xs: np.ndarray = None) -> np.ndarray:
    """Calculates distance between values in two arrays

    Adapted from PyMC3: https://github.com/pymc-devs/pymc/blob/main/pymc/gp/cov.py

    Args:
        X (np.ndarray): N-dimensional array
        Xs (np.ndarray, optional): N-dimensional array for which to calculate distance
        with X. If not provided, distance is calculated for X with itself. Defaults to None.

    Returns:
        np.ndarray: Distance matrix
    """
    X = jnp.multiply(X, 1.0)
    X2 = jnp.sum(jnp.square(X), 1)
    if Xs is None:
        sqd = -2.0 * jnp.dot(X, jnp.transpose(X)) + (
            jnp.reshape(X2, (-1, 1)) + jnp.reshape(X2, (1, -1))
        )
    else:
        Xs = jnp.multiply(Xs, 1.0)
        Xs2 = jnp.sum(jnp.square(Xs), 1)
        sqd = -2.0 * jnp.dot(X, jnp.transpose(Xs)) + (
            jnp.reshape(X2, (-1, 1)) + jnp.reshape(Xs2, (1, -1))
        )
    return jnp.clip(sqd, 0.0, jnp.inf)


# Precompute Manhattan distances 
@jit
def manhattan_dist(X: np.ndarray, ls: float, Xs: np.ndarray = None) -> np.ndarray:
    """Calculates Manhattan distance between values in two arrays, 
    subject to a given length scale parameter.

    Args:
        X (np.ndarray): N-dimensional array
        ls (float): Length scale parameter.
        Xs (np.ndarray, optional): N-dimensional array for which to calculate distance
        with X. If not provided, distance is calculated for X with itself. Defaults to None.

    Returns:
        np.ndarray: Manhattan distance matrix
    """
    X = jnp.multiply(X, 1.0 / ls)
    if Xs is None:
        Xs = X
    else:
        Xs = jnp.multiply(Xs, 1.0 / ls)

    # Compute Manhattan distance
    manhattan_dist = jnp.sum(jnp.abs(X[:, None] - Xs[None, :]), axis=2)
    
    return manhattan_dist


@jit
def exp_quad(X: np.ndarray, ls: float, Xs: np.ndarray = None) -> np.ndarray:
    """Exponentiated Quadratic kernel (also known as Squared Exponential or Radial
    Basis Function kernel).

    Args:
        X (np.ndarray): N-dimensional array
        ls (float): Length scale parameter.
        Xs (np.ndarray, optional): N-dimensional array for which to calculate covariance
        with X. If not provided, covariance is calculated for X with itself. Defaults to None.

    Returns:
        np.ndarray: Covariance
    """
    return jnp.exp(-0.5 * square_dist(X, ls, Xs))


@jit
def dist_to_exp_quad(dist: np.ndarray, ls: float) -> np.ndarray:
    """Converts a (non-squared) distance matrix to exponentiated quadratic (RBF) covariance matrix

    Args:
        dist (np.ndarray): Distance matrix
        ls (float): Length scale parameter

    Returns:
        np.ndarray: Covariance matrix
    """
    cov = (dist * (1 / ls)) ** 2

    return jnp.exp(-0.5 * cov)



@jit
def gp_predict(
    X: np.ndarray, y: np.ndarray, distances: np.ndarray, ls: float, mean: float = 0
) -> Union[np.ndarray, np.ndarray]:
    """Provides estimates of the mean and variance for a gaussian process at a set of points, based on
    prior observations. Uses a Exponentiated Quadratic kernel.

    Note: Takes a precomputed (non-squared) distance matrix instead of raw data to speed up computation slightly

    This code is adapted from the GPy implementation of GP posteriors:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/posterior.py

    This uses a slower but more accurate and more stable method to calculate covariance that involves
    solving equations properly rather than using matrix inversion
    (see https://github.com/SheffieldML/GPy/issues/253)

    Args:
        X (np.ndarray): CS+ location
        y (np.ndarray): CS+ value
        distances (np.ndarray): Distance matrix for all possible X locations.
        ls (float): Length scale parameter.
        mean (float, optional): Mean function. Assumed to be uniform, and is set to the value provided.
        Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: Returns mean and variance at the supplied points.
    """

    K = dist_to_exp_quad(distances[X[:, None], X[None, :]], ls) #RBF cov matrix, distances to x. 
    K = K + (1e-4 * jnp.identity(K.shape[0]))

    print(K)
    print(K.shape)
    LW = cholesky(K, lower=True, overwrite_a=True, check_finite=False) #inverts
    woodbury_vector = jax.scipy.linalg.cho_solve(
        (LW, True), y[:, None], check_finite=False, overwrite_b=True
    )

    Kx = dist_to_exp_quad(distances[X, :], ls)  # X test is all indices in the distance matrix

    # Compute mean
    mu = jnp.dot(Kx.T, woodbury_vector) + mean

    # Compute variance
    Kxx = jnp.diag(dist_to_exp_quad(distances, ls))
    tmp = jax.scipy.linalg.solve_triangular(LW, Kx, lower=True, check_finite=False, overwrite_b=True)
    var = (Kxx - jnp.square(tmp).sum(0))[:, None]

    return mu, var



@jit
def dist_to_linear_decay(dist: np.ndarray, ls: float) -> np.ndarray:
        # Linear decay: w values >0
        return jnp.maximum(1 - dist / ls, 0)
    
@jit
def gp_predict_lin(
    X: np.ndarray, y: np.ndarray, distances: np.ndarray, ls: float, mean: float = 0
) -> Union[np.ndarray, np.ndarray]:
    """GP using a linear decay kernel based on distances.

    Args:
        X (np.ndarray): Locations of previous observations.
        y (np.ndarray): Results of previous observations (e.g. prob of bad weather)
        distances (np.ndarray): Distance matrix for all possible X locations.
        ls (float): Length scale parameter.
        mean (float, optional): Mean function. Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: Returns mean and variance at the supplied points.
    """

    # Covariance matrix using linear kernel 
    K = dist_to_linear_decay(distances[X[:, None], X[None, :]], ls)
    K = K + (1e-4 * jnp.identity(K.shape[0]))

    # Cholesky decomposition
    LW = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
    
    # Woodbury vector
    woodbury_vector = jax.scipy.linalg.cho_solve(
        (LW, True), y[:, None], check_finite=False, overwrite_b=True
    )

    # Covariance between CS and all otehr points 
    Kx = dist_to_linear_decay(distances[X, :], ls)

    # Compute mean
    mu = jnp.dot(Kx.T, woodbury_vector) + mean

    # Diagonal of covariance matrix
    Kxx = jnp.diag(dist_to_linear_decay(distances, ls))

    # Solve for variance
    tmp = jax.scipy.linalg.solve_triangular(
        LW, Kx, lower=True, check_finite=False, overwrite_b=True
    )
    
    var = (Kxx - jnp.square(tmp).sum(0))[:, None]

    return mu, var


# Manhattan RBF
@jit
def gp_predict_man_rbf(
    X: np.ndarray, y: np.ndarray, distances: np.ndarray, ls: float, mean: float = 0
) -> Union[np.ndarray, np.ndarray]:
    """GP using a rbf kernel based on Manhattan distances.

    Args:
        X (np.ndarray): Locations of previous observations.
        y (np.ndarray): Results of previous observations (e.g. prob of bad weather)
        distances (np.ndarray): Distance matrix for all possible X locations.
        ls (float): Length scale parameter.
        mean (float, optional): Mean function. Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: Returns mean and variance at the supplied points.
    """

    K = dist_to_exp_quad(distances[X[:, None], X[None, :]], ls)
    
    K = K + (1e-4 * jnp.identity(K.shape[0]))

    print(K)
    print(K.shape)
    LW = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
    woodbury_vector = jax.scipy.linalg.cho_solve(
        (LW, True), y[:, None], check_finite=False, overwrite_b=True
    )

    Kx = dist_to_exp_quad(
        distances[X, :], ls
    )

    # Compute mean
    mu = jnp.dot(Kx.T, woodbury_vector) + mean

    Kxx = jnp.diag(dist_to_exp_quad(distances, ls))
    tmp = jax.scipy.linalg.solve_triangular(
        LW, Kx, lower=True, check_finite=False, overwrite_b=True
    )

    # Compute variance
    var = (Kxx - jnp.square(tmp).sum(0))[:, None]

    return mu, var





def sampleLUI(
    model_func: callable,
    prompts: np.ndarray,
    ratings: np.ndarray,
    csloc: np.ndarray,
    csval: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    n_samples: int,
    n_warmup: int,
    num_chains: int,
    seed: int = 0,
    max_tree_depth: Union[int, Tuple] = 10,
) -> MCMC:
    """
    Samples from the posterior distribution of the model using NUTS, implemented in NumPyro.

    Args:
        model_func (callable): Model function
        prompts (np.ndarray): Coordinates of prompts
        ratings (np.ndarray): Subject of ratings
        distances (np.ndarray): Matrix of distances between all locations on the grid
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        n_samples (int, optional): Number of samples. Defaults to 4000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 2000.
        num_chains (int, optional): Number of chains to run. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 0.
        max_tree_depth (Union[int, Tuple], optional): Maximum tree depth for the NUTS algorithm. Defaults to 10.

    Returns:
        MCMC: Results of sampling.
    """

    nuts_kernel = NUTS(model_func, max_tree_depth=max_tree_depth, target_accept_prob = 0.9)
    print(f"num_chains: {num_chains}")
    print(f"n_warmup: {n_warmup}")
    print(f"n_samples: {n_samples}")
    
    mcmc = MCMC(
        nuts_kernel, num_chains=num_chains, num_samples=n_samples, num_warmup=n_warmup 
    )

    rng_key = jax.random.PRNGKey(seed)

    mcmc.run(
        rng_key,
        prompts,
        ratings,
        csloc,
        csval,
        distances,
        n_subs,
        n_blocks,
        n_trials,
    )

    return mcmc




#fitting both ls and csvalue
def grid_model_SIM_ALT(
    prompts: np.ndarray,
    ratings: np.ndarray,
    csloc: np.ndarray,
    csval: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
):
    """
    Try fitting both ls and csvalue

    """

    # If centered 
    #ls_group_mean, ls_group_sd, ls_subject_transformed= create_subject_params("ls", n_subs)

    # If not centered 
    ls_group_mean, ls_group_sd, ls_offset= create_subject_params("ls", n_subs)
    
    # Subject-level
    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 50 + 0.1, #from 10 to 40
    )
     
    #PEAK
    peak_group_mean, peak_group_sd, peak_offset= create_subject_params_extended("peak", n_subs)

    # Subject-level
    peak_subject_transformed = numpyro.deterministic(
        "peak_subject_transformed",
        jax.scipy.special.expit(peak_group_mean + peak_group_sd * peak_offset) * 100, 
    )
    

    # Run model --> this would retun mu and var 
    res = model_func_SIM_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        0,
        csloc, #here single CS+ ratings goes in 
        peak_subject_transformed,
        distances,
        n_trials,
    )
    
    res = numpyro.deterministic('res', res)
    

    #ll
    numpyro.sample(
        "obs",
        dist.Normal(res.squeeze()[0][prompts], jnp.sqrt(res.squeeze()[1][prompts])), #select only tiles that were rated
        obs=ratings,
    )



def grid_model_LIN_ALT(
    prompts: np.ndarray,
    ratings: np.ndarray,
    csloc: np.ndarray,
    csval: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
):
    """
    Simple model for the grid task.

    Args:
        prompts (np.ndarray): Coordinates of locations where ratings were prompted
        rating (np.ndarray): PPT/SIM Ratings 
        distances (np.ndarray): Distances between each pair of coordinates
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
    """

    # Length scale
    ls_group_mean, ls_group_sd, ls_offset = create_subject_params("ls", n_subs)
    
    # Subject-level
    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 100 + 0.1, #ensures it is positive    
    )

    #PEAK
    peak_group_mean, peak_group_sd, peak_offset= create_subject_params_extended("peak", n_subs)

    # Subject-level
    peak_subject_transformed = numpyro.deterministic(
        "peak_subject_transformed",
        jax.scipy.special.expit(peak_group_mean + peak_group_sd * peak_offset) * 100, 
    )

    # Run model --> this would retun mu and var 
    res = model_func_LIN_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        0,
        csloc, #here single CS+ ratings goes in 
        peak_subject_transformed,
        distances,
        n_trials,
    )
    
    res = numpyro.deterministic('res', res)
    

    #ll
    numpyro.sample(
        "obs",
        dist.Normal(res.squeeze()[0][prompts], jnp.sqrt(res.squeeze()[1][prompts])), #select only tiles that were rated
        obs=ratings,
    )




def model_func_SIM(
    ls: np.ndarray,
    mean: float,
    csloc: np.ndarray,
    csval: np.ndarray,
    distances: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """
    Iterates over trials, applying the GP model across blocks & subjects at each trial.
    //uses no decision strategy (for now)

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        ls (np.ndarray): Length scale parameter for GP kernel
        mean (float): Value of constant mean function
        prompts (np.ndarray): Previously prompted points on the grid. - Should be provided as a
        2D array of shape (n subjects * n blocks, n trials)
        ratings (np.ndarray): Ratings provided at points X_observed - Should be provided as a 2D array of shape (n subjects * n blocks, n trials)
        distances (np.ndarray): Distance matrix for all possible X locations.
        n_trials (int): Number of trials. The model is applied to all but the last trial,
        as no choices are observed after the last outcome.

    Returns:
        np.ndarray: not sure 
    """

    this_loc = csloc.reshape(-1) 
    this_val = csval.reshape(-1) 

    # Get mean and variance
    mu, var = gp_predict(this_loc, this_val, distances, ls, mean)
    
    res = (mu,var)
    
    return jnp.stack(res)


# Set number of trials to static
model_func_SIM_jit = jit(model_func_SIM, static_argnums=(5,))





def model_func_LIN(
    ls: np.ndarray,
    mean: float,
    csloc: np.ndarray,
    csval: np.ndarray,
    distances: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """
    Iterates over trials, applying the GP model across blocks & subjects at each trial.
    //uses no decision strategy (for now)

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        ls (np.ndarray): Length scale parameter for GP kernel
        mean (float): Value of constant mean function
        prompts (np.ndarray): Previously prompted points on the grid. - Should be provided as a
        2D array of shape (n subjects * n blocks, n trials)
        ratings (np.ndarray): Ratings provided at points X_observed - Should be provided as a 2D array of shape (n subjects * n blocks, n trials)
        distances (np.ndarray): Distance matrix for all possible X locations.
        n_trials (int): Number of trials. The model is applied to all but the last trial,
        as no choices are observed after the last outcome.

    Returns:
        np.ndarray: not sure 
    """
    
    this_loc = csloc.reshape(-1) 
    this_val = csval.reshape(-1) 

    # Get mean and variance
    mu, var = gp_predict_lin(this_loc, this_val, distances, ls, mean)
    
    res = (mu,var)
    
    return jnp.stack(res)


# Set number of trials to static
model_func_LIN_jit = jit(model_func_LIN, static_argnums=(5,))




def create_subject_params_extended(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    ls and peak 
    """

    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(0.5))
    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0.85, 0.5))
    offset = numpyro.sample("{0}_offset".format(name), dist.Normal(0, 0.5), sample_shape=(n_subs,))
    

    return group_mean, group_sd, offset




def sample_coordinates_by_distance(distance_euc, target_distances):
    """
    Returns 5 random coordinates on the grid, each corresponding to a unique Euclidean distance.

    ...
    """
    coordinates = []
    
    for dist in target_distances:
        # Get all the positions where the distance matches the target distance
        matching_positions = np.argwhere(np.round(distance_euc, 1) == dist)
        
        chosen_position = matching_positions[np.random.choice(len(matching_positions))]
        coordinates.append(tuple(chosen_position))
    
    return coordinates



def extract_sampled_values(grids, sampled_coords, grid_size=11):
    """
    Extract values from the grids corresponding to sampled coordinates.

    Args:
        grids (np.ndarray): Array of flattened grids (shape: [ls_length, grid_size * grid_size]).
        sampled_coords (list of tuples): List of sampled coordinates [(x1, y1), (x2, y2), ...].
        grid_size (int): Size of the grid (default is 11x11).

    Returns:
        pd.DataFrame: DataFrame containing the sampled coordinates, their indices, and the corresponding values from grids.
    """
    sampled_values_list = []
    coord_indices = [x * grid_size + y for x, y in sampled_coords]  # coordinates to indices

    for i, grid in enumerate(grids):
        sampled_values = [grid[idx] for idx in coord_indices]
        sampled_values_list.append(sampled_values)

    # Create and populate df
    df_list = []
    for i, sampled_values in enumerate(sampled_values_list):
        df = pd.DataFrame({
            'coordinate': sampled_coords,
            'rating': sampled_values
        })
        
        # Make pretty and add info 
        df[['x', 'y']] = pd.DataFrame(df['coordinate'].tolist(), index=df.index)
        df["promptIDX"] = df["x"] * grid_size + df["y"]
        df["ls_index"] = i
        df["ID"] = i

        df_list.append(df)

    # combine dfs
    final_df = pd.concat(df_list, ignore_index=True)

    return final_df



@jit
def softmax(v: np.ndarray, b: float):
    prob = jnp.exp(v / b)
    prob = jnp.nan_to_num(prob, nan=0)
    prob += 1e-20  # Make sure probabilities are never exactly zero

    # Normalise
    prob = prob / prob.sum()
    prob = jnp.clip(prob, 0.0001, 0.9999)
    return prob


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_subject_params(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.

    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects

    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(0.5))  # use with expit 
    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 0.5)) #0.5
    offset = numpyro.sample("{0}_offset".format(name), dist.Normal(0, 0.5), sample_shape=(n_subs,)) #increased from 0, 0.5
    

    return group_mean, group_sd, offset

