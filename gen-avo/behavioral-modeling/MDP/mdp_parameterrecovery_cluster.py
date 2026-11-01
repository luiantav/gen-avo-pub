import numpy as np
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import jit
from numpyro.infer import MCMC, NUTS
from typing import Union, Tuple
from jax import lax
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import os

# Custom
from mdp_functions import *
from mdp_functions import GridMDP
from mdp_functions import GridMDPDynamic

#saving 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
directory = os.path.join(parent_dir, "fitted/mdp_parameterrecovery")

# Select models to run 
run_model_A = True
run_model_B = True
run_model_C = True
run_model_D = True
run_model_E = True

# Common config
size = 11
start = (1, 5)
goal = (9, 5)
goal_value = 0
step_cost = -0.04
gamma = 0.9
use_softmax = True
reward_var = 2  # for reward generation
p = []

# Init. results df
all_results = []

# Model A: balanced/ unweighted
if run_model_A:
    beta_values = np.linspace(0.01, 40, 100)
    w = 0
    model = 'basic'
    k = 1
    goal_value = 0 
    
    for beta in beta_values:
        print(f"\n[Model: basic] beta={beta}")
        rewards = create_gaussian_reward_function(size, reward_var)
        mdp_sim = GridMDP(model, size, start, goal, goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, joint_risk_unweighted, k)
        mdp_sim.value_iteration()
        trajectory = mdp_sim.simulate_agent()
        actions = convert_state_sequence_to_actions(trajectory)

        mdp_fit = GridMDP(model, size, trajectory[0], trajectory[-1], goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, joint_risk_unweighted, k)
        fitted = mdp_fit.fit(actions)

        all_results.append({
            "model": "basic",
            "input_beta": beta,
            "input_weight": np.nan,
            "fitted_beta": fitted[0][0],
            "fitted_weight": np.nan,
            "input_p": np.nan,
            "fitted_p": np.nan
        })

        df_sv = pd.DataFrame(all_results)
        output_path = os.path.join(directory, f"mdf_paramrec_{model}_{beta}.csv")
        df_sv.to_csv(output_path, index=False)



# Model B: weighted
if run_model_B:

    weight_values = np.linspace(0.1, 0.9, 10)
    beta_values = np.linspace(0.01, 40, 10)

    model = "weighted"
    k = 2
    for beta in beta_values:
        for w in weight_values:
            print(f"\n[Model: weighted] beta={beta}, weight={w}")
            rewards = create_gaussian_reward_function(size, reward_var)
            mdp_sim = GridMDP(model, size, start, goal, goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, joint_risk_weighted, k)
            mdp_sim.value_iteration()
            trajectory = mdp_sim.simulate_agent()
            actions = convert_state_sequence_to_actions(trajectory)

            mdp_fit = GridMDP(model, size, trajectory[0], trajectory[-1], goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, joint_risk_weighted, k)
            fitted = mdp_fit.fit(actions)

            all_results.append({
                "model": "weighted",
                "input_beta": beta,
                "input_weight": w,
                "fitted_beta": fitted[0][0],
                "fitted_weight": fitted[0][1],
                "input_p": np.nan,
                "fitted_p": np.nan
            })

            df_sv = pd.DataFrame(all_results)
            output_path = os.path.join(directory, f"mdf_paramrec_{model}_{beta}_{w}.csv")
            df_sv.to_csv(output_path, index=False)


# Model C: dynamic unweighted
if run_model_C:

    beta_values = np.linspace(0.01, 40, 100)
    
    w = 0.5
    model = "dynbasic"
    k = 1
    for beta in beta_values:
        print(f"\n[Model: dynbasic] beta={beta}")
        rewards = create_gaussian_reward_function(size, reward_var)
        mdp_sim = GridMDPDynamic(size, start, goal, goal_value, step_cost, use_softmax, w, beta, gamma, rewards, dynamic_risk_unweighted, 1)
        trajectory = mdp_sim.simulate_agent()
        actions = convert_state_sequence_to_actions(trajectory)

        mdp_fit = GridMDPDynamic(size, trajectory[0], trajectory[-1], goal_value, step_cost, use_softmax, w, beta, gamma, rewards, dynamic_risk_unweighted, 1)
        fitted = mdp_fit.fit(actions)

        all_results.append({
            "model": "dynbasic",
            "input_beta": beta,
            "input_weight": w,
            "fitted_beta": fitted[0][0],
            "fitted_weight": np.nan,
            "input_p": np.nan,
            "fitted_p": np.nan
        })

        df_sv = pd.DataFrame(all_results)
        output_path = os.path.join(directory, f"mdf_paramrec_{model}_{beta}.csv")
        df_sv.to_csv(output_path, index=False)


if run_model_D:
    weight_values = np.linspace(0.1, 0.9, 10)
    beta_values = np.linspace(0.01, 40, 10)

    model = "dynweighted"
    k = 2
    for w in weight_values:
        for beta in beta_values:
            print(f"\n[Model: dynweighted] beta={beta}, weight={w}")
            rewards = create_gaussian_reward_function(size, reward_var)
            mdp_sim = GridMDPDynamic(size, start, goal, goal_value, step_cost, use_softmax, w, beta, gamma, rewards, dynamic_risk_weighted, k)
            trajectory = mdp_sim.simulate_agent()
            actions = convert_state_sequence_to_actions(trajectory)

            mdp_fit = GridMDPDynamic(size, trajectory[0], trajectory[-1], goal_value, step_cost, use_softmax, w, beta, gamma, rewards, dynamic_risk_weighted, k)
            fitted = mdp_fit.fit(actions)

            all_results.append({
                "model": "dynweighted",
                "input_beta": beta,
                "input_weight": w,
                "fitted_beta": fitted[0][0],
                "fitted_weight": fitted[0][1],
                "input_p": np.nan,
                "fitted_p": np.nan
            })

            df_sv = pd.DataFrame(all_results)
            output_path = os.path.join(directory, f"mdf_paramrec_{model}_{beta}_{w}.csv")
            df_sv.to_csv(output_path, index=False)




if run_model_E:
    model = "pessimistic" #here simple pessimism + step cost
    beta_values = np.linspace(0.01, 40, 10)
    p_values = np.linspace(0.5, 1, 10)


    w = []
    k = 2
    goal_value = 1
    
    for p in p_values:
        for beta in beta_values:
            print(f"\n[Model: pessimisticcost] p={p}")
            #rewards = create_gaussian_reward_function(size, reward_var)

            rewards = np.zeros((11, 11)) #1 hot
            rewards[5, 5] = -1

            mdp_sim = GridMDP(model, size, start, goal, goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, risk_pessimistic_costly, k)
            mdp_sim.value_iteration_pessimistic()
            trajectory = mdp_sim.simulate_agent()
            actions = convert_state_sequence_to_actions(trajectory)

            mdp_fit = GridMDP(model, size, trajectory[0], trajectory[-1], goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, risk_pessimistic_costly, k)
            fitted = mdp_fit.fit(actions)

            all_results.append({
                "model": "pessimistic",
                "input_beta": beta,
                "input_weight": np.nan,
                "fitted_beta": fitted[0][0],
                "fitted_weight": np.nan,
                "input_p": p,
                "fitted_p": fitted[0][1]
            })
            df_sv = pd.DataFrame(all_results)
            output_path = os.path.join(directory, f"mdf_paramrec_{model}_{p}.csv")
            df_sv.to_csv(output_path, index=False)



#df_all = pd.DataFrame(all_results)
#print(df_all.head())
#print(df_all.tail())

#save
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
#directory = os.path.join(parent_dir, "fitted/mdp_parameterrecovery")
#output_path = os.path.join(directory, f"parameterrecovery.csv")
#df_all.to_csv(output_path, index=False)