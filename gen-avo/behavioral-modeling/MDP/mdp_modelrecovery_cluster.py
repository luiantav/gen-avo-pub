import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import scipy.optimize as opt

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import ast
import random
import os
import itertools
import argparse

from mdp_functions import *
from mdp_functions import GridMDP
from mdp_functions import GridMDPDynamic


#saving 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))

directory = os.path.join(parent_dir, "fitted/mdp_modelrecovery")


FITMODELS = ['baseline','threshold','weighted', 'pessimistic']


################ PREP Cluster ###################
run_on_cluster = 0

if run_on_cluster == 1:
    # Parse Sub name 
    parser = argparse.ArgumentParser(description='Process model ID.')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    args = parser.parse_args()
    MODEL = [args.subject]
else:
    MODEL = ["pessimistic"]


print("sim model = ", MODEL)


def fit_all_models(trajectory, size, step_cost, gamma, use_softmax):
    print("now fitting")
    fit_results = {}
    start = trajectory[0][0]
    goal = trajectory[-1][0]

    for model in FITMODELS:
        if model == 'basic':
            print("basic")
            k = 1
            goal_value = 0
            rewards_model = create_gaussian_reward_function(size, 2)
            mdp = GridMDP(model, size, start, goal, goal_value, step_cost, rewards_model, use_softmax,
                          w, beta, p, gamma, joint_risk_unweighted, k, rule)
            _, _, _, bic = mdp.fit(trajectory)

        elif model == 'weighted':
            print("weighted")
            k = 2
            goal_value = 0
            rewards_model = create_gaussian_reward_function(size, reward_var)
            mdp = GridMDP(model, size, start, goal, goal_value, step_cost, rewards_model, use_softmax,
                          w, beta, p, gamma, joint_risk_weighted, k, rule)
            _, _, _, bic = mdp.fit(trajectory)

        elif model == 'pessimistic':
            print("pessimistic")            
            k = 2
            goal_value = 1
            rewards_model = np.zeros((size, size))
            rewards_model[5, 5] = -1 
            rewards_model[goal] = 1

            mdp = GridMDP(model, size, start, goal, goal_value, step_cost, rewards_model, use_softmax,
                          w, beta, p, gamma, risk_pessimistic_costly, k, rule)
            _, _, _, bic = mdp.fit(trajectory)

        elif model == 'dyn_unweighted':
            print("dyn_unweighted")
            k = 1
            goal_value = 0
            rewards_model = create_gaussian_reward_function(size, reward_var)
            mdp = GridMDPDynamic(size, start, goal, goal_value, step_cost, use_softmax,
                                 w, beta, gamma, rewards_model, dynamic_risk_unweighted, k, rule)
            _, _, _, bic = mdp.fit(trajectory)

        elif model == 'dyn_weighted':
            print("dyn_weighted")
            k = 2
            goal_value = 0
            rewards_model = create_gaussian_reward_function(size, reward_var)
            mdp = GridMDPDynamic(size, start, goal, goal_value, step_cost, use_softmax,
                                 w, beta, gamma, rewards_model, dynamic_risk_weighted, k, rule)
            _, _, _, bic = mdp.fit(trajectory)

        elif model == 'baseline':
            print("baseline")
            k = 0
            goal_value = 0
            # Reward irrelevant; fit_greedy uses greedy behavior
            mdp = GridMDP(model, size, start, goal, goal_value, step_cost, np.zeros((size, size)), use_softmax,
                          w, beta, p, gamma, joint_risk_weighted, k, rule)
            _, _, _, bic = mdp.fit_greedy(trajectory)

        
        elif model == 'threshold':
            print("threshold")
            k = 1
            goal_value = 0
            # Reward irrelevant;
            mdp = GridMDP(model, size, start, goal, goal_value, step_cost, np.zeros((size, size)), use_softmax,
                          w, beta, p, gamma, joint_risk_weighted, k, rule)
            _, _, _, bic = mdp.fit_heuristic_float(trajectory)


        fit_results[model] = bic

    return fit_results


def simulate_trajectory_from_model(model, size, step_cost, gamma, use_softmax, w,beta, p, rule):
    print("now simulating")
    start = (1, 5)
    goal = (9, 5)
    goal_value = 0 if model != 'pessimistic' else 1
    rewards = create_gaussian_reward_function(size, reward_var)

    if model == 'pessimistic':
        rewards = np.zeros((size, size))
        rewards[5, 5] = -1 
        rewards[goal] = 1

    risk_func = {
        'basic': joint_risk_unweighted,
        'weighted': joint_risk_weighted,
        'pessimistic': risk_pessimistic_costly,
        'dyn_unweighted': dynamic_risk_unweighted,
        'dyn_weighted': dynamic_risk_weighted,
        'baseline': joint_risk_weighted,
        'threshold': joint_risk_weighted
    }[model]

    k = {'basic':1, 'weighted':2, 'pessimistic':2, 'dyn_unweighted':1, 'dyn_weighted':2, 'baseline':0, 'threshold':1}[model]

    if 'dyn' in model:
        mdp = GridMDPDynamic(size, start, goal, goal_value, step_cost, use_softmax, w, beta, gamma, rewards, risk_func, k, rule)
        mdp.value_iteration()
        trajectory = mdp.simulate_agent()
    else:
        mdp = GridMDP(model, size, start, goal, goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, risk_func, k, rule)
        if model == 'baseline':
            trajectory = mdp.simulate_greedy_to_goal()
        elif model == 'threshold':
            trajectory = mdp.simulate_heuristic()
        else:
            mdp.value_iteration()
            trajectory = mdp.simulate_agent()

    return trajectory, rewards







#Prep
size = 11
start = (1, 5)
goal = (9, 5)
beta = 20
w = 0.6
gamma =0.9
p = 0.6
step_cost = 0.04
reward_var = 2
use_softmax = True
rule = 2


n_simulations = 100
beta_values = [5, 15, 25, 35]
w_values = [0.2, 0.5, 0.8]
p_values = [0.2, 0.4, 0.6, 0.8]
r_values = [0,1,2,3]

all_combinations = list(itertools.product(beta_values, w_values, p_values, r_values))
n_repeat = int(np.ceil(n_simulations / len(all_combinations)))
all_combinations = all_combinations * n_repeat
all_combinations = all_combinations[:n_simulations]

beta_sim = np.array([c[0] for c in all_combinations])
w_sim = np.array([c[1] for c in all_combinations])
p_sim = np.array([c[2] for c in all_combinations])
r_sim = np.array([c[3] for c in all_combinations])

print("Beta:", beta_sim)
print("Weight:", w_sim)
print("P:", p_sim)
print("Rule:", r_sim)

#Sim & Fit 

for true_model in MODEL:
    for i in range(n_simulations):
        recovery_records = []
        print("Simulating model:", true_model, "current iter:", i)

        w = w_sim[i]
        beta = beta_sim[i]
        p = p_sim[i]
        rule = r_sim[i]
        print("rule", rule)
        trajectory, rewards = simulate_trajectory_from_model(true_model, size, step_cost, gamma, use_softmax, w,beta, p, rule)
        simulated_trajectory = convert_state_sequence_to_actions(trajectory)
        bics = fit_all_models(simulated_trajectory, size, step_cost, gamma, use_softmax)

        best_fit_model = min(bics, key=bics.get)
        recovery_records.append({
            'true_model': true_model,
            'best_fit_model': best_fit_model,
            **bics
        })

        df_rec = pd.DataFrame(recovery_records)
        output_path = os.path.join(directory, f"mdp_modelrec_{true_model}_{i+69}.csv")
        df_rec.to_csv(output_path, index=False)