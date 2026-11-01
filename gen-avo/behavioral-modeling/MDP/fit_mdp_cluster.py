print("Script started")

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from pathlib import Path
import argparse
from mdp_functions import *
from mdp_functions import GridMDP
from mdp_functions import GridMDPDynamic
import numpy as np
import os

#specify models to fit 
pessimistic = True
basic = False
weighted = True
dyn_unweighted = False
dyn_weighted = False
baseline = True 
heuristic = True


############### Get data & Set paths ###########
here = Path(__file__).resolve().parent
reward_path = here / "fitted_mu.csv" # mu predictions from fitted GPs
print(reward_path)
rews_imported = pd.read_csv(reward_path)

one_up = here.parent 
trajectory_path = one_up / "data-for-models/alldata.csv" #ppt data with trajectories
print(trajectory_path)
df = pd.read_csv(trajectory_path)


################ PREP Cluster ###################
run_on_cluster = 1

if run_on_cluster == 1:
    # Parse Sub name 
    parser = argparse.ArgumentParser(description='Process subject ID.')
    parser.add_argument('-s', '--subject', type=str, required=True, help='Subject ID')
    args = parser.parse_args()
    subject = [args.subject]
else:
    subject = ["294353176824689"]

print("subject = ", subject)

############## MDP Setup ########################
size = 11
step_cost = -0.04
gamma = 0.9
use_softmax = True

#init all parameters
w = np.zeros(1)
beta = np.zeros(1)
p = np.zeros(1)
rule = np.zeros(1)
results = []


#get imported rewards (from GPs)
ppt_runs = rews_imported[['participant', 'run']].drop_duplicates()
ppt_runs = ppt_runs[ppt_runs['participant'].isin(subject)]
print(ppt_runs)


# Iterate & fit over each (participant, run, trial) 
for _, row in ppt_runs.iterrows():
    ppt_id = row['participant']
    run_id = row['run']
    
    trial_subset = df[(df['ppt'] == ppt_id) & (df['run'] == run_id)]
    if trial_subset['trial'].nunique() < 10:
        print(f"Skipping entire run: ppt {ppt_id}, run {run_id} — incomplete trials")
        continue

    # Get rewards (once per run) and make usable for model
    ppt_rews = rews_imported[(rews_imported['participant'] == ppt_id) & (rews_imported['run'] == run_id)]
    
    rewards_str = ppt_rews["rewards"].iloc[0]
    rewards_clean = rewards_str.replace('[', '').replace(']', '').replace('\n', ' ')
    rewards_array = np.fromstring(rewards_clean, sep=' ')    
    rewards = rewards_array.reshape((11, 11))
    rewards = -1 * (rewards / 100)
    rewards_aligned = np.rot90(rewards, k=4) # k=3 rotates, k=4 leaves at original 
    #print(np.round(rewards_aligned, decimals=2))

    # Iterate over trials
    for trial_id in range(10):
        ppt_df = df[(df['ppt'] == ppt_id) & (df['run'] == run_id) & (df['trial'] == trial_id)]
        print(f"ppt: {ppt_id}, run: {run_id}, trial: {trial_id}")

        #get trajectory for trial from data 
        steps_pretty = process_steps(ppt_df["path"].iloc[0])
        trajectory = convert_state_sequence_to_actions(steps_pretty)
        positions_only = [step[0] for step in trajectory]

        print(trajectory[0][0]) #check start/goal
        print(trajectory[-1][0])

        #fit all models set to True         
        if heuristic:
            print("Fitting heuristic")
            model = 'heuristic'
            k = 1 #n params, 1 (distance kept)
            goal_value = 0 

            mdp_fit = GridMDP(
                model, 
                size, 
                trajectory[0][0], 
                trajectory[-1][0], 
                goal_value, 
                step_cost, 
                rewards_aligned, 
                use_softmax, 
                w.copy(),
                beta.copy(),
                p.copy,
                gamma, 
                joint_risk_weighted, #can be replaced by any reward function, not accounted for 
                k,
                rule
            )
            #optimal_params_r, neg_ll_r, aic_r, bic_r = mdp_fit.fit_heuristic(trajectory)
            optimal_params_r, neg_ll_r, aic_r, bic_r = mdp_fit.fit_heuristic_float(trajectory)



        if pessimistic:
            print("Fitting pessimistic")
            model = 'pessimistic'
            k = 2 # Beta, p
            goal_value = 1

            #custom rewards i.e. 0/1 assumptions of pessimistic
            rewards_custom = np.zeros((11, 11))
            cs_df = df[(df['ppt'] == ppt_id) & (df['run'] == run_id) & (df['trial'] == 1)] #get cs+ location
            cs_pos = cs_df["csPosition"].iloc[0]
            cs_idx = tuple(ast.literal_eval(cs_pos))
            rewards_custom[cs_idx] = -1 
            #print(rewards_custom)

            #init MDP
            mdp_fit = GridMDP(
                model, 
                size, 
                trajectory[0][0], 
                trajectory[-1][0], 
                goal_value, 
                step_cost, 
                rewards_custom, 
                use_softmax, 
                w.copy(),
                beta.copy(),
                p.copy(),
                gamma, 
                risk_pessimistic_costly, # also acconts for step cost (not pure baseline)
                k,
                rule
            )
            
            optimal_params_p, neg_ll_p, aic_p, bic_p = mdp_fit.fit(trajectory)


        if basic:
            print("Fitting unweighted")
            model = 'basic'
            k = 1 #w
            goal_value = 0 

            mdp_fit = GridMDP(
                model, 
                size, 
                trajectory[0][0], 
                trajectory[-1][0], 
                goal_value, 
                step_cost, 
                rewards_aligned, 
                use_softmax, 
                w.copy(),
                beta.copy(), #tau.copy(),
                p.copy(),
                gamma, 
                joint_risk_unweighted,
                k, #n params
                rule
            )
            optimal_params, neg_ll, aic, bic = mdp_fit.fit(trajectory)
        #print("Fitted parameters:", optimal_params)
        #print("Likelihood at fitted parameters:", neg_ll)


        if weighted:
            print("Fitting weighted")
            model = 'weighted'
            k = 2 #beta, w
            goal_value = 0 

            mdp_fit = GridMDP(
                model,
                size, 
                trajectory[0][0], 
                trajectory[-1][0], 
                goal_value, 
                step_cost, 
                rewards_aligned, 
                use_softmax, 
                w.copy(),
                beta.copy(),
                p.copy(),
                gamma, 
                joint_risk_weighted,
                k,
                rule
            )
            optimal_params_w, neg_ll_w, aic_w, bic_w = mdp_fit.fit(trajectory)
            #print("Fitted parameters:", optimal_params_w)
            #print("Likelihood at fitted parameters:", neg_ll_w)




        #preview
        #print(f"basic: {bic}, weighted: {bic_w}, dyn_w: {bic_d_w}, dyn: {optimal_params_d_w[0]}, pess: {bic_p}, greedy: {bic_b}")

        # Save results
        results.append({
            'participant': ppt_id,
            'run': run_id,
            'trial': trial_id,
            'beta_weigthed': optimal_params_w[0],
            'weight_weigthed': optimal_params_w[1],
            'll_weighted': neg_ll_w,
            'aic_w': aic_w,
            'bic_w': bic_w,
            'll_baseline': neg_ll_b,
            'aic_baseline': aic_b,
            'bic_baseline': bic_b,
            'beta_pessimistic': optimal_params_p[0],
            'p_pessimistic': optimal_params_p[1],
            'll_pessimistic': neg_ll_p,
            'aic_p': aic_p,
            'bic_p': bic_p,
            'beta_rule': optimal_params_r[0],
            'r_rule': optimal_params_r[1],
            'll_rule': neg_ll_r,
            'aic_r': aic_r,
            'bic_r': bic_r

        })
        

# Convert to DataFrame and save
results_df = pd.DataFrame(results)

#save 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
directory = os.path.join(parent_dir, "fitted/mdp_fitted")
output_path = os.path.join(directory, f"mdp_{ppt_id}.csv")
results_df.to_csv(output_path, index=False)