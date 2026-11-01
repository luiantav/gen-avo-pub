import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import ast
import random



#################General shared functions #################### 

#simulate a GP output 
def create_gaussian_reward_function(grid_size, var, csplus_row=5, csplus_col=5, csplus_value=0.7):
    """Creates a Gaussian reward function centered at (csplus_row, csplus_col) and scales it to csplus_value."""
    mean = [csplus_row, csplus_col]
    cov = [[var, 0], [0, var]]

    # grid coordinates
    x, y = np.arange(grid_size), np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((yy, xx))

    # Generate Gaussian & scale peak to csplus_value
    rv = multivariate_normal(mean, cov)
    grid = rv.pdf(pos)
    grid = (grid / np.max(grid)) * csplus_value  
    reward_function = np.round(grid, 1)  * -1
    #print(reward_function)
    return reward_function



#Get trajectory into usable format
def convert_state_sequence_to_actions(state_sequence):
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    action_data = []

    for i in range(len(state_sequence) - 1):
        state = state_sequence[i]
        next_state = state_sequence[i + 1]

        if state == next_state:
            continue  # Skip if the agent remains in the same statei.e. hits wall 

        # Find action taken
        action = (next_state[0] - state[0], next_state[1] - state[1])

        if action in actions:
            action_data.append((state, action))
        else:
            raise ValueError(f"Invalid transition from {state} to {next_state}") #there should not be

    #include last state without action
    action_data.append((state_sequence[-1], (0,1))) #random action, does not matter

    return action_data


# get json (phaser output) into usable coordinates
def process_steps(step_dict_str):
    
    step_dict = ast.literal_eval(step_dict_str) # string to dict
    sorted_steps = sorted(step_dict.items(), key=lambda x: int(x[0])) # Sort by the keys (step indices)
    ordered_steps = [(v['row'], v['col']) for k, v in sorted_steps] # convert to tuple
    
    return ordered_steps



# Get rewards i.e. mu (GP output) --> takes a little forever
def load_mu_data(directory):
    """Load posterior JSONs and return a DataFrame with full 11x11 Mu grids per participant/run."""
    
    data = []

    # Handle relative path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    directory = os.path.join(parent_dir, directory)

    for filename in os.listdir(directory):
        if filename.startswith("ps_") and filename.endswith(".json"):
            participant = filename[len("ps_"):-5]  # Extract participant ID

            with open(os.path.join(directory, filename), "r") as f:
                json_data = json.load(f)

            for run, run_data in json_data.items():
                # Check for structure and required data
                if not isinstance(run_data, dict) or "posterior_samples" not in run_data:
                    continue
                posterior_samples = run_data["posterior_samples"]
                if "res" not in posterior_samples:
                    continue

                res = np.array(posterior_samples["res"])
                mean_res = jnp.mean(res, axis=0)

                if mean_res.shape[0] < 1 or mean_res.shape[1] < 121:
                    continue

                # Extract and reshape Mu grid
                mu_grid = jnp.rot90(jnp.reshape(mean_res[0, :, 0], (11, 11)), k=1)
                
                # Store as one row: participant, run, rewards
                data.append({
                    "participant": participant,
                    "run": run,
                    "rewards": mu_grid
                })

    return pd.DataFrame(data)




#Plot reward and value grids 
def plot_survival_and_value(env, trajectory, sim_trajectory):
    """
    Plot the survival probabilities and value function grids saved in env,
    along with the trajectory.

    env: your environment instance with
        - survive_not_sink (np.array)
        - survive_not_starve (np.array)
        - joint_survival_score (np.array)
        - value_function (np.array)
        - size (int)
    trajectory: list of (x, y) tuples representing the path taken
    """

    size = env.size
    titles = [
        "Survive Not Sink",
        "Survive Not Starve",
        "Joint Survival Probability",
        "Value Function with Trajectory"
    ]
    cmap_survival = plt.cm.Reds_r
    cmap_value = plt.cm.bwr_r

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))


    # For unified normalization of survival plots
    all_survival = np.concatenate([
        env.survive_not_sink.flatten(),
        env.survive_not_starve.flatten(),
        env.joint_survival_score.flatten()
    ])

    norm_survival = plt.Normalize(vmin=all_survival.min(), vmax=all_survival.max())

    # Plot Survive Not Sink
    ax = axes[0]
    for x in range(size):
        for y in range(size):
            val = env.survive_not_sink[x, y]
            ax.add_patch(plt.Rectangle((y, size - x - 1), 1, 1, color=cmap_survival(norm_survival(val))))
            ax.text(y + 0.5, size - x - 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title(titles[0])

    # Plot Survive Not Starve
    ax = axes[1]
    for x in range(size):
        for y in range(size):
            val = env.survive_not_starve[x, y]
            ax.add_patch(plt.Rectangle((y, size - x - 1), 1, 1, color=cmap_survival(norm_survival(val))))
            ax.text(y + 0.5, size - x - 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title(titles[1])

    # Plot Joint Survival Probability
    ax = axes[2]
    for x in range(size):
        for y in range(size):
            val = env.joint_survival_score[x, y]
            ax.add_patch(plt.Rectangle((y, size - x - 1), 1, 1, color=cmap_survival(norm_survival(val))))
            ax.text(y + 0.5, size - x - 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title(titles[2])
    #x_coords = [step[1] + 0.5 for step in trajectory]
    #y_coords = [size - step[0] - 0.5 for step in trajectory]
    #ax.plot(x_coords, y_coords, color='black', linewidth=2, alpha=0.7)

    # Plot Value Function with Trajectory
    ax = axes[3]
    val_min = env.value_function.min()
    val_max = env.value_function.max()
    norm_value = plt.Normalize(vmin=val_min, vmax=val_max)
    for x in range(size):
        for y in range(size):
            val = env.value_function[x, y]
            ax.add_patch(plt.Rectangle((y, size - x - 1), 1, 1, color=cmap_value(norm_value(val))))
            ax.text(y + 0.5, size - x - 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")

    # Plot trajectory path
    #x_coords = [step[1] + 0.5 for step in sim_trajectory]
    #y_coords = [size - step[0] - 0.5 for step in sim_trajectory]
    #ax.plot(x_coords, y_coords, color='black', linewidth=2, alpha=0.7)

    # Arrows along trajectory
    #for i in range(len(sim_trajectory) - 1):
    #    start = sim_trajectory[i]
    #    end = sim_trajectory[i + 1]
    #    ax.arrow(start[1] + 0.5, size - start[0] - 0.5,
    #             (end[1] - start[1]) * 0.8, -(end[0] - start[0]) * 0.8,
    #             head_width=0.2, head_length=0.2, fc='black', ec='black')

    ax.set_title(titles[3])

    # Final formatting for all subplots
    for ax in axes:
        ax.set_xticks(np.arange(0, size, 1))
        ax.set_yticks(np.arange(0, size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='black')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

    plt.tight_layout()
    plt.show()
    #return fig, axes





#################MDP Models #################### 
#static 
class GridMDP:
    def __init__(self, model, size, start, goal, goal_value, step_cost, rewards, use_softmax, w, beta, p, gamma, risk_function, k, rule):
        self.model = model #basic, weighted, pessimistic
        self.size = size
        self.start = start
        self.goal = goal
        self.goal_value = goal_value
        self.step_cost = step_cost
        self.w = w
        #self.tau = tau  # Softmax temp
        self.beta = beta  # Softmax inverse temperature 
        self.use_softmax = use_softmax  # Whether to use softmax
        self.rewards = rewards
        self.rewards[self.goal] = self.goal_value
        self.actions = [(0,1), (1,0), (0,-1), (-1,0)]
        self.gamma = gamma
        self.threshold = 1e-4
        self.value_function = np.zeros((size, size))
        self.policy_probs = np.zeros((size, size, len(self.actions)))  # for softmax probs
        self.risk_function = risk_function #how to weigh risk 
        self.k = k 
        self.p = p
        self.rule = rule

    def is_valid(self, state):
        return 0 <= state[0] < self.size and 0 <= state[1] < self.size
    
    def get_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        return next_state if self.is_valid(next_state) else state

    def get_step_cost(self, state):
        return self.step_cost * (abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1]))

    def softmax(self, q_values):
        q_values = np.array(q_values)
        max_q = np.max(q_values) 
        shifted = q_values - max_q #improves numerical stability
        exp_q = np.exp(self.beta * shifted)
        return exp_q / np.sum(exp_q)

    def value_iteration(self):

        self.value_function = np.zeros((self.size, self.size))
        self.survive_not_sink = np.zeros((self.size, self.size))
        self.survive_not_starve = np.zeros((self.size, self.size))
        self.joint_survival_score = np.zeros((self.size, self.size))

        while True:
            delta = 0
            new_values = np.copy(self.value_function)

            # Get value function (cumulative rew for following optimal policy )
            for x in range(self.size):
                for y in range(self.size):
                    next_states = [self.get_next_state((x, y), a) for a in self.actions]  
                    q_values = []

                    for s in next_states:
                        joint_survival_score, sink_survival, starve_survival  =  self.risk_function(self.rewards[s], self.get_step_cost(s), self.w)
                        q_value = joint_survival_score + self.gamma * self.value_function[s]
                        q_values.append(q_value)

                        #(store for plotting)
                        self.survive_not_sink[s] = sink_survival
                        self.survive_not_starve[s] = starve_survival
                        self.joint_survival_score[s] = joint_survival_score
                    
                    best_q = max(q_values)  # Best action value
                    delta = max(delta, abs(self.value_function[x, y] - best_q))
                    new_values[x, y] = best_q  # Update value function
 
            self.value_function = new_values
            if delta < self.threshold:  # stop once change too small 
                #plt.imshow(self.value_function, cmap='viridis'); plt.colorbar(); plt.show()
                break


    def value_iteration_pessimistic(self):
        #init for plotting 
        self.value_function = np.zeros((self.size, self.size))
        self.survive_not_sink = np.zeros((self.size, self.size))
        self.survive_not_starve = np.zeros((self.size, self.size))
        self.joint_survival_score = np.zeros((self.size, self.size))

        while True:
            delta = 0
            new_values = np.copy(self.value_function)

            for x in range(self.size):
                for y in range(self.size):

                    #store for plots 
                    joint_survival_score, sink_survival, starve_survival = self.risk_function( self.rewards[(x, y)], self.get_step_cost((x, y)), self.w)
                    self.joint_survival_score[x, y] = joint_survival_score
                    self.survive_not_sink[x, y] = sink_survival
                    self.survive_not_starve[x, y] = starve_survival

                    q_values = []
                    for a in self.actions: #immediate actions

                        s_prime = self.get_next_state((x, y), a)
                        reward, _, _ = self.risk_function( self.rewards[s_prime], self.get_step_cost(s_prime), self.w)
                        #reward = self.rewards[s_prime]
                        q_value = reward + self.gamma * self.value_function[s_prime]
                        q_values.append(q_value)

                    pessimistic_v = self.p * max(q_values) + (1 - self.p) * min(q_values)
                    delta = max(delta, abs(new_values[x,y] - pessimistic_v))
                    new_values[x,y] = pessimistic_v

            self.value_function = new_values
            if delta < self.threshold:
                #plt.imshow(self.value_function, cmap='viridis'); plt.colorbar(); plt.show()
                break

    def simulate_agent(self):
        trajectory = [self.start]
        current = self.start

        # Get q Values resulting from all possible action
        q_values_per_state = {}
        for x in range(self.size):
            for y in range(self.size):
                next_states = [self.get_next_state((x, y), a) for a in self.actions]
                q_values = []
                for s in next_states:
                    joint_survival_score, sink_survival, starve_survival  =  self.risk_function(self.rewards[s], self.get_step_cost(s), self.w)
                    q_value = joint_survival_score + self.gamma * self.value_function[s] #here value function is the result of prev. value iteration
                    q_values.append(q_value)
                
                q_values_per_state[(x, y)] = q_values  # Store Q-values x state

        # Simulate trajectory 
        while current != self.goal:
            q_values = q_values_per_state[current]

            if self.use_softmax:
                action_probs = self.softmax(q_values)  # Get softmax probabilities
                action_idx = np.random.choice(len(self.actions), p=action_probs)  # Sample action based on probs
            else:
                action_idx = np.argmax(q_values)

            action = self.actions[action_idx]
            next_state = self.get_next_state(current, action)

            trajectory.append(next_state)
            current = next_state  # Move to next state
            #print(f"goal: {self.goal}")
        return trajectory
    



    def log_likelihood(self, data, *params):
        """Computes negative log likelihood based on whether softmax is used."""

        if self.model == 'basic':
            self.beta = params[0]
            self.value_iteration()  
        elif self.model == 'weighted':
            self.beta = params[0]
            self.w = params[1]  # `w` is always last parameter
            self.value_iteration()  
        elif self.model == 'pessimistic':
            self.beta = params[0]
            self.p = params[1]
            self.value_iteration_pessimistic()  
        elif self.model == 'baseline':
            "print, no parameters"

        log_likelihood = 0
        simulated_trajectory = []
        q_values_per_state = {}

        # loop over state and action pairs (from data)
        for (state, action) in data:

            x, y = state 
            next_states = [self.get_next_state((x, y), a) for a in self.actions] #possible next states
            q_values = []

            #get q value of possible next states under this model 
            for s in next_states: 
                joint_survival_score, _, _  =  self.risk_function(self.rewards[s], self.get_step_cost(s), self.w)
                q_value = joint_survival_score + self.gamma * self.value_function[s] #value function from value iteration 
                q_values.append(q_value)
            q_values_per_state[(x, y)] = q_values

            action_probs = self.softmax(q_values)  
            action_idx = self.actions.index(action)

            #likelihood of observed action given model
            log_likelihood += np.log(action_probs[action_idx] + 1e-9)  # Add small value for stability/ avoid log0
            simulated_trajectory.append((state, self.actions[np.argmax(action_probs)]))

        self.simulatedTrajectory = simulated_trajectory
        return -log_likelihood  




    def fit(self, data):
        """Fits parameters by maximizing log likelihood."""
        if self.model == 'basic':
            bounds = [(0, 40)]  # Bounds only for B
            x0 = [0.1]  # Initial value
        elif self.model == 'weighted':
            bounds = [(1e-3, 40), (1e-3, 1)]  # Bounds for tau/B and w
            x0 = [5, 0.5]  # Initial values
        elif self.model == 'pessimistic':
            bounds = [(1e-3, 40), (1e-3, 1)]  # Bounds for tau/B and p
            x0 = [5, 0.5]  # Initial values
        elif self.model == 'baseline':
            print("no parameters to fit")


        def objective(params):
            likelihood = self.log_likelihood(data, *params)
            return likelihood

        result = opt.minimize(objective, x0=x0, bounds=bounds)
        
        if self.model == 'basic':
            self.beta = result.x[0]
        elif self.model == 'weighted':
            self.beta, self.w = result.x 
        elif self.model == 'pessimistic':
            self.beta, self.p = result.x 

        #save params and like.
        best_params = result.x
        neg_ll = self.log_likelihood(data, *best_params)

        k = self.k #n params
        n =  len(data)
        ll = -neg_ll  # convert negative log-likelihood back to log-likelihood
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return result.x, neg_ll, aic, bic
    
    
    
    ##### here for greedy 
    def simulate_greedy_to_goal(self):
        """Greedy policy that always moves toward the goal using Manhattan distance."""
        trajectory = [self.start]
        current = self.start

        while current != self.goal:
            min_distance = float('inf')
            best_next_state = current

            for action in self.actions:
                next_state = self.get_next_state(current, action)
                # distance
                dist = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
                if dist < min_distance:
                    min_distance = dist
                    best_next_state = next_state

            current = best_next_state
            trajectory.append(current)

        #placeholder for plotting 
        self.survive_not_sink = np.ones((self.size, self.size))
        self.survive_not_starve = np.ones((self.size, self.size))
        self.joint_survival_score = np.ones((self.size, self.size))

        return trajectory
    
    def simulate_heuristic(self):
        """Heuristic policy that moves toward the goal while avoiding the storm, with a rule distance."""
        
        # Find the storm location
        sx, sy = np.unravel_index(np.argmin(self.rewards), self.rewards.shape)
        self.storm = (sx, sy)

        current = self.start
        trajectory = [current]

        while current != self.goal:
            x, y = current
            gx, gy = self.goal

            costs = []
            next_states = []

            for a in self.actions:
                next_state = self.get_next_state(current, a)
                nx, ny = next_state

                # Manhattan distance to goal and storm
                dist_to_goal = abs(nx - gx) + abs(ny - gy)
                dist_to_storm = abs(nx - sx) + abs(ny - sy)

                # Smooth penalty for being too close/far from rule distance
                penalty = 1000 / (1 + np.exp(-(self.rule - dist_to_storm) * 5))

                # Total cost
                cost = dist_to_goal + penalty

                # Relax penalty if close to goal (otherswise rule > 2 will fail as goal is at min distance 2 from storm. )
                if dist_to_goal <= 3:
                    cost = dist_to_goal

                costs.append(cost)
                next_states.append(next_state)

            # Softmax over -costs
            costs = np.array(costs)
            exp_vals = np.exp(-self.beta * (costs - np.min(costs)))  # stabilize
            probs = exp_vals / np.sum(exp_vals)

            # Choose next action probabilistically
            chosen_idx = np.random.choice(len(self.actions), p=probs)
            current = next_states[chosen_idx]
            trajectory.append(current)

        # Optional: fill survival grids
        self.survive_not_sink = np.ones((self.size, self.size))
        self.survive_not_starve = np.ones((self.size, self.size))
        self.joint_survival_score = np.ones((self.size, self.size))

        return trajectory
    
        
    def log_likelihood_greedy(self, data):
        """
        Computes negative log likelihood for the greedy-to-goal baseline model.
        """

        log_likelihood = 0
        simulated_trajectory = []

        for (state, action) in data:
            x, y = state

            min_distance = float('inf')
            best_action = None # Select best greedy action based on Manhattan distance

            for a in self.actions:
                next_state = self.get_next_state((x, y), a)
                dist = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
                if dist < min_distance:
                    min_distance = dist
                    best_action = a

            #greedy
            action_probs = np.zeros(len(self.actions))
            best_idx = self.actions.index(best_action)
            action_probs[best_idx] = 1.0
            action_idx = self.actions.index(action)

            log_likelihood += np.log(action_probs[action_idx] + 1e-9)
            predicted_action = self.actions[np.argmax(action_probs)]
            simulated_trajectory.append((state, predicted_action))

        self.simulatedTrajectory = simulated_trajectory
        return -log_likelihood
    

    

    
    def log_likelihood_heuristic_float(self, data, rule, beta):
        """
        Compute *negative* log likelihood for the heuristic-to-goal model
        using a continuous (sigmoid) penalty and softmax policy.
        """
        self.rule = rule
        self.beta = beta
        log_likelihood = 0
        self.simulatedTrajectory = []

        # storm = lowest reward location
        mx, my = np.unravel_index(np.argmin(self.rewards), self.rewards.shape)
        self.storm = (mx, my)

        for (state, action) in data:
            x, y = state
            gx, gy = self.goal

            # cost per action
            costs = []
            for a in self.actions:
                next_state = self.get_next_state((x, y), a)

                dist_to_goal = abs(next_state[0] - gx) + abs(next_state[1] - gy)
                dist_to_storm = abs(next_state[0] - mx) + abs(next_state[1] - my)

                # smooth penalty (continuous)
                penalty = 1000 / (1 + np.exp(-(rule - dist_to_storm) * 5)) #5 = smoothness
                cost = dist_to_goal + penalty

                # Relax penalty if close to goal (otherswise rule > 4 will fail as goal is at min distance 2 from storm. )
                if dist_to_goal <= 3:
                    cost = dist_to_goal

                costs.append(cost)

            # softmax over -costs (prob of choosing each action)
            costs = np.array(costs)
            exp_vals = np.exp(-beta * (costs - np.min(costs)))  # stabilize
            probs = exp_vals / np.sum(exp_vals)

            action_idx = self.actions.index(action)
            log_likelihood += np.log(probs[action_idx] + 1e-9)

            predicted_action = self.actions[np.argmax(probs)]
            self.simulatedTrajectory.append((state, predicted_action))

        return -log_likelihood 
    
    def fit_greedy(self, data):
        """ For the 'baseline' model (greedy to goal)
        """
        neg_ll = self.log_likelihood_greedy(data)
        k = 0  # No parameters
        n = len(data)
        ll = -neg_ll
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return None, neg_ll, aic, bic



    def fit_heuristic_float(self, data):
        """ For the heuristic to goal
        """

        bounds = [(1e-3, 40), (1e-3, 15)]  # beta and rule bounds
        x0 = [5.0, 3]  # starting value

        def objective(params):
            return self.log_likelihood_heuristic_float(data, *params)

        result = opt.minimize(objective, x0=x0, bounds=bounds)
        best_params = result.x
        neg_ll = self.log_likelihood_heuristic_float(data, *best_params)
        n = len(data)
        k = 2  # two fitted parameters
        ll = -neg_ll

        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return (result.x,neg_ll,aic,bic)
    
    
    




##### Risk functions (Model A,B,C,D)

# Risk function for stable models (Option A & B)

#A
def joint_risk_unweighted(reward, step_cost, w):
    survivalfrom_sinking = 1 - abs(reward)
    survivalfrom_starving = min(1, max(0, 1 - abs(step_cost)))
    joint_survival_score = survivalfrom_sinking * survivalfrom_starving

    return joint_survival_score, survivalfrom_sinking, survivalfrom_starving

#B
def joint_risk_weighted(reward, step_cost, w):
    risk_sinking = abs(reward) * w
    survivalfrom_sinking = 1 - risk_sinking

    opp_w = 1 - w
    survivalfrom_starving = min(1, max(0, 1 - abs(step_cost * opp_w)))
    joint_survival_score = survivalfrom_sinking * survivalfrom_starving
    return joint_survival_score, survivalfrom_sinking, survivalfrom_starving

#C
def dynamic_risk_unweighted(reward,step_togoal, step, step_cost,w):
    risk_sinking = 1- abs(reward)
    accum_starve = abs(step*step_cost)
    joint_starve = accum_starve + abs(step_togoal)
    joint_starve = max(0, min(joint_starve, 1))
    risk_starving = 1 - joint_starve
    joint_survival_score = risk_sinking* risk_starving
    return joint_survival_score, risk_sinking, risk_starving

#D
def dynamic_risk_weighted(reward,step_togoal, step, step_cost, w):
    risk_sinking = abs(reward)*w
    survivalfrom_sinking = 1- risk_sinking
    accum_starve = abs(step*step_cost)
    joint_starve = accum_starve + abs(step_togoal)
    opp_w = 1 - w
    risk_starving = max(0, min(joint_starve*opp_w, 1))
    survivalfrom_starving = 1 - risk_starving
    joint_survival_score = survivalfrom_sinking* survivalfrom_starving
    return joint_survival_score, survivalfrom_sinking, survivalfrom_starving


#Pessimistic
def risk_pessimistic(reward, step_cost, w):
    survivalfrom_sinking = reward
    survivalfrom_starving = 1 #min(1, max(0, 1 - abs(step_cost)))
    joint_survival_score = reward# survivalfrom_sinking * survivalfrom_starving
    #print(joint_survival_score, survivalfrom_sinking, survivalfrom_starving)
    return joint_survival_score, survivalfrom_sinking, survivalfrom_starving

def risk_pessimistic_costly(reward, step_cost, w):
    survivalfrom_sinking = reward
    survivalfrom_starving = min(1, max(0, 1 - abs(step_cost)))
    joint_survival_score = survivalfrom_sinking * survivalfrom_starving
    #print(joint_survival_score, survivalfrom_sinking, survivalfrom_starving)
    return joint_survival_score, survivalfrom_sinking, survivalfrom_starving
