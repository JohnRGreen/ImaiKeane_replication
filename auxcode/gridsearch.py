
from itertools import product
import random

fixed_params = [1.5, 3, None, None, .05, None, None]

# Define the parameter ranges for the grid search
param_ranges = [
    np.linspace(0.92, 0.99, 10),  # 3rd parameter
    np.linspace(0.25, 1.75, 10),    # 4th parameter
    np.linspace(0.005, 0.1, 10),  # 6th parameter
    np.linspace(1.1, 2.5, 10)     # 7th parameter
]

# Create all combinations of parameters
param_combinations = list(product(*param_ranges))
random.shuffle(param_combinations)

# Initialize lists to store results
results = []

# Perform grid search
for combo in param_combinations:
    print(combo)
    # Create the full parameter vector
    params = fixed_params.copy()
    params[2], params[3], params[5], params[6] = combo
    
    # Execute the function
    mean_a, mean_c, mean_h, mean_k, mean_y, crit = getmeanpath_step6(params, moments, M_grid, k_grid, params_sim, start)
    
    # Store the results
    results.append({
        'params': params,
        'mean_a': mean_a,
        'mean_c': mean_c,
        'mean_h': mean_h,
        'mean_k': mean_k,
        'mean_y': mean_y,
        'crit': crit
    })
