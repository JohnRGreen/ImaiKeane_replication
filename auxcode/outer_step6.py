# Next step: add stochastic wage shocks 
# Also going to adjust so that individuals work in the last period
# The solution to the problem does not change, but now I need to solve the problem on an income grid

# set working directory
import os
cwd = os.getcwd()
print("Current working directory:", cwd)
os.chdir('C:/Users/johng/Dropbox/SYP/Sep24/step6b')
# os.chdir('C:/Users/bobne/Dropbox/SYP/Sep24/step6b')

# Clear the environment
import gc
# Clear all user-defined variables
vars_to_remove = [var for var in globals() if not var.startswith("__")]
for var in vars_to_remove:
    del globals()[var]

# libraries:
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import estimagic as em
import scipy as scipy
# from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interpn
import time
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

# global variables:
# Grid points:
nM = 100 # assets
nK = 100 # human capital
nq = 6 # quadrature integration
# lifespan:
T = 66
t0 = 20
lifespan = T-t0
# minimum consumption
cmin = 1e-8
# interest rate
R = 1.04
# wage growth rate
growth = 0.02
growth=0.01

# moments
moments = [ 0.057,
            0.108,
            0.147,
            0.173,
            2.762708143,
            17.2,
            21.1,
            22.6,
            23,
            ]

# Set exogeneous grid for assets
with open("../getGrid.py") as getgrid:
    exec(getgrid.read())

# Set exogeneous grid for market resources
Mmin = 10*np.ones(lifespan)
Mmax = np.linspace(1e5, 5e5, lifespan)
M_grid = np.ascontiguousarray(GetGrid(Mmin, Mmax, nM, "power",2))*1e-5
M_grid = np.vstack(((2*cmin)*np.ones((1, M_grid.shape[1])), M_grid))
M_grid = np.vstack((M_grid, 200*np.ones((1, M_grid.shape[1]))))

# Then set a grid for wages
# minimum wage of 4
kmin = np.repeat([4],lifespan)
# simple static max level of human capital (income does not grow)
kmax = np.linspace(25, 80, lifespan)
# define the grid
# log_k_grid = np.ascontiguousarray(GetGrid(kmin,kmax,nK,"log",0))/1e1
# k_grid = np.ascontiguousarray(GetGrid(kmin,kmax,nK,"linear"))/1e1
# Set grid and scale to thousands of dollars
k_grid = np.ascontiguousarray(GetGrid(kmin, kmax, nK, "power",1.5))/1e3
minwage = 0.003
minwage= 0.00001
k_grid = np.vstack((minwage * np.ones((1, k_grid.shape[1])), k_grid))

# starting points:
# pull in starting points
startingHCmean = pd.read_csv('../../data_extracts/startingHCmean97.csv')
startingHCsd = pd.read_csv('../../data_extracts/startingHCsd.csv')
startingAmean = pd.read_csv('../../data_extracts/startingAmean97.csv')
startingAsd = pd.read_csv('../../data_extracts/startingAsd.csv')
start = pd.concat([startingHCmean, startingHCsd, startingAmean, startingAsd], axis=1)
# set the names:
column_names = ['HCmean', 'HCsd', 'Amean', 'Asd']
start.columns = column_names

# paramaters for simulation
params_sim = {
    "numSims": 1e4, 
    "simlifespan" : 65,    
    "startage" : 20,
}

with open("solveproblem_step6c.py") as solveproblem:
    exec(solveproblem.read())
with open("simulate_step6.py") as sim:
    exec(sim.read())
with open("getSimMoments_step6.py") as getSimMoments:
    exec(getSimMoments.read())
with open("getCrit_step6.py") as getCrit:
    exec(getCrit.read())
with open("../functions.py") as functions:
    exec(functions.read())
with open("getmeanpath_step6.py") as getmeanpath_step6:
    exec(getmeanpath_step6.read())
with open("../discretize_log_distribution.py") as discretize_log_distribution:
    exec(discretize_log_distribution.read())
with open("getH.py") as getH:
    exec(getH.read())
# exec(open("getH.py").read())

#### Check logic ###################################################
sdshock = .05   
mu = -(1/2)*sdshock**2 # mean of wage shock
discretize_log_distribution(nq, mu, sdshock)
growth = np.linspace(0.03, 0.01, 45)

# Not really able to estimate the model: just want to check results are sensible
# sdshock = 1e-5
# growth = np.linspace(0.0, 0.0, 45)
params1 =  [1.5, # curvature of terminal value
             3,   # coef for terminal value
             1/R,
             0.75,
             sdshock,  # variance of wage shock
             .005, # coef for LS disutility
             1.5, #
            ]
means1 = getmeanpath_step6(params1, moments, M_grid, k_grid, params_sim, start)
mean_a1 = means1[0]
mean_c1 = means1[1]
mean_h1 = means1[2]

params2 =  [1.5, # curvature of terminal value
             3,   # coef for terminal value
             1/R,
             0.75,
             sdshock,  # variance of wage shock
             .01, # coef for LS disutility
             1.5, #
            ]
# Check first point
means2 = getmeanpath_step6(params2, moments, M_grid, k_grid, params_sim, start)
mean_a2 = means2[0]
mean_c2 = means2[1]
mean_h2 = means2[2]

# Then decrease patience a bit
params3 =   [1.5, # curvature of terminal value
             3,   # coef for terminal value
             1/R,
             0.75,
             sdshock,  # variance of wage shock
             .015, # coef for LS disutility
             1.5, #
            ]
# Check first point
means3 = getmeanpath_step6(params3, moments, M_grid, k_grid, params_sim, start)
mean_a3 = means3[0]
mean_c3 = means3[1]
mean_h3 = means3[2]

growth = np.linspace(0.03, 0.01, 45)
# Then give more terminal value
params4 =  [1.5, # curvature of terminal value
             3,   # coef for terminal value
             1/R,
             0.75,
             sdshock,  # variance of wage shock
             .02, # coef for LS disutility
             1.5, #
            ]
# Check first point
means4 = getmeanpath_step6(params4, moments, M_grid, k_grid, params_sim, start)
mean_a4 = means4[0]
mean_c4 = means4[1]
mean_h4 = means4[2]

# plot mean assets
plt.plot(mean_a1.index, mean_a1.values, label='Baseline', color='blue')
plt.plot(mean_a2.index, mean_a2.values, label='Increase patience', color='green')
plt.plot(mean_a3.index, mean_a3.values, label='Decrease patience', color='yellow')
plt.plot(mean_a4.index, mean_a4.values, label='Terminal value', color='red')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('Average Assets')
plt.title('Average Assets Across Age')
plt.legend()
# Show the plot
plt.show()

# mean consumption
plt.plot(mean_c1.index, mean_c1.values, label='Baseline', color='blue')
plt.plot(mean_c2.index, mean_c2.values, label='Increase patience', color='green')
plt.plot(mean_c3.index, mean_c3.values, label='Decrease patience', color='black')
plt.plot(mean_c4.index, mean_c4.values, label='Terminal value', color='red')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('Average Consumption')
plt.title('Average Consumption Across Age')
plt.legend()
# Show the plot
plt.show()

# mean labor supply
plt.plot(mean_h1.index, mean_h1.values, label='Baseline', color='blue')
plt.plot(mean_h2.index, mean_h2.values, label='Increase patience', color='green')
plt.plot(mean_h3.index, mean_h3.values, label='Decrease patience', color='black')
plt.plot(mean_h4.index, mean_h4.values, label='Terminal value', color='red')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('Average labor supply')
plt.title('Labor Supply Across Age')
plt.legend()
# Show the plot
plt.show()
#### end check logic ###################################################

#### OPTIMIZE ##################################################
params0vec = [1.5, # curvature of terminal value
             3,   # coef for terminal value
             1/R,
             1.5,
             .05,  # variance of wage shock
             .05, # coef for LS disutility
             1.25, #
            ]
params0vec = [1.5, # curvature of terminal value
             3,   # coef for terminal value
             0.98576334,
             1.45190277,
             .05,  # variance of wage shock
             .05396774, # coef for LS disutility
             1.15355682, #
            ]
# check the criterion:
crit = getCrit_step6(params0vec, moments, M_grid, k_grid, params_sim, start)

# now set additional arguments:
additional_arguments = {
    'moments': moments,
    'M_grid': M_grid,
    'k_grid': k_grid,
    'params_sim': params_sim,
    'start': start,
    }

# (Not using -- model really can't be calibrated to anything)
# set method
# met = "scipy_lbfgsb"
met = "scipy_neldermead"
algo_options = {
    "stopping.max_criterion_evaluations": 5e2,
    # "stopping.max_criterion_evaluations": 2,
}

# fix any parameters?
const = [
        {"loc": [0], "type": "fixed"}, # fixing terminal period gamma
        {"loc": [1], "type": "fixed"}, # fixing terminal period coef 
        # {"loc": [2], "type": "fixed"}, # fixing terminal period coef 
        # {"loc": [3], "type": "fixed"}, # fixing terminal period coef 
        {"loc": [4], "type": "fixed"}, # fixing the SD of shocks
        ]

# bounds:
lower_bounds=np.array([0, 1e-3, 0.9,  1e-1,1e-5,1e-2,1.01])
upper_bounds=np.array([5, 30,   0.99, 3,   1,   1,   3]) 

res = em.minimize(
    # criterion=getCrit_step6,
    criterion=getCrit_step6,
    params=np.asarray(params0vec),
    criterion_kwargs=additional_arguments,
    algorithm=met,
    lower_bounds = lower_bounds,
    upper_bounds=upper_bounds,
    constraints=const,
    error_handling="continue",
    algo_options = algo_options,
)

# Check solution
getCrit_step6(res.params, moments, A_grid, params_sim, start)

means_res = getmeanpath_step6(res.params, moments, M_grid, k_grid, params_sim, start)
mean_a = means_res[0]
mean_c = means_res[1]
mean_h = means_res[2]

# plot mean assets
plt.plot(mean_a.index, mean_a.values, label='assets', color='blue')
plt.plot(mean_c.index, mean_c.values, label='consumption', color='green')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('$ value')
plt.title('Average Assets and Consumption by age')
plt.legend()
# Show the plot
plt.show()

# plot mean assets
plt.plot(mean_h.index, mean_h.values, label='consumption', color='green')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('$ value')
plt.title('Labor Supply by age')
plt.legend()
# Show the plot
plt.show()

#### END OPTIMIZE ##################################################

res.params
((-(0.99/3e-5))*(20/1e5)*1**(-1.89))**(1/(1.25-1))
((-(0.99/1e-5))*(20/1e5)*1**(-1.89))**(.5/(1.25-1))

# # Check first point
# means1 = getmeanpath_step6(params1, moments, A_grid, params_sim, start)
# mean_a1 = means1[0]
# mean_c1 = means1[1]

# # mean assets
# plt.plot(mean_a1.index, mean_a1.values, label='Params 1', color='blue')
# # Add labels and title
# plt.xlabel('Age')
# plt.ylabel('Average Assets')
# plt.title('Average Assets Across Age')
# # Show the plot
# plt.show()

# # mean consumption
# plt.plot(mean_c1.index, mean_c1.values, label='Params 1', color='blue')
# # Add labels and title
# plt.xlabel('Age')
# plt.ylabel('Average Consumption')
# plt.title('Average Consumption Across Age')
# # Show the plot
# plt.show()


# # crit = getCrit(params0vec, moments, A_grid, params_sim, start)

# # np.exp(res.params[0])
# # np.exp(res.params[1])
# # 1/(1+np.exp(-res.params[2]))
# # np.exp(res.params[3])

# # params0vec = res.params
# res_step6 = em.minimize(
#     criterion=getCrit_step6,
#     params=np.asarray(params0vec),
#     criterion_kwargs=additional_arguments_step6,
#     algorithm=met,
#     lower_bounds = lower_bounds,
#     upper_bounds=upper_bounds,
#     constraints=const_step6,
#     # error_handling="continue",
#     algo_options = algo_options,
# )


# params0vec
# start_time = time.time()
# crit = getCrit_step6(params0vec, moments, A_grid, k_grid, params_sim, start)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")


# crit = getCrit_step6(res_step6.params, moments, A_grid, k_grid, params_sim, start)
# crit = getCrit_step6([0.3629645,    2.72060731,   2.98224646,  -0.21560951, -33.], moments, A_grid, k_grid, params_sim, start)

# np.exp(res_step6.params[0])
# np.exp(res_step6.params[1])
# 1/(1+np.exp(-res_step6.params[2]))
# np.exp(res_step6.params[3])
