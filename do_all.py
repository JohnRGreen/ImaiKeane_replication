# Runtime ~1300 seconds
import sys
sys.path.append('./auxcode')

# %% 1. Setup
print('1. Solve the model and display its policy functions')
# Import libraries
import os
# Set working directory to be this folder
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the working directory to that folder
os.chdir(script_dir)
# Verify the change
print("Working directory set to:", os.getcwd())
import numpy as np
import pandas as pd
# import scikit-learn
# import matplotlib
from scipy import stats
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interpn
import time
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

# Hyper-parameters
nM = 100 # grid points for market resources
nK = 100 # grid points for human capital
nq = 6 # points for the quadrature integration
T = 66 # retirement age
t0 = 20 # starting age
lifespan = T-t0 # length of working life
cmin = 1e-8 # minimum consumption
R = 1.04 # risk free interest rate

# pull in starting points
startingHCmean = pd.read_csv('./data/startingHCmean97.csv')
startingHCsd = pd.read_csv('./data/startingHCsd.csv')
startingAmean = pd.read_csv('./data/startingAmean97.csv')
startingAsd = pd.read_csv('./data/startingAsd.csv')
start = pd.concat([startingHCmean, startingHCsd, startingAmean, startingAsd], axis=1)
column_names = ['HCmean', 'HCsd', 'Amean', 'Asd']
start.columns = column_names

# Set up the grid
# Read in the function
with open("./auxcode/getGrid.py") as getgrid:
    exec(getgrid.read())

# Grid for market resources
Mmin = 50*np.ones(lifespan)
Mmax = np.linspace(1e5, 5e5, lifespan)
M_grid = np.ascontiguousarray(GetGrid(Mmin, Mmax, nM, "power",2))*1e-5
M_grid = np.vstack(((2*cmin)*np.ones((1, M_grid.shape[1])), M_grid))
M_grid = np.vstack((M_grid, 200*np.ones((1, M_grid.shape[1]))))

# Grid for human capital
# minimum wage of 4
kmin = np.repeat([4],lifespan)
# maximum level of human capital
kmax = np.linspace(25, 70, lifespan)
# Set grid and scale to thousands of dollars
k_grid = np.ascontiguousarray(GetGrid(kmin, kmax, nK, "power",1.5))/1e3
minwage= 0.00001
k_grid = np.vstack((minwage * np.ones((1, k_grid.shape[1])), k_grid))

# Auxiliary functions
# For discretizing distribution:
with open("./auxcode/discretize_log_distribution.py") as discretize:
    exec(discretize.read())
# Function to numerically get optimal h at credit constraint
with open("./auxcode/getH.py") as getH:
    exec(getH.read())
# Read in the function
with open("./auxcode/solveproblem.py") as solveproblem:
    exec(solveproblem.read())
with open("./auxcode/simulate.py") as simulate:
    exec(simulate.read())

# %% 2. Set the parameter vector and solve the agent's problem with sigma = 0 and G = 0
print('2. Solving and simulating with no uncertainty and no wage growth.')
params = {
# Utility function parameters
"phi"           : 3,
"beta"          : 0.98,
"gamma"         : 0.75,
"sigma"         : 0.00,
"alpha"         : .02,
"eta"           : 1.25,
"growth"        : np.linspace(0.00, 0.00, 45)
}

# Call the function
out1 = solveProblem(M_grid,k_grid,**params)
policyC1 = out1[0]
policyH1 = out1[1]
endk1    = out1[2]
policyM1 = out1[3]

# Simulate
params_sim = {
    "numSims"       : 1e4,         # Simulate 10,000 agents
    "simlifespan"   : 65,     # Lifespan of 65 years 
    "startage"      : 20,        # starting point
    "simseed"       : 1032024,
    "A_mean"        : start.Amean[1],
    "V_A"           : start.Asd[1],
    'HCt0_mean'     : start.HCmean[1],
    'HCt0_sd'       : start.HCsd[1],
    'sigma'         : params['sigma'],
    'growth'        : params['growth'],
}

# simulate
simdf1 = simulate(policyC1, policyH1, endk1, policyM1, params_sim)
mean_a_1 = simdf1.groupby('age')['a'].mean() # savings
mean_c_1 = simdf1.groupby('age')['c'].mean() # consumption
mean_h_1 = simdf1.groupby('age')['h'].mean() # hours of work

# Save the problem solution
# Define the directory path
directory = "./results/output1"
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
# save results
np.save("./results/output1/policyC.npy", policyC1)
np.save("./results/output1/policyH.npy", policyH1)
np.save("./results/output1/endk.npy", endk1)
np.save("./results/output1/policyM.npy", policyM1)
simdf1.to_csv('./results/output1/simdf.csv', index=False)

# %% 3. Set the parameter vector and solve the agent's problem with sigma > 0 and G = 0
print('3. Solving and simulating with uncertainty and no wage growth.')
params["sigma"] = 0.05
params_sim["sigma"] = 0.05

# Call the function
# Solve the problem
out2 = solveProblem(M_grid, k_grid, **params)
policyC2 = out2[0]
policyH2 = out2[1]
endk2    = out2[2]
policyM2 = out2[3]

# simulate
simdf2 = simulate(policyC2, policyH2, endk2, policyM2, params_sim)
mean_a_2 = simdf2.groupby('age')['a'].mean() # savings
mean_c_2 = simdf2.groupby('age')['c'].mean() # consumption
mean_h_2 = simdf2.groupby('age')['h'].mean() # hours of work

# Save the problem solution
# Define the directory path
directory = "./results/output2"
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
# save results
np.save("./results/output2/policyC.npy", policyC2)
np.save("./results/output2/policyH.npy", policyH2)
np.save("./results/output2/endk.npy", endk2)
np.save("./results/output2/policyM.npy", policyM2)
simdf2.to_csv('./results/output2/simdf.csv', index=False)

# %% 4. Set the parameter vector and solve the agent's problem with sigma > 0 and G > 0
print('4. Solving and simulating with uncertainty and wage growth.')
params["growth"] = np.linspace(0.03,0.01,45)
params_sim["growth"] = np.linspace(0.03,0.01,45)

# Call the function
# Solve the problem
out3 = solveProblem(M_grid, k_grid, **params)
policyC3 = out3[0]
policyH3 = out3[1]
endk3    = out3[2]
policyM3 = out3[3]

# simulate
simdf3 = simulate(policyC3, policyH3, endk3, policyM3, params_sim)
mean_a_3 = simdf3.groupby('age')['a'].mean() # savings
mean_c_3 = simdf3.groupby('age')['c'].mean() # consumption
mean_h_3 = simdf3.groupby('age')['h'].mean() # hours of work

# Save the problem solution
# Define the directory path
directory = "./results/output3"
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
# save results
np.save("./results/output3/policyC.npy", policyC3)
np.save("./results/output3/policyH.npy", policyH3)
np.save("./results/output3/endk.npy", endk3)
np.save("./results/output3/policyM.npy", policyM3)
simdf3.to_csv('./results/output3/simdf.csv', index=False)


# %% 5. Present more detailed figures on discrepancies for the last periods of
# life.
print('5. Make and save graphs.')
# Visuals
plt.plot(mean_a_1.index, mean_a_1.values, label='Assets, $\sigma = 0$ and $G=0$', color='red')
plt.plot(mean_a_2.index, mean_a_2.values, label='Assets, $\sigma = 0.05$ and $G=0$', color='blue')
plt.plot(mean_a_3.index, mean_a_3.values, label='Assets, $\sigma = 0.05$ and $G>0$', color='green')
plt.plot(mean_c_1.index, mean_c_1.values, label='Consumption, $\sigma = 0$ and $G=0$', color='red', linestyle='--')
plt.plot(mean_c_2.index, mean_c_2.values, label='Consumption, $\sigma = 0.05$ and $G=0$', color='blue', linestyle='--')
plt.plot(mean_c_3.index, mean_c_3.values, label='Consumption, $\sigma = 0.05$ and $G>0$', color='green', linestyle='--')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('$100,000')
plt.title('Assets and consumption across Age')
plt.legend()
# Save the plot
plt.savefig('./results/figs/assets_consumption_across_age.png')
# Show the plot
plt.show()

# graph labor supply
plt.plot(mean_h_1.index, mean_h_1.values, label='Labor supply, $\sigma = 0$ and $G=0$', color='red')
plt.plot(mean_h_2.index, mean_h_2.values, label='Labor supply, $\sigma = 0.05$ and $G=0$', color='blue')
plt.plot(mean_h_3.index, mean_h_3.values, label='Labor supply, $\sigma = 0.05$ and $G>0$', color='green')
# Add labels and title
plt.xlabel('Age')
plt.ylabel('100 hours')
plt.title('Labor supply across Age')
plt.legend()
# Save the plot
plt.savefig('./results/figs/laborsupply_across_age.png')

# Show the plot
plt.show()