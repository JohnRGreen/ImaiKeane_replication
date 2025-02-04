# Imai and Keane (2004) replication

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JohnRGreen/ImaiKeane_replication/HEAD)

This repository will replicate the lifecycle savings and labor supply of [Imai and Keane (2004)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-2354.2004.00138.x) in Python, contributing to the [REMARK](https://github.com/econ-ark/REMARK) project from [Econ-ARK](https://github.com/econ-ark). The main product of interest is a Jupyter Notebook  which reviews that paper and introduces the machinery behind the replication.

The repository contains the following components:
1. A **binder** folder containing **environment.yml** which specifies the environment for the code and installs dependencies from **requirements.txt**
2. A Jupyter Notebook **Imai_and_Keane_2004** which contains summarizes the original paper and then explores the reproduction
3. An **aux** folder which contains auxiliary code called in the notebook but not of primary interest to the reader
4. A **reproduce.sh** script that installs the requirements and runs and reproduces the results; it does so primarily by calling the Python script **do_all**

This project is still in progress and will be updated with further results.
