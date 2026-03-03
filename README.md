# Imai and Keane (2004) replication

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JohnRGreen/ImaiKeane_replication/HEAD)

## Overview

This repository aims to replicate the lifecycle savings and labor supply model presented in [Imai and Keane (2004)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-2354.2004.00138.x). The replication is done in Python. 

This work represents a direct contribution to the [REMARK](https://github.com/econ-ark/REMARK) project led by the open-source community at [Econ-ARK](https://github.com/econ-ark). The primary objective of REMARK is to promote reproducibility and transparency across computational economics.

The central piece of this repository is an educational and interactive Jupyter Notebook. This notebook reviews the insights of the original paper, breaking down the model and introducing the computational machinery and algorithms necessary for a complete replication.

## Paper Summary and Objectives

Imai and Keane (2004) studies the dynamics of lifecycle savings and labor supply decisions when wages are endogenous. At its core, the study highlights how the accumulation of human capital may explain the divergence between micro- and macro-estimates of the intertemporal elasticity of subsitution.

The authors established that when the future return to human capital is appropriately accounted for in the modeling, the elasticity of intertemporal substitution in labor supply emerges significantly higher than estimated by preceding, less nuanced models.

By shifting this paradigm into Python, we ensure that the model behaves transparently and is accessible to researchers, academics, and students alike within the broader economics community.

An hour of work today rewards the agent with income, but it also increases utility through the rest of the lifecycle by increasing human capital and thus leading to higher earnings throughout the lifecycle. Because of this dynamic, at younger ages the true return to work may be much larger than the observed wage. When agents are older, their wages are higher but the returns to increased human capital are lower because they have fewer future years of work. The net result is a reasonably flat shadow wage which captures the net returns to work.

If the econometrician focuses solely on the variation between hourly wage and hours of work, the reasonably flat lifetime profile of hours and reasonably steep profile of wages will erroneously suggest a weak relationship between the two. Accounting for the full returns to work inclusive of human capital accumulation leads to a much higher responsiveness and leads to the higher i.e.s. that Imai and Keane find.

## The Model

The utility function modeled has agents receive utility from consumption, given by a CRRA specification. In order to discourage agents from borrowing when young to smooth consumption, the authors introduce age effects into utility. They do so with a linear spline $A(s_t)$ where $s_t$ is the agent's age at time period $t$. $A(s_t) = C_0 C_1$ at age 20, gradually moves to $C_0 C_2$ at age 25, and ends at $C_0$ at age 33, whree it remains for the rest of the lifecycle.

Wages are determined by human capital in a perfectly competitive labor market. Workers receive a rental rate for their human capital. Because the market is competitive and human capital is homogeneous, the wage rate is directly tied to the level of human capital an agent possesses.

A complication emerges here because I solve the problem with *no borrowing* so that $M_t>0$ always. At some grid points in $T$, it may be that the agent must have had $M_{T-1} < 0$ and thus the EGM step will result in negative market resources. We need to handle this carefully, because we cannot perform EGM at points where the agent was not strictly following the FOCs of the problem.

First, I loop through the points of $n_K$ to find the level of $M_{T-1}$ at which an agent with human capital $K_{T-1,j}$ would have $c_{T-1} >= M_{T-1}$ (eg the point at which they become credit constrained). When the EGM results in some levels of $M_{T-1}$ associated with human capital grid point $j$ which are negative, I isolate the last negative grid point $M_{T-1}^{- j}$ and the first positive gridpoint $M_{T-1}^{+ j}$. I perform a linear interpolation to find the point between these two points at which the agent consumes $c_{T-1,j} = M_{T-1}^{0 j}$.

## Repository Architecture

This codebase is systematically organized to isolate specific concerns and enable easy navigation:

*   **`binder/`**: The directory dedicated to the interactive Binder environment configuration. It houses `environment.yml` for broad Python environment definitions and `requirements.txt` identifying specific dependency versions.
*   **`data/`**: The folder containing pre-processed datasets or configuration files specifically formatted to feed the replication processes.
*   **`auxcode/`**: A supplementary folder comprising underlying helper functions, solvers, and utilities. This code is imported and leveraged by the main notebook, keeping the educational notebook clean and readable.
*   **`results/`**: A folder intended to catch the generated outputs, tables, and serialized representations arising from the completed replication runs.
*   **`Imai_and_Keane_2004.ipynb`**: The flagship Jupyter Notebook. It acts as both a written review of the paper and a step-by-step interactive demonstration of the reproduction code.
*   **`reproduce.sh`**: A top-level bash script functioning as the primary entry point for executing the entire replication automatically. 
*   **`do_all.py`**: A centralized Python orchestration script that executes the computational load when triggered by `reproduce.sh`.
*   **`myst.yml`**: A configuration file enabling seamless integration with the MyST Markdown system for enhanced documentation rendering.

## Quickstart and Execution Guide

### Reproducing with Docker (Recommended)

To run the replication completely isolated from your host system using Docker, ensuring that environmental factors do not interfere with the results:

1. Clone the project locally: `git clone https://github.com/JohnRGreen/ImaiKeane_replication.git`
2. Navigate into the folder: `cd ImaiKeane_replication`
3. Process the `Dockerfile` to create an image: `docker build -t imai_keane_container .`
4. Deploy the replication via a disposable container shell: `docker run --rm imai_keane_container`
5. The output should be successfully captured the `results` directory. 

### Executing Natively

If you prefer executing the code directly on your local workstation without Docker abstraction:

1. Confirm that Python 3.10 or a slightly newer compatible version is cleanly installed in your path.
2. Initialize a local virtual environment to avoid polluting host packages.
3. Install dependencies from the `binder` specification: `pip install -r binder/requirements.txt`
4. Trigger the end-to-end execution workflow by simply invoking: `./reproduce.sh`
5. Alternatively, run the python file directly: `python do_all.py`

## Current Progress and Next Steps

This repository is currently tagged as a work in progress. While the foundational scaffolding, environments, and basic solvers exist, we expect continuous iterations moving forward. The logic will be frequently updated with improved algorithms, enriched methodologies, and finalized replication matrices. We are actively aiming to provide full parity with the outputs highlighted in the original Imai and Keane publication.

## Contributor Guidelines

We aim to keep this repository well-documented. If you'd like to help test the accuracy of the structural economic models, debug dependencies, or port the legacy code, please submit a pull request against the active branch. Ensure `reproduce.sh` works seamlessly before marking your pull request as ready for review.

### Coding Approach

The repository follows a test-driven development cycle locally to prevent regressions. All major equations and model transformations are coded modularly to permit isolation tracing.

### Reporting Issues

When reporting an issue, clearly list out the parameters, dataset utilized, and python runtime environment so we can smoothly troubleshoot the unexpected behaviour.

## License and Terms of Use

This project and entirely of its corresponding code is distributed openly under the permissive MIT License. Please consult the supplementary `LICENSE` file found at the root of the repository for detailed warranty disclaimers and specific attribution terms.
