def getCrit_step6(params,moments,M_grid,k_grid,params_sim,start):
    print(params) 
    # testing:
    # params= params1
    # params=res.params

    # Make a params dictionary:
    params_dict = {
    # Utility function parameters
    "gamma_Tp1"   : params[3], # forcing terminal period curvature to match CRRA
    "beta_Tp1"    : params[1],
    "beta"        : params[2],
    "gamma"       : params[3],
    "sigma"       : params[4],
    "alpha"       : params[5],
    "eta"         : params[6],
    # "sigma"       : np.exp(params[4]),
    
    # starting points
    "A_mean"   : start.Amean[1],
    "V_A"      : start.Asd[1],
    'HCt0_mean': start.HCmean[1],
    'HCt0_sd'  : start.HCsd[1],
    }

    # solve the problem:
    # policy_out = solveProblem(A_grid,**params_dict)
    policy_out = solveProblem_step6(M_grid,k_grid,**params_dict)
    policyC = policy_out[0]
    policyH = policy_out[1]
    endk    = policy_out[2]
    policyM = policy_out[3]

    # Now we can simulate forward:
    # Add the new row at the top of k_grid
    simdf = simulate_step6(policyC, policyH, 562024, endk, policyM, params_dict, params_sim)

    simmoments = getSimMoments_step6(simdf)

    # now get the loss
    simmoments_np = np.array(simmoments)
    moments_np = np.array(moments)
    diff = (simmoments_np - moments_np)/moments_np    
    diff[4] = 0 
    crit = np.sum(diff * diff)

    print(crit)

    return crit