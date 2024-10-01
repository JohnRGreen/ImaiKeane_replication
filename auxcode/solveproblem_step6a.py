# Restarting this project from scratch in May, 2024.
def solveProblem_step6a(A_grid, k_grid, **params_sp):
    # params_sp = params_dict

    # get the parameters 
    # gamma_Tp1   = params_sp['gamma_Tp1']
    gamma_Tp1   = params_sp['gamma']
    beta_Tp1    = params_sp['beta_Tp1']
    beta        = params_sp['beta']
    gamma       = params_sp['gamma']
    alpha       = params_sp['alpha']
    eta         = params_sp['eta']

    # wage shocks:
    sigma       = params_sp['sigma']
    mu = -(1/2)*sigma**2 # mean of wage shock

    # dsicretize the shock distribution
    quad = discretize_log_distribution(nq, mu, sigma)

    #### Set up grids ####
    policyC          = np.full((nM+1,nK+1,lifespan),np.nan)
    policyH          = np.full((nM+1,nK+1,lifespan),np.nan)
    ECprime =     np.full((nM,nK,lifespan),np.nan)
    endH          = np.full((nM,nK,lifespan),np.nan)
    Ecprime       = np.full((nM,nK),np.nan)
    endM          = np.full((nM+1,lifespan),np.nan)
    endK          = np.full((nM+1,lifespan),np.nan)
    endC1          = np.full((nM,nK,lifespan-1),np.nan)
    endM1          = np.full((nM,nK,lifespan-1),np.nan)
    endH1          = np.full((nM,nK,lifespan-1),np.nan)
    nextC         = np.full((nM,nK,lifespan),np.nan)
    nextH         = np.full((nM,nK,lifespan),np.nan)
    value         = np.full((nM+1,nK+1,lifespan),np.nan)
    evalue         = np.full((nM+1,nK+1,lifespan),np.nan)
    nextM         = np.full((nM,lifespan),np.nan)
    picc              = np.full(lifespan, np.nan)
    #### end grid setup ####

    #### PERIOD T ####
    ixt = lifespan-1
    # In period T+1 we consume everything, use terminal period value function to get utility
    
    # Calculate terminal period value
    value[1:,1:,ixt] = (beta_Tp1/(1-gamma)) * ((R*A_grid[:,lifespan-1])[:, None])**(1-gamma)

    # back out consumption in period T:
    ixt = lifespan-2
    policyC[1:,1:,ixt] = ((R*beta_Tp1)**(-1/(gamma)))*(R*A_grid[:,ixt+1])[:, None]
    value[1:,1:,ixt] = (policyC[1:,1:,ixt]**(1-gamma))/(1-gamma) + beta*value[1:,1:,ixt+1] 

    # then work out what M_t must have been
    endM[1:,ixt] = A_grid[:,ixt] + policyC[1:,1,ixt]

    # get limiting consumption func
    lambdaconst = (beta_Tp1 * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst) / ( 1 + (R*lambdaconst))

    # set lower bound as k goes to 0
    endK[0,ixt] = 0
    endK[1:,ixt] = k_grid[:,ixt]
    policyC[:,0,ixt] = picc[ixt] * endM[:,ixt]
    value[:,0,ixt] = (policyC[:,0,ixt]**(1-gamma))/(1-gamma) 

    # set the lower bound as M goes to 0
    endM[0,ixt] = 0
    policyC[0,:,ixt] = cmin
    value[0,:,ixt] = (policyC[0,:,ixt]**(1-gamma))/(1-gamma) 
    evalue[:,:,ixt] = value[:,:,ixt]
    #### END PERIOD T ####

    #### period T-1 ####
    # In T-1, since we won't be working in T, human capital in T is not relevant
    ixt = lifespan - 3
    picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

    # Add a bad value point for M T 0
    minterp = endM[:,ixt+1]

    # limiting value at upper bound
    ub = np.max(value[:,:,ixt+1])

    # limit when wages go to 0
    kinterp = endK[:, ixt+1]

    for ixk in range(nK):
        for ixa in range(nM):
            # getPolicy([0.1,100],[A_grid[ixa,ixt],k_grid[ixk,ixt]],minterp,kinterp,evalue[:,:,ixt+1],ub)
            def objective(choice):
                return getPolicy(choice, [A_grid[ixa,ixt],k_grid[ixk,ixt]], minterp, kinterp, evalue[:,:,ixt+1], ub)
            bounds = [(cmin,A_grid[ixa,ixt]), (1, 600)]  # Example bounds for consumption and housing
            sol = minimize(objective, [A_grid[ixa,ixt]-.1,55], bounds=bounds,method='L-BFGS-B')
            policyC[ixa+1,ixk+1,ixt] = sol.x[0]
            policyH[ixa+1,ixk+1,ixt] = sol.x[0]
            value[ixa+1,ixk+1,ixt] = -sol.fun


    # endogeneous C
    endC[:,:,ixt] = (beta * R * ECprime[:,:,ixt+1])**(-1/(gamma))

    # endogeneous H
    prevK = np.tile(k_grid[:,ixt+1], (75, 1))/(1+growth)
    endH[:,:,ixt] = ((beta/(alpha*eta)) * (prevK/1e2)* ECprime[:,:,ixt+1])**(1/(eta-1))
    # ((beta/(alpha*eta)) * (prevK[0,ixk]/100)* ECprime[0,ixk,ixt+1])**(1/(eta-1))
    # diff = h - ((beta/(alpha*eta))*(k)*ecprime)**(1/(eta-1))

    # Then back out M
    endM1[:,:,ixt] = ((np.tile(endM[:,ixt+1],(75,1)).T - endH[:,:,ixt] * np.tile(k_grid[:,ixt+1], (75, 1))/(1+growth))/R) + endC[:,:,ixt]
    # mtest = np.tile(endM[:,ixt+1],(75,1)).T

    # Want to numerically find optimal H when M_{T-1} = 0
    minterp = np.insert(endM[:,ixt+1], 0, 0)
    minterp = np.insert(minterp, minterp.shape, 10*np.max(endM[:,ixt+1]))
    lower_bound = 1e-8
    upper_bound = 1e8
    
    for ixk in range(nK):
        ecinterp = np.insert(ECprime[:,ixk,ixt+1], 0, cmin**(-gamma))
        ecinterp = np.insert(ecinterp, ecinterp.shape, (picc[ixt+1]*np.max(minterp))**(-gamma))
        lower_bound = 1e-8
        upper_bound = 1e8
        solution = brentq(getH_cc, lower_bound, upper_bound, args=(k_grid[ixk,ixt+1]/(1+growth), minterp, ecinterp, beta, alpha, gamma, eta))


    nextM = np.tile(k_grid[:,ixt], (75, 1)) *(2040/1e2) +  np.tile(A_grid[:,ixt]*R,(75,1)).T

    # Now interpolate next period's C based on this period's M and K (which lead to next period M)
    minterp = np.insert(endM[:,ixt+1], 0, 0)
    cinterp = np.insert(endC[:,0,ixt+1], 0, 0)
    minterp = np.insert(minterp, minterp.shape, 10*np.max(nextM))
    cinterp = np.insert(cinterp, cinterp.shape, picc[ixt+1]*np.max(minterp))
    nextC[:,:,ixt] = np.interp(nextM, minterp, cinterp)
    # np.interp(minterp[20], minterp, cinterp) - cinterp[20]

    # now get this period's C
    endC1[:,:,ixt] = (beta*R)**(-1/(gamma)) * nextC[:,:,ixt]

    # back out what must have been this period's M
    endM1[:,:,ixt] = np.tile(A_grid[:,ixt],(75,1)).T + endC1[:,:,ixt]

    # Now let's add some structure to the grid
    mmin        = np.array([np.min(endM1[:,:,ixt])])
    # mmin        = np.array([Mmin[ixt]])
    mmax        = np.array([np.max(endM1[:,:,ixt])])
    endM[:,ixt] = GetGrid(mmin, mmax, nM, "power",2).reshape(1,-1)

    # set limiting consumption function for this period
    lambdaconst = (beta * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

    # Will interpolate on this period's solution for our new grid
    for ixk in range(nK):
        minterp = endM1[:,ixk,ixt]
        cinterp = endC1[:,ixk,ixt]
        # Set the lower bound
        minterp = np.insert(minterp, 0, 0)
        cinterp = np.insert(cinterp, 0, 0)
        # And the limiting function as m -> inf
        minterp = np.insert(minterp, minterp.shape, 10*np.max(minterp))
        cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*np.max(minterp))

        # interpolate:
        endC[:, ixk, ixt] = np.interp(endM[:,ixt], minterp, cinterp,-999,999)
    # We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM
    #### end period T-1 ####

    #### ITERATE BACKWARDS ####
    for ixt in range(lifespan-3, -1, -1):
        # ixt = lifespan-1-1-1

        # find next period's M based on our exogeneous savings and wage grids
        # row is savings, column is human capital
        nextM = np.tile(k_grid[:,ixt], (75, 1)) *(2040/1e2) +  np.tile(A_grid[:,ixt]*R,(75,1)).T
        nextM = np.repeat(nextM[:, :, np.newaxis], 6, axis=2)
        m_flat = nextM.ravel()
        possibleK = np.outer(k_grid[:,ixt]*(1+growth), quad)
        possibleK = np.repeat(possibleK[np.newaxis, :, :], nM, axis=0)
        k_flat = possibleK.ravel()

        # Clip k at the upper value -- this is just an approximation
        k_flat=np.clip(k_flat, None, np.max(k_grid[:,ixt+1]))

        # Now interpolate next period C
        # Set up limit points for next period's consumption as M -> 0 and M -> inf
        minterp = np.insert(endM[:,ixt+1], 0, 0)
        new_row = np.full((1, endC.shape[1]), 0)
        cinterp = np.vstack((new_row, endC[:,:,ixt+1]))

        # then add a point for the upper bound, which uses the limiting function
        minterp = np.insert(minterp, minterp.shape, 20*np.max(nextM))
        new_row = np.full((1, endC.shape[1]), picc[ixt+1] * 20*np.max(nextM))
        cinterp =  np.vstack((cinterp,new_row))

        # limit when wages go to 0
        kinterp = k_grid[:, ixt+1].flatten()
        # Now add a new column, where k = 0 and we only have consumption from our resources
        # calculate limiting consumption fraction
        picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))
        # then add in points
        kinterp = np.insert(kinterp, 0, 0)
        newcol =  picc[ixt] * minterp
        cinterp =  np.hstack((newcol.reshape(-1,1),cinterp))

        # interpolate
        endCtemp = interpn((minterp,kinterp), cinterp, (m_flat,k_flat))
        endC_reshaped = endCtemp.reshape(nextM.shape)
        Ecprime[:,:] = np.mean(MV(endC_reshaped,gamma),2)

        # 2. Get this period's optimal consumption:
        endC1[:,:,ixt] = (beta * R * Ecprime[:,:])**(-1 / gamma)
        # endCfull[:,22]
        # 3. Then back out this period's M_t:
        endM1[:,:,ixt] = A_grid[:,ixt][:,None] + endC1[:,:,ixt]  

        # Now let's add some structure to the grid
        mmin        = np.array([np.min(endM1[:,:,ixt])])
        # mmin        = np.array([Mmin[ixt]])
        mmax        = np.array([np.max(endM1[:,:,ixt])])
        endM[:,ixt] = GetGrid(mmin, mmax, nM, "power",2).reshape(1,-1)

        # Will interpolate on this period's solution for our new grid
        for ixk in range(nK):
            minterp = endM1[:,ixk,ixt]
            cinterp = endC1[:,ixk,ixt]
            # Set the lower bound
            minterp = np.insert(minterp, 0, 0)
            cinterp = np.insert(cinterp, 0, 0)
            # And the limiting function as m -> inf
            minterp = np.insert(minterp, minterp.shape, 10*np.max(minterp))
            cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*np.max(minterp))

            # interpolate:
            endC[:, ixk, ixt] = np.interp(endM[:,ixt], minterp, cinterp,-999,999)

    #### END BACKWARDS RECURSION ####

    # Now set up the grids to cover limits
    endM = np.vstack((np.zeros((1, endM.shape[1])), endM))
    # endC = np.pad(endC, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cmin)
    endC = np.pad(endC, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # and add value for the limiting function
    newrow = 20*endM[nM,:]
    endM = np.vstack((endM, newrow))
    # now add the new consumption
    crow = newrow * picc
    cmat = np.tile(crow, (nK, 1))
    endC = np.concatenate((endC, cmat[np.newaxis, :, :]), axis=0)
    
    # Lastly, add in points for when k = 0
    k0c = endM * picc
    # k0c[:,0] == endM[:,0] * picc[0]
    # k0c[:,40] == endM[:,40] * picc[40]
    endC = np.concatenate((k0c[:,np.newaxis,  :],endC), axis=1)

    return endM, endC