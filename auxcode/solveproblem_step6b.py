# Restarting this project from scratch in May, 2024.
def solveProblem_step6(M_grid, k_grid, **params_sp):
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
    policyC          = np.full((nM+2,nK+1,lifespan),np.nan)
    policyCprime          = np.full((nM+2,nK+1,lifespan),np.nan)
    ECprime =     np.full((nM+2,nK+1,lifespan),np.nan)
    policyH          = np.full((nM+2,nK+1,lifespan),np.nan)
    policyM           = np.full((nM+2,lifespan),np.nan)
    policyM[:,lifespan-1] = M_grid[:,lifespan-1]
    endC1          = np.full((nM,nK,lifespan-1),np.nan)
    endM1          = np.full((nM,nK,lifespan-1),np.nan)
    endH1          = np.full((nM,nK,lifespan-1),np.nan)
    picc              = np.full(lifespan, np.nan)
    endk            = np.full((nK+1,lifespan),np.nan)
    endk[0,:] =  k_grid[0,:]
    endk[1:,lifespan-1] = k_grid[1:,lifespan-1]
    egmmask = np.full((nM+2,nK+1,lifespan),np.nan)
    egmmask[:,0,:] = 0
    #### end grid setup ####

    #### PERIOD T ####
    ixt = lifespan-1
    # In period T+1 we consume everything, use terminal period value function to get utility
    
    # get consumption func
    lambdaconst = (beta_Tp1 * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst) / ( 1 + (R*lambdaconst))

    # exclude M = 0 point
    policyC[1:,:,ixt] = picc[ixt]*M_grid[1:,ixt][:, None]
    # set lower bound
    policyC[0,:,ixt]  = cmin 

    # H is just 0
    policyH[:,:,ixt] = 0

    # get marginal value grid
    ECprime[:,:,ixt] = (policyC[:,:,ixt])**(-gamma)
    #### END PERIOD T ####

    #### period T-1 ####
    # In T-1, since we won't be working in T, human capital in T is not relevant
    ixt = lifespan - 1 - 1

    # set limiting consumption function for this period
    lambdaconst = (beta * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

    # endogeneous C; avoid the limit points
    endC1[:,:,ixt] = (beta * R * ECprime[1:(nM+1),1:,ixt+1])**(-1/(gamma))

    # endogeneous H
    endk[1:,ixt] = endk[1:,ixt+1]/(1+growth)
    prevK = np.tile(endk[1:,ixt], (nK, 1))
    endH1[:,:,ixt] = ((beta/alpha) * prevK * ECprime[1:(nM+1),1:,ixt+1])**(1/(eta-1))
    # ((beta/(alpha*eta)) * (prevK[0,ixk]/100)* ECprime[0,ixk,ixt+1])**(1/(eta-1))
    # diff = h - ((beta/(alpha*eta))*(k)*ecprime)**(1/(eta-1))

    # Then back out M_{T-1}
    Mtemp = np.tile(M_grid[1:(nM+1),ixt+1],(nM,1)).T
    endM1[:,:,ixt] = (Mtemp - prevK*endH1[:,:,ixt])/R + endC1[:,:,ixt]

    # Now we have, for a fixed K_{T-1}, a combination of M_{T-1} and optimal c_{T-1} and h_{T-1}
    # Want to fit to our structured grid

    # get mask for positive M
    # Will need to be careful -- this will break if all M < 0
    negmmask = endM1[:,:,ixt] < 0
    consmask = endC1[:,:,ixt] > endM1[:,:,ixt]
    dropmask = ~(negmmask | consmask)
    # np.where(np.logical_and(consmask, np.logical_not(negmmask)))
    first_true_row = np.argmax(np.all(dropmask, axis=1))
    nonsoncstrained = np.where(dropmask, endM1[:,:,ixt], np.nan)
    mincolval = np.nanmin(nonsoncstrained, axis=0)
    colarg = np.argmax(dropmask, axis=0)
    colarg[np.isnan(mincolval)] = nK
    # testk = 80
    # min_values_per_column[testk]
    # endM1[np.min(np.where(dropmask[:,testk])),testk,ixt]

    # In period T-1, we know optimal solution when M = 0
    policyH[0,0,ixt]  = ( (beta/alpha) * (M_grid[0,ixt]**(1-gamma)) * picc[ixt+1]**(-gamma) )**(1/(eta-1+gamma))
    policyH[0,1:,ixt] = ( (beta/alpha) * (endk[1:,ixt]**(1-gamma)) * picc[ixt+1]**(-gamma) )**(1/(eta-1+gamma))
    policyC[0,:,ixt] = cmin

    # Now for fixed K given by endk, we have a clean interpolation on the optimal solution for each mincolval. starting at the max of mincolval
    # Going to set up new M values to interpolate on, starting at this max, so that we keep a clean optimal portion of the grid to do the EGM on (below is just linear for the constrained)
    gridmin = np.full(1,np.nanmin(mincolval, axis=0))
    gridmax = np.full(1,np.maximum(np.nanmax(nonsoncstrained),M_grid[nM,ixt]))
    # if M_grid[nM,ixt] - gridmin < 1:
    #     gridmax = np.nanmax(nonsoncstrained)
    # else:
    #     gridmax = M_grid[nM,ixt]

    policyM[1:(nM+1),ixt] = GetGrid(gridmin, gridmax, nM, "power",2).flatten()
    policyM[0,ixt] = 0
    policyM[nM+1,ixt] = 200

    # set up a mask for use in next period to show where we are using the true solution and where we are using the constrained solution
    # Find the index where each value in mincolval first exceeds policyM[:,ixt]
    for ixk in range(len(mincolval)):
        egmmask[:,ixk+1,ixt] = policyM[:,ixt] > mincolval[ixk]

    # Will interpolate on this period's solution for our new grid
    for ixk in range(nK):
        # drop constrained points
        # minterp = np.insert(endM1[first_true_row:,ixk,ixt],0,0)
        # cinterp = np.insert(endC1[first_true_row:,ixk,ixt],0,0)
        # hinterp = np.insert(endH1[first_true_row:,ixk,ixt],0,policyH[0,ixk,ixt])
        # # And the limiting function as m -> inf; C goes to limiting solution and H goes to 0
        # minterp = np.insert(minterp, minterp.shape, 20*np.max(minterp))
        # cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*np.max(minterp))
        # hinterp = np.insert(hinterp, hinterp.shape, 0)

        minterp = np.insert(endM1[colarg[ixk]:,ixk,ixt],0,0)
        cinterp = np.insert(endC1[colarg[ixk]:,ixk,ixt],0,0)
        hinterp = np.insert(endH1[colarg[ixk]:,ixk,ixt],0,policyH[0,ixk,ixt])
        # And the limiting function as m -> inf; C goes to limiting solution and H goes to 0
        # minterp = np.insert(minterp, minterp.shape, 20*np.max(minterp))
        # cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*np.max(minterp))
        minterp = np.insert(minterp, minterp.shape, policyM[nM+1,ixt])
        cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*policyM[nM+1,ixt])
        hinterp = np.insert(hinterp, hinterp.shape, 0)

        # interpolate:
        policyC[1:(nM+1), ixk+1, ixt] = np.interp(policyM[1:(nM+1),ixt], minterp, cinterp)
        policyH[1:(nM+1), ixk+1, ixt] = np.interp(policyM[1:(nM+1),ixt], minterp, hinterp)

    # When K --> 0, h --> 0 and c follows limiting solution
    policyH[1:,0,ixt] = 0
    policyC[1:,0,ixt] = picc[ixt] * policyM[1:,ixt]

    # When M --> Inf, same
    policyH[nM+1,:,ixt] = 0
    policyC[nM+1,:,ixt] = picc[ixt] * policyM[nM+1,ixt]
   
    # # Check monotonicity of consumption:
    # np.all(np.diff(policyC[:,:,ixt], axis=1) >= 0)
    # np.where(np.diff(policyC[:,:,ixt], axis=1) < 0)
    # np.all(np.diff(policyC[:,:,ixt], axis=0) >= 0)
    # # # and for value:
    # np.all(np.diff(policyH[:,1:,ixt], axis=1) >= 0)
    # np.where(np.diff(policyH[:,1,ixt]) < 0)
    # np.all(np.diff(policyH[:,1:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM
    # np.all(np.diff(policyH[1:,:,ixt], axis=1) >= 0)
    # np.all(np.diff(policyH[1:,:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM

    # No interpolation when M = 0
    ECprime[0,:,ixt] = (policyC[0,:,ixt])**(-gamma)
    ECprime[nM+1,:,ixt] = (policyC[nM+1,:,ixt])**(-gamma)
    policyCprime[:,:,ixt] = (policyC[:,:,ixt])**(-gamma)

    # Want to create an extrapolating function as K -> inf
    # first calculate the slopes for last two k-grid points, holding M constant:
    slope1 = (policyCprime[1:(nM+1),nK-1,ixt] - policyCprime[1:(nM+1),nK-2,ixt])/(endk[nK-1,ixt] - endk[nK-2,ixt])
    slope2 = (policyCprime[1:(nM+1),nK,ixt] - policyCprime[1:(nM+1),nK-1,ixt])/(endk[nK,ixt] - endk[nK-1,ixt])
    # % change in slopes:
    delta = (slope2 - slope1)/slope1
    # set the slope for extrapolation
    extrapslope = (1 + delta)*slope2

    # quad points of K
    Kvec = endk[1:,ixt][:,np.newaxis] * quad
    # mask for extrapolation
    above_mask = Kvec > endk[1:,ixt].max() 

    # Set up grids for expected marginal u of consumption
    for ixm in range(nM):
        # print(ixm)
        # interpolate v
        cprime_vec = np.interp(Kvec, endk[1:,ixt], policyCprime[ixm+1,1:,ixt])

        # Apply the linear extrapolation formula for values beyond the limit
        Knew = Kvec[above_mask]
        Kold = endk[nK,ixt]
        Cprimeold = policyCprime[ixm+1,nK,ixt]
        cprime_vec[above_mask] = extrapslope[ixm] * (Knew - Kold) + Cprimeold

        # Now get the expectation
        ECprime[ixm+1,1:,ixt] = np.mean(cprime_vec, axis=1)

    ECprime[:,0,ixt] = policyC[:,0,ixt]**(-gamma)

    # np.all(np.diff(ECprime[:,:,ixt], axis=1) <=0)
    # np.where(np.diff(ECprime[:,:,ixt], axis=0) > 0)
    # np.all(np.diff(ECprime[:,:,ixt], axis=0)<= 0)
    # np.all(np.diff(policyCprime[:,:,ixt], axis=0)<= 0)
    # np.where(np.diff(ECprime[:,:,ixt], axis=0) > 0)
    #### end period T-1 ####

    #### ITERATE BACKWARDS ####
    for ixt in range(lifespan-3, -1, -1):
        # ixt = lifespan-1-1-1
        print(ixt)
        # limiting function
        picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

        # Get endogeneous consumption
        # Create a mask for when we were not using the liquidity constrained solution
        ECprimesub = ECprime[:,:,ixt+1] * (egmmask[:,:,ixt+1])
        ECprimesub[ECprimesub == 0] = np.nan
        endC1[:,:,ixt] = (beta * R * ECprimesub[1:(nM+1),1:])**(-1 / gamma)

        # then get the accompanying labor supply
        endk[1:,ixt] = endk[1:,ixt+1]/(1+growth)
        prevK = np.tile(endk[1:,ixt], (nK, 1))
        endH1[:,:,ixt] = ((beta/alpha) * prevK * ECprimesub[1:(nM+1),1:])**(1/(eta-1))

        # Then back out M_{T-1}
        Mtemp = np.tile(policyM[1:(nM+1),ixt+1],(nM,1)).T
        endM1[:,:,ixt] = (Mtemp - prevK*endH1[:,:,ixt])/R + endC1[:,:,ixt]

        # get mask for positive M
        # Will need to be careful -- this will break if all M < 0
        negmmask = endM1[:,:,ixt] < 0
        nanmask = np.isnan(endM1[:,:,ixt])
        negmmask = negmmask | nanmask
        consmask = endC1[:,:,ixt] > endM1[:,:,ixt]
        dropmask = ~(negmmask | consmask)
        first_true_row = np.argmax(np.all(dropmask, axis=1))
        nonsoncstrained = np.where(dropmask, endM1[:,:,ixt], np.nan)
        mincolval = np.nanmin(nonsoncstrained, axis=0)
        colarg = np.argmax(dropmask, axis=0)
        colarg[np.isnan(mincolval)] = nK

        # set up policyM
        gridmin = np.full(1,np.nanmin(mincolval, axis=0))
        gridmax = np.full(1,np.maximum(np.nanmax(nonsoncstrained),M_grid[nM,ixt]))
        # gridmax = np.full(1,M_grid[nM,ixt])
        # if M_grid[nM,ixt] - gridmin < 1:
        #     gridmax = np.nanmax(nonsoncstrained)
        # else:
        #     gridmax = M_grid[nM,ixt]

        policyM[1:(nM+1),ixt] = GetGrid(gridmin, gridmax, nM, "power",2).flatten()
        policyM[0,ixt] = 0
        policyM[nM+1,ixt] = np.maximum(200, 20*np.max(policyM[1:(nM+1),ixt]))

        # set up a mask for use in next period to show where we are using the true solution and where we are using the constrained solution
        # Find the index where each value in mincolval first exceeds policyM[:,ixt]
        for ixk in range(len(mincolval)):
            egmmask[:,ixk+1,ixt] = policyM[:,ixt] > mincolval[ixk]

        # numerically solve for optimal H when M=0
        # Set grid for interpolation in next period
        minterp = policyM[:,ixt+1]
        kinterp = endk[:,ixt+1]
        # Create a 2D interpolation function
        interpolator = RegularGridInterpolator((minterp, kinterp), ECprime[:,:,ixt+1],
                                            method='linear', bounds_error=False, fill_value=None)
          
        # Will interpolate on this period's solution for our new grid
        for ixk in range(nK):
            # print(ixk)
            # First get the M = 0limiting solution numerically
            bounds = [(1e-8, None)] 
            start  = policyH[0,ixk+1,ixt+1]
            thisK = endk[ixk+1,ixt]
            sol = minimize(getH, start, 
                    args=(thisK, alpha,eta,beta,interpolator),
                    bounds=bounds, method='L-BFGS-B', tol=1e-10)
            policyH[0,ixk+1,ixt] = sol.x
            policyC[0,ixk+1,ixt] = cmin

            # drop constrained points
            # minterp = np.insert(endM1[first_true_row:,ixk,ixt],0,0)
            # cinterp = np.insert(endC1[first_true_row:,ixk,ixt],0,0)
            # hinterp = np.insert(endH1[first_true_row:,ixk,ixt],0,policyH[0,ixk+1,ixt])
            minterp = np.insert(endM1[colarg[ixk]:,ixk,ixt],0,0)
            cinterp = np.insert(endC1[colarg[ixk]:,ixk,ixt],0,0)
            hinterp = np.insert(endH1[colarg[ixk]:,ixk,ixt],0,policyH[0,ixk+1,ixt])
            # And the limiting function as m -> inf; C goes to limiting solution and H goes to 0
            # minterp = np.insert(minterp, minterp.shape, 20*np.max(minterp))
            # cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*np.max(minterp))
            minterp = np.insert(minterp, minterp.shape, policyM[nM+1,ixt])
            cinterp = np.insert(cinterp, cinterp.shape, picc[ixt]*policyM[nM+1,ixt])
            hinterp = np.insert(hinterp, hinterp.shape, 0)

            # interpolate:
            policyC[1:(nM+1), ixk+1, ixt] = np.interp(policyM[1:(nM+1),ixt], minterp, cinterp)
            policyH[1:(nM+1), ixk+1, ixt] = np.interp(policyM[1:(nM+1),ixt], minterp, hinterp)

        # When K = 0, h = 0 and c is limiting solution
        policyH[1:,0,ixt] = 0
        policyC[1:,0,ixt] = picc[ixt] * policyM[1:,ixt]

        # edge case
        policyC[0,0,ixt] = cmin
        policyH[0,0,ixt] = 0

        # When M --> Inf, same
        policyH[nM+1,:,ixt] = 0
        policyC[nM+1,:,ixt] = picc[ixt] * policyM[nM+1,ixt]

        # # Check monotonicity of consumption:
        # np.all(np.diff(policyC[:,:,ixt], axis=1) >= 0)
        # np.all(np.diff(policyC[:,:,ixt], axis=0) >= 0)
        # # # and for value:
        # np.all(np.diff(policyH[:,1:,ixt], axis=1) >= 0)
        # np.where(np.diff(policyH[:,1:,ixt], axis=1) < 0)
        # np.all(np.diff(policyH[:,1:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM
        # np.all(np.diff(policyH[1:,:,ixt], axis=1) >= 0)
        # np.all(np.diff(policyH[1:,:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM

        # No interpolation when M = 0
        ECprime[0,:,ixt] = (policyC[0,:,ixt])**(-gamma)
        ECprime[nM+1,:,ixt] = (policyC[nM+1,:,ixt])**(-gamma)
        policyCprime[:,:,ixt] = (policyC[:,:,ixt])**(-gamma)

        # Want to create an extrapolating function as K -> inf
        # first calculate the slopes for last two k-grid points, holding M constant:
        slope1 = (policyCprime[1:(nM+1),nK-1,ixt] - policyCprime[1:(nM+1),nK-2,ixt])/(endk[nK-1,ixt] - endk[nK-2,ixt])
        slope2 = (policyCprime[1:(nM+1),nK,ixt] - policyCprime[1:(nM+1),nK-1,ixt])/(endk[nK,ixt] - endk[nK-1,ixt])
        # % change in slopes:
        delta = (slope2 - slope1)/slope1
        # set the slope for extrapolation
        extrapslope = (1 + delta)*slope2

        # # alternatively just use slope to extrapolate linearly
        # extrapslope = -slope2

        # quad points of K
        Kvec = endk[1:,ixt][:,np.newaxis] * quad
        # mask for extrapolation
        above_mask = Kvec > endk[1:,ixt].max() 

        # Set up grids for expected marginal u of consumption
        for ixm in range(nM):
            # print(ixm)
            # interpolate v
            cprime_vec = np.interp(Kvec, endk[1:,ixt], policyCprime[ixm+1,1:,ixt])

            # Apply the linear extrapolation formula for values beyond the limit
            Knew = Kvec[above_mask]
            Kold = endk[nK,ixt]
            Cprimeold = policyCprime[ixm+1,nK,ixt]
            cprime_vec[above_mask] = extrapslope[ixm] * (Knew - Kold) + Cprimeold

            # Now get the expectation
            ECprime[ixm+1,1:,ixt] = np.mean(cprime_vec, axis=1)

        ECprime[:,0,ixt] = policyC[:,0,ixt]**(-gamma)

    #### END BACKWARDS RECURSION ####

    # # Now set up the grids to cover limits
    # endM = np.vstack((np.zeros((1, endM.shape[1])), endM))
    # # endC = np.pad(endC, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cmin)
    # endC = np.pad(endC, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # # and add value for the limiting function
    # newrow = 20*endM[nM,:]
    # endM = np.vstack((endM, newrow))
    # # now add the new consumption
    # crow = newrow * picc
    # cmat = np.tile(crow, (nK, 1))
    # endC = np.concatenate((endC, cmat[np.newaxis, :, :]), axis=0)
    
    # # Lastly, add in points for when k = 0
    # k0c = endM * picc
    # # k0c[:,0] == endM[:,0] * picc[0]
    # # k0c[:,40] == endM[:,40] * picc[40]
    # endC = np.concatenate((k0c[:,np.newaxis,  :],endC), axis=1)

    return policyC, policyH, endk, policyM