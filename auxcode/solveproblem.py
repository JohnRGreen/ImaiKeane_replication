# Restarting this project from scratch in May, 2024.
def solveProblem(M_grid, k_grid, **params_sp):
    # params_sp = params_dict

    # get the parameters 
    phi         = params_sp['phi']
    beta        = params_sp['beta']
    gamma       = params_sp['gamma']
    alpha       = params_sp['alpha']
    eta         = params_sp['eta']
    growth      = params_sp['growth']

    # wage shocks:
    sigma       = params_sp['sigma']
    mu = -(1/2)*sigma**2 # mean of wage shock

    # dsicretize the shock distribution
    quad = discretize_log_distribution(nq, mu, sigma)

    #### Set up grids ####
    Vgrid            = np.full((nM+2,nK+1,lifespan),np.nan)
    EVgrid           = np.full((nM+2,nK+1,lifespan),np.nan)
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
    egmk            = np.full((nK,lifespan-1),np.nan)
    egmmask = np.full((nM+2,nK+1,lifespan),np.nan)
    egmmask[:,0,:] = 0
    subegmmask = np.full((nM,nK,lifespan-1),np.nan)
    cpath = np.full((nM+2,nK+1,lifespan),0)
    egmgrid = np.full((nM,nK, lifespan-1),np.nan)
    #### end grid setup ####

    #### PERIOD T ####
    ixt = lifespan-1
    # In period T+1 we consume everything, use terminal period value function to get utility
    
    # get consumption func
    lambdaconst = (phi * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst) / ( 1 + (R*lambdaconst))

    # exclude M = 0 point
    policyC[1:,:,ixt] = picc[ixt]*M_grid[1:,ixt][:, None]
    # set lower bound
    policyC[0,:,ixt]  = cmin 

    # H is just 0
    policyH[:,:,ixt] = 0

    # get marginal value grid
    ECprime[:,:,ixt] = (policyC[:,:,ixt])**(-gamma)

    # Calculate value
    Vgrid[:,:,ixt] = (policyC[:,:,ixt]**(1-gamma))/(1-gamma) + (phi/(1-gamma))*((R * (1 - picc[ixt])*M_grid[:,ixt][:, None])**(1-gamma))
    EVgrid[:,:,ixt] = Vgrid[:,:,ixt]
    #### END PERIOD T ####

    #### period T-1 ####
    # In T-1, since we won't be working in T, human capital in T is not relevant
    ixt = lifespan - 1 - 1

    # set limiting consumption function for this period
    lambdaconst = (beta * R)**(-1/gamma)
    picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

    # endogeneous C; avoid the limit points
    endC1[:,:,ixt] = lambdaconst * policyC[1:(nM+1),1:,ixt+1]

    # endogeneous H
    # endk[1:,ixt] = endk[1:,ixt+1]/(1+growth[ixt])
    endk[1:,ixt] = endk[1:,ixt+1]
    prevK = np.tile(endk[1:,ixt], (nK, 1))
    endH1[:,:,ixt] = ((beta/alpha) * prevK * ECprime[1:(nM+1),1:,ixt+1])**(1/(eta-1))

    # Then back out M_{T-1}
    Mtemp = np.tile(M_grid[1:(nM+1),ixt+1],(nM,1)).T
    endM1[:,:,ixt] = (Mtemp - prevK*endH1[:,:,ixt])/R + endC1[:,:,ixt]

    # Now we have, for a fixed K_{T-1}, a combination of M_{T-1} and optimal c_{T-1} and h_{T-1}
    # Want to fit to our structured grid

    # get mask for positive M
    # Will need to be careful -- this will break if all M < 0
        # get mask for valid, nonconstrained points:
    dropmask = ~(endC1[:,:,ixt] > endM1[:,:,ixt])
    # get mask for where we didn't even do EGM:
    valid_mask = ~np.isnan(endM1[:,:,ixt])
    # Find first column we have valid, nonconstrained values:
    firstvalid = np.where(np.all(valid_mask==False, axis=0), nK, np.argmax(valid_mask, axis=0))
    # Find first row in each column where we have valid, nonconstrained values:
    colarg = np.where(np.all(valid_mask == False, axis=0), nK, np.argmax(np.where(valid_mask, dropmask, -np.inf), axis=0))
    # Then calculate the K points at which we can do interpolation
    valid_indices = np.max(np.where(valid_mask, dropmask, -np.inf), axis=0)==1

    # Now let's find the point where the credit constraint first bites
    cc = np.full(nK,np.nan)
    slopes =np.full(nK,np.nan)
    # Get slopes where we can find a midpoint from the constrained and unconstrained endogeneous points
    midpointmask = (valid_indices) & (firstvalid < colarg)
    extrapmask = (valid_indices) & (firstvalid == colarg)
    # If we have sufficient points, we prefer to find the point of constraint by interpolation
    # find point of constraint for those where we can use a midpoint
    slopes[midpointmask] = (endC1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] - 
            endC1[colarg[midpointmask] - 1, np.arange(nK)[midpointmask], ixt]) / (
            endM1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] - 
            endM1[colarg[midpointmask] - 1, np.arange(nK)[midpointmask], ixt])
    cc[midpointmask] = -slopes[midpointmask] * endM1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] + endC1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt]
    
    if np.sum(midpointmask) > .2*nK:
        # then interpolate for those where we have no constrained endogeneous point
        consinterp = np.array([cmin] + list(cc[midpointmask]))
        kinterp = np.array([0.0005] + list(endk[1:, ixt][midpointmask]))
        cc[extrapmask] = np.interp(endk[1:, ixt][extrapmask], kinterp, consinterp)
    else:
        # if we don't have enough points we can interpolate, then we can do an extrapolation
        slopes[extrapmask] = (endC1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] - 
            endC1[colarg[extrapmask] + 1, np.arange(nK)[extrapmask], ixt]) / (
            endM1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] - 
            endM1[colarg[extrapmask] + 1, np.arange(nK)[extrapmask], ixt])
        cc[extrapmask] = -slopes[extrapmask] * endM1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] + endC1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt]
        # Then use a line of best fit to insure monotonicity.
        # Extract the relevant data
        X = endk[1:, ixt].reshape(-1, 1)[valid_indices]  # Independent variable
        y = cc[valid_indices]  # Dependent variable
        # Fit the linear regression model
        model = LinearRegression().fit(X, y)
        # Predict the values for cc
        X_full = endk[1:, ixt].reshape(-1, 1)
        cc = model.predict(X_full)
        # make sure regression line is not higher than any of the values we will use for relevant points
        diff = cc[valid_indices] - endC1[colarg[valid_indices], np.arange(nK)[valid_indices], ixt]
        shift = max(0, np.max(diff))
        cc = cc - shift
        cc = np.maximum(cmin, cc)
    
    # Set up grid values for M
    nonsoncstrained = np.where(dropmask, endM1[:,:,ixt], np.nan)
    # min col
    mincolval = np.full(nK,np.nan)
    mincolval[valid_indices] = endM1[colarg[valid_indices], np.arange(nK)[valid_indices], ixt]
    # Now for fixed K given by endk, we have a clean interpolation on the optimal solution for each mincolval. starting at the max of mincolval
    # Going to set up new M values to interpolate on, starting at this max, so that we keep a clean optimal portion of the grid to do the EGM on (below is just linear for the constrained)
    gridmin = np.full(1,np.nanmin(mincolval, axis=0))
    if gridmin > M_grid[nM,ixt]:
        gridmax = np.full(1,np.nanmax(nonsoncstrained))
    else:
        gridmax = np.full(1,M_grid[nM,ixt])
    policyM[1:(nM+1),ixt] = GetGrid(gridmin, gridmax, nM, "power",2).flatten()
    policyM[0,ixt] = cmin*2
    policyM[nM+1,ixt] = 200

    # Now set up a mask for us in the next period to show where we are using the true solution and where we are using the constrained solution
    for ixk in range(nK):
        cpath[:,ixk+1,ixt] = (policyM[:,ixt] >= cc[ixk]) & (policyM[:,ixt] < mincolval[ixk])
        egmmask[:,ixk+1,ixt] = policyM[:,ixt] >= mincolval[ixk]

    # Will interpolate on this period's solution for our new grid
    if sum(valid_indices)<nK:
        kstart = np.argmin(valid_indices)
    else:
        kstart = nK

    # In period T-1, we have analytic form for optimal solution when credit constrained
    policyH[0,:,ixt] = ((beta/alpha) * (endk[:,ixt]**(1-gamma)) * picc[ixt+1]**(-gamma) )**(1/(eta-1+gamma))
    policyC[0,:,ixt] = cmin
    # policyH0= policyH[0,:,ixt].copy()

    # Whenever we have anough values, interpolate
    for ixk in range(kstart):
        # print(ixk)
        # insert the point at which agent becomes credit constrained
        policyC[1:(nM+1), ixk+1, ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)] = policyM[1:(nM+1),ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)]
        policyH[1:(nM+1), ixk+1, ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)] = policyH[0,ixk+1,ixt]

        # on the segment to being credit constrained:
        minterp = [cc[ixk], endM1[colarg[ixk],ixk,ixt]]
        cinterp = [cc[ixk], endC1[colarg[ixk],ixk,ixt]]
        hinterp = [policyH[0,ixk+1,ixt],endH1[colarg[ixk],ixk,ixt]]
        policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][cpath[1:(nM+1),ixk+1,ixt]==1], minterp, cinterp)
        policyH[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][cpath[1:(nM+1),ixk+1,ixt]==1], minterp, hinterp)
        # On this segment, it is possible interpolation underestimates consumption; we can check that values are strictly increasing
        if ixk >0:
            if np.any(policyC[1:(nM+1), ixk, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] > policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1]):
                # print('C not increasing')
                # print(ixk)
                policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.maximum(policyC[1:(nM+1), ixk, ixt][cpath[1:(nM+1),ixk+1,ixt]==1], 
                                                                                         policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1])
                
        # And the limiting function as m -> inf; C goes to limiting solution and H goes to 0
        minterp = np.insert(endM1[colarg[ixk]:,ixk,ixt], endM1[colarg[ixk]:,ixk,ixt].shape, policyM[nM+1,ixt])
        cinterp = np.insert(endC1[colarg[ixk]:,ixk,ixt], endC1[colarg[ixk]:,ixk,ixt].shape, picc[ixt]*policyM[nM+1,ixt])
        hinterp = np.insert(endH1[colarg[ixk]:,ixk,ixt], endH1[colarg[ixk]:,ixk,ixt].shape, 0)

        # interpolate:
        policyC[1:(nM+1), ixk+1, ixt][egmmask[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][egmmask[1:(nM+1),ixk+1,ixt]==1], minterp, cinterp)
        policyH[1:(nM+1), ixk+1, ixt][egmmask[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][egmmask[1:(nM+1),ixk+1,ixt]==1], minterp, hinterp)

    # Above this range, they are all credit constrained 
    for ixk in range(kstart,nK):
        # print(ixk)
        # credit constrained
        policyC[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)] = policyM[:,ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)]
        policyH[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)] = policyH[0,ixk+1,ixt]

        # increase c by small number
        cons = np.minimum(1,1.01* policyC[:, ixk, ixt]/policyM[:,ixt])[(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)]
        policyC[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)] = cons*policyM[:,ixt][(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)]
        # then increase H
        policyH[1:(nM+1), ixk+1, ixt] = np.minimum((policyH[0,ixk+1,ixt]/policyH[0,ixk,ixt])*policyH[1:(nM+1), ixk, ixt],policyH[0,ixk+1,ixt])

    # When K --> 0, h --> 0 and c follows limiting solution
    # policyH[:,0,ixt] = np.linspace(policyH[0,0,ixt], 0, nM+2)
    policyH[1:,0,ixt] = 0
    policyC[1:,0,ixt] = picc[ixt] * policyM[1:,ixt]

    # When M --> Inf, same
    policyH[nM+1,:,ixt] = 0
    policyC[nM+1,:,ixt] = picc[ixt] * policyM[nM+1,ixt]
   
    # try numerical optimization here
    # minterp = policyM[:,ixt+1]
    # kinterp = endk[:,ixt+1]
    # # Create a 2D interpolation function
    # interpolator = RegularGridInterpolator((minterp, kinterp), EVgrid[:,:,ixt+1],
    #                                     method='linear', bounds_error=False, fill_value=None)
    # for ixk in range(nK):
    #     bounds = [(1e-8, None)] 
    #     # start  = policyH[0,ixk+1,ixt+1]
    #     start = 20.4
    #     thisK = endk[ixk+1,ixt]
    #     sol = minimize(getH, start, 
    #             args=(thisK, alpha,eta,beta,interpolator,growth[ixt]),
    #             bounds=bounds, method='L-BFGS-B', tol=1e-10)
    #     policyH[0,ixk+1,ixt] = sol.x

    # Check monotonicity of consumption:
    # np.all(np.diff(policyC[:,:,ixt], axis=1) >= 0)
    # np.where(np.diff(policyC[:,:,ixt], axis=1) < 0)
    # np.all(np.diff(policyC[:,:,ixt], axis=0) >= 0)
    # np.where(np.diff(policyC[:,:,ixt], axis=0) < 0)
    # # # and for value:
    # np.all(np.diff(policyH[:,1:,ixt], axis=1) >= 0)
    # np.where(np.diff(policyH[:,1,ixt]) < 0)
    # np.all(np.diff(policyH[:,1:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM
    # np.all(np.diff(policyH[1:,:,ixt], axis=1) >= 0)
    # np.where(np.diff(policyH[1:,:,ixt], axis=1) < 0)

    # np.all(np.diff(policyH[1:,:,ixt], axis=0) <= 0)# We now have a grid of optimal consumption for k-grid and a grid of endogeneous M, endM

    # No interpolation when M = 0
    ECprime[0,:,ixt] = (policyC[0,:,ixt])**(-gamma)
    ECprime[nM+1,:,ixt] = (policyC[nM+1,:,ixt])**(-gamma)
    policyCprime[:,:,ixt] = (policyC[:,:,ixt])**(-gamma)

    # first calculate the slopes for last two k-grid points, holding M constant:
    # slope1c = (policyC[1:(nM+1),nK-1,ixt] - policyC[1:(nM+1),nK-2,ixt])/(endk[nK-1,ixt] - endk[nK-2,ixt])
    # slope2c = (policyC[1:(nM+1),nK,ixt] - policyC[1:(nM+1),nK-1,ixt])/(endk[nK,ixt] - endk[nK-1,ixt])
    # deltac = np.where(slope1c != 0, (slope2c - slope1c) / slope1c, 0)
    # extrapslopec = np.where(slope1c != 0, (1 + deltac) * slope2c, slope2c)
    # extrapslopec[extrapslopec<0] = 0

    # quad points of K
    Kvec = endk[1:,ixt][:,np.newaxis] * quad
    # mask for extrapolation
    # above_mask = Kvec > endk[1:,ixt].max() 

    # In order to keep this going, we need to get a dense grid of E[c_t] around the "good", non-constrained EGM points. 
    ixk1 = np.argmax(valid_indices)
    ixk2 = len(valid_indices) - np.argmax(valid_indices[::-1]) - 1
    egmkmin = np.full(1,endk[ixk1+1,ixt])
    egmkmax = np.full(1,endk[ixk2+1,ixt])
    egmk[:,ixt] = GetGrid(egmkmin, egmkmax, nK, "power",1.5).flatten()
    egmkindex = np.full(nK,np.nan)
    for ixk in range(nK):
        egmkindex[ixk] = np.abs(endk[:, ixt] - egmk[ixk, ixt]).argmin()
    # egmkindex = egmkindex.astype(int)
    subegmmask[:,:,ixt] = egmmask[1:(nM+1), egmkindex.astype(int), ixt] + cpath[1:(nM+1), egmkindex.astype(int), ixt]

    # Get value grid
    minterp = policyM[:,ixt+1]
    kinterp = endk[:,ixt+1]
    # Create a 2D interpolation function
    nextm = R*(np.tile(policyM[:,ixt],(nK+1,1)).T - policyC[:,:,ixt]) + np.tile(endk[:,ixt],(nM+2,1))*policyH[:,:,ixt]
    nextk = np.tile(endk[:,ixt],(nM+2,1)) * (1+growth[ixt])
    interpolator = RegularGridInterpolator((minterp, kinterp), EVgrid[:,:,ixt+1],
                                        method='linear', bounds_error=False, fill_value=None)
    Vgrid[:,:,ixt] = (policyC[:,:,ixt]**(1-gamma))/(1-gamma) - (alpha/eta)* policyH[:,:,ixt]**eta + beta * interpolator((nextm,nextk))
    Kvecfull = endk[:,ixt][:,np.newaxis] * quad

    for ixm in range(nM+2):
        v_vec = np.interp(Kvecfull, endk[:,ixt], Vgrid[ixm,:,ixt])
        EVgrid[ixm,:,ixt] = np.mean(v_vec, axis=1)

    # Set up grids for expected marginal u of consumption
    for ixm in range(nM):
        # print(ixm)
        # interpolate v
        c_vec = np.interp(Kvec, endk[1:,ixt], policyC[ixm+1,1:,ixt])

        # Apply the linear extrapolation formula for values beyond the limit
        # Knew = Kvec[above_mask]
        # Kold = endk[nK,ixt]
        # Cold = policyC[ixm+1,nK,ixt]
        # c_vec[above_mask] = extrapslopec[ixm] * (Knew - Kold) + Cold
        # c_vec[c_vec > policyM[ixm+1,ixt]] = policyM[nM+1,ixt]
        # Now get the expectation
        ECprime[ixm+1,1:,ixt] = np.mean(c_vec**(-gamma), axis=1)

        # then interpolate EC onto egm grid
        egmgrid[ixm,:,ixt] = np.interp(egmk[:,ixt], endk[1:,ixt], ECprime[ixm+1,1:,ixt])

    ECprime[:,0,ixt] = policyC[:,0,ixt]**(-gamma)

    # np.all(np.diff(ECprime[:,:,ixt], axis=1) <=0)
    # np.where(np.diff(ECprime[:,:,ixt], axis=0) > 0)
    # np.all(np.diff(ECprime[:,:,ixt], axis=0)<= 0)
    # np.all(np.diff(policyCprime[:,:,ixt], axis=0)<= 0)
    # np.where(np.diff(ECprime[:,:,ixt], axis=0) > 0)
    #### end period T-1 ####
    # for ixt in range(lifespan-3, 37, -1):
    #     print(ixt)

    #### ITERATE BACKWARDS ####
    for ixt in range(lifespan-3, -1, -1):
    # for ixt in range(lifespan-3, 9, -1):
        # ixt = lifespan-1-1-1
        # print(ixt)
        # limiting function
        picc[ixt] = (R*lambdaconst*picc[ixt+1]) / ( 1 + (R*lambdaconst*picc[ixt+1]))

        # Get endogeneous consumption
        ECprimesub = egmgrid[:,:,ixt+1] * (subegmmask[:,:,ixt+1])
        ECprimesub[subegmmask[:,:,ixt+1] == 0] = np.nan
        endC1[:,:,ixt] = (beta * R * ECprimesub)**(-1 / gamma)

        # then get the accompanying labor supply
        endk[1:,ixt] = egmk[:,ixt+1]/(1+growth[ixt])
        prevK = np.tile(endk[1:,ixt], (nK, 1))
        endH1[:,:,ixt] = ((beta/alpha) * prevK * ECprimesub)**(1/(eta-1))

        # Then back out M_{T-1}
        Mtemp = np.tile(policyM[1:(nM+1),ixt+1],(nM,1)).T
        endM1[:,:,ixt] = (Mtemp - prevK*endH1[:,:,ixt])/R + endC1[:,:,ixt]

        # get mask for valid, nonconstrained points:
        dropmask = ~(endC1[:,:,ixt] > endM1[:,:,ixt])
        # get mask for where we didn't even do EGM:
        valid_mask = ~np.isnan(endM1[:,:,ixt])
        # Find first column we have valid, nonconstrained values:
        firstvalid = np.where(np.all(valid_mask==False, axis=0), nK, np.argmax(valid_mask, axis=0))
        # Find first column we have valid, nonconstrained values:
        colarg = np.where(np.all(valid_mask == False, axis=0), nK, np.argmax(np.where(valid_mask, dropmask, -np.inf), axis=0))
        # Then calculate the K points at which we can do interpolation
        valid_indices = np.max(np.where(valid_mask, dropmask, -np.inf), axis=0)==1

        # Now let's find the point where the credit constraint first bites
        cc = np.full(nK,np.nan)
        slopes =np.full(nK,np.nan)
        # Get slopes where we can find a midpoint from the constrained and unconstrained endogeneous points
        midpointmask = (valid_indices) & (firstvalid < colarg)
        extrapmask = (valid_indices) & (firstvalid == colarg) & (colarg < nK-1)
        # If we have sufficient points, we prefer to find the point of constraint by interpolation
        # find point of constraint for those where we can use a midpoint
        slopes[midpointmask] = (endC1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] - 
                endC1[colarg[midpointmask] - 1, np.arange(nK)[midpointmask], ixt]) / (
                endM1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] - 
                endM1[colarg[midpointmask] - 1, np.arange(nK)[midpointmask], ixt])
        cc[midpointmask] = -slopes[midpointmask] * endM1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt] + endC1[colarg[midpointmask], np.arange(nK)[midpointmask], ixt]
        
        if np.sum(midpointmask) > .2*nK:
            # then interpolate for those where we have no constrained endogeneous point
            consinterp = np.array([cmin] + list(cc[midpointmask]))
            kinterp = np.array([0.0005] + list(endk[1:, ixt][midpointmask]))
            cc[extrapmask] = np.interp(endk[1:, ixt][extrapmask], kinterp, consinterp)
        else:
            # if we don't have enough points we can interpolate, then we can do an extrapolation
            slopes[extrapmask] = (endC1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] - 
                endC1[colarg[extrapmask] + 1, np.arange(nK)[extrapmask], ixt]) / (
                endM1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] - 
                endM1[colarg[extrapmask] + 1, np.arange(nK)[extrapmask], ixt])
            cc[extrapmask] = -slopes[extrapmask] * endM1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt] + endC1[colarg[extrapmask], np.arange(nK)[extrapmask], ixt]
            # Then use a line of best fit to insure monotonicity.
            # Extract the relevant data
            X = endk[1:, ixt].reshape(-1, 1)[extrapmask | midpointmask]  # Independent variable
            y = cc[extrapmask | midpointmask]  # Dependent variable
            # Fit the linear regression model
            model = LinearRegression().fit(X, y)
            # Predict the values for cc
            X_full = endk[1:, ixt].reshape(-1, 1)
            cc = model.predict(X_full)
            # make sure regression line is not higher than any of the values we will use for relevant points
            diff = cc[valid_indices] - endC1[colarg[valid_indices], np.arange(nK)[valid_indices], ixt]
            shift = max(0, np.max(diff))
            cc = cc - shift
            cc = np.maximum(cmin, cc)

        # Set up grid values for M
        nonsoncstrained = np.where(dropmask, endM1[:,:,ixt], np.nan)
        mincolval = np.full(nK,np.nan)
        mincolval[valid_indices] = endM1[colarg[valid_indices], np.arange(nK)[valid_indices], ixt]
        # Now for fixed K given by endk, we have a clean interpolation on the optimal solution for each mincolval. starting at the max of mincolval
        # Going to set up new M values to interpolate on, starting at this max, so that we keep a clean optimal portion of the grid to do the EGM on (below is just linear for the constrained)
        gridmin = np.full(1,np.nanmin(mincolval, axis=0))
        if gridmin > M_grid[nM,ixt]:
            gridmax = np.full(1,np.nanmax(nonsoncstrained))
        else:
            gridmax = np.full(1,M_grid[nM,ixt])
        policyM[1:(nM+1),ixt] = GetGrid(gridmin, gridmax, nM, "power",2).flatten()
        policyM[0,ixt] = cmin*2
        policyM[nM+1,ixt] = 200

        # Now set up a mask for us in the next period to show where we are using the true solution and where we are using the constrained solution
        for ixk in range(nK):
            cpath[:,ixk+1,ixt] = (policyM[:,ixt] >= cc[ixk]) & (policyM[:,ixt] < mincolval[ixk])
            egmmask[:,ixk+1,ixt] = policyM[:,ixt] >= mincolval[ixk]

        # Will interpolate on this period's solution for our new grid
        if sum(valid_indices)<nK:
            kstart = np.argmin(valid_indices)
        else:
            kstart = nK

        # numerically solve for optimal H when M=0
        # Set grid for interpolation in next period
        minterp = policyM[:,ixt+1]
        kinterp = endk[:,ixt+1]
        # Create a 2D interpolation function
        interpolator = RegularGridInterpolator((minterp, kinterp), EVgrid[:,:,ixt+1],
                                            method='linear', bounds_error=False, fill_value=None)
        
        # Will interpolate on this period's solution for our new grid
        for ixk in range(kstart):
            # print(ixk)
            # First get the M = 0limiting solution numerically
            bounds = [(1e-8, None)] 
            # start  = policyH[0,ixk+1,ixt+1]
            if ixk>=1:
                start = policyH[0,ixk,ixt]
            else:
                start = 20.4
            thisK = endk[ixk+1,ixt]
            sol = minimize(getH, start, 
                    args=(thisK, alpha,eta,beta,interpolator,growth[ixt]),
                    bounds=bounds, method='L-BFGS-B', tol=1e-10)
            if ixk>=1:
                policyH[0,ixk+1,ixt] = max(sol.x,policyH[0,ixk,ixt])
                policyH[0,ixk+1,ixt] = sol.x
            else:
                policyH[0,ixk+1,ixt] = sol.x
            policyC[0,ixk+1,ixt] = cmin

            # Get the credit constrained:
            policyC[1:(nM+1), ixk+1, ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)] = policyM[1:(nM+1),ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)]
            policyH[1:(nM+1), ixk+1, ixt][(egmmask[1:(nM+1),ixk+1,ixt]==0) & (cpath[1:(nM+1),ixk+1,ixt]==0)] = policyH[0,ixk+1,ixt]

            # on the segment to being credit constrained:
            minterp = [cc[ixk], endM1[colarg[ixk],ixk,ixt]]
            cinterp = [cc[ixk], endC1[colarg[ixk],ixk,ixt]]
            hinterp = [policyH[0,ixk+1,ixt],endH1[colarg[ixk],ixk,ixt]]
            policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][cpath[1:(nM+1),ixk+1,ixt]==1], minterp, cinterp)
            policyH[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][cpath[1:(nM+1),ixk+1,ixt]==1], minterp, hinterp)
            # On this segment, it is possible interpolation underestimates consumption; we can check that values are strictly increasing
            if ixk >0:
                if np.any(policyC[1:(nM+1), ixk, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] > policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1]):
                    # print('C not increasing')
                    # print(ixk)
                    policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1] = np.maximum(policyC[1:(nM+1), ixk, ixt][cpath[1:(nM+1),ixk+1,ixt]==1], 
                                                                                            policyC[1:(nM+1), ixk+1, ixt][cpath[1:(nM+1),ixk+1,ixt]==1])

            # And the limiting function as m -> inf; C goes to limiting solution and H goes to 0
            minterp = np.insert(endM1[colarg[ixk]:,ixk,ixt], endM1[colarg[ixk]:,ixk,ixt].shape, policyM[nM+1,ixt])
            cinterp = np.insert(endC1[colarg[ixk]:,ixk,ixt], endC1[colarg[ixk]:,ixk,ixt].shape, picc[ixt]*policyM[nM+1,ixt])
            hinterp = np.insert(endH1[colarg[ixk]:,ixk,ixt], endH1[colarg[ixk]:,ixk,ixt].shape, 0)

            # interpolate:
            policyC[1:(nM+1), ixk+1, ixt][egmmask[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][egmmask[1:(nM+1),ixk+1,ixt]==1], minterp, cinterp)
            policyH[1:(nM+1), ixk+1, ixt][egmmask[1:(nM+1),ixk+1,ixt]==1] = np.interp(policyM[1:(nM+1),ixt][egmmask[1:(nM+1),ixk+1,ixt]==1], minterp, hinterp)
        
        # Outside, we are just going to assume that C increases and h decreases at same rate as previous solution
        # Whenever we have anough values
        for ixk in range(kstart,nK):
            bounds = [(1e-8, None)] 
            start  = policyH[0,ixk+1,ixt+1]
            thisK = endk[ixk+1,ixt]
            sol = minimize(getH, start, 
                    args=(thisK, alpha,eta,beta,interpolator,growth[ixt]),
                    bounds=bounds, method='L-BFGS-B', tol=1e-10)
            policyH[0,ixk+1,ixt] = max(sol.x,policyH[0,ixk,ixt])
            policyC[0,ixk+1,ixt] = cmin
            # print(ixk)

            # credit constrained
            policyC[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)] = policyM[:,ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)]
            policyH[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==0) & (cpath[:,kstart-1,ixt]==0)] = policyH[0,ixk+1,ixt]

            # increase c by small number
            cons = np.minimum(1,1.01* policyC[:, ixk, ixt]/policyM[:,ixt])[(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)]
            policyC[:, ixk+1, ixt][(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)] = cons*policyM[:,ixt][(egmmask[:,kstart-1,ixt]==1) | (cpath[:,kstart-1,ixt]==1)]
            # then increase H
            policyH[1:(nM+1), ixk+1, ixt] = np.minimum((policyH[0,ixk+1,ixt]/policyH[0,ixk,ixt])*policyH[1:(nM+1), ixk, ixt],policyH[0,ixk+1,ixt])

        # fill in M = 0 C
        policyC[0,:,ixt] = cmin

        # When K --> 0, h --> 0 and c is limiting solution
        policyH[1:,0,ixt] = 0
        policyC[1:,0,ixt] = picc[ixt] * policyM[1:,ixt]

        # When M --> Inf, same
        policyH[nM+1,:,ixt] = 0
        policyC[nM+1,:,ixt] = picc[ixt] * policyM[nM+1,ixt]

        # edge case
        policyC[0,0,ixt] = cmin
        # policyH[0,0,ixt] = minimize(getH, start, 
        #             args=(endk[0,ixt], alpha,eta,beta,interpolator,growth[ixt]),
        #             bounds=bounds, method='L-BFGS-B', tol=1e-10).x[0]
        policyH[0,0,ixt] = 0

        # for ixk in range(nK):
        #     if np.all(np.diff(policyH[:,ixk,ixt])<=0)==False:
        #         print(ixk)
        #         print("H not decreasing with K at time")
        #         print(ixt)
        #     if np.all(np.diff(policyC[:,ixk,ixt])>=0)==False:
        #         print(ixk)
        #         print("C not increasing with K at time")
        #         print(ixt)
        # for ixm in range(nM):
        #     if np.all(np.diff(policyH[ixm,:,ixt])>=0)==False:
        #         print(ixm)
        #         print("H not decreasing with M at time")
        #         print(ixt)
        #     if np.all(np.diff(policyC[ixm,:,ixt])>=0)==False:
        #         print(ixk)
        #         print("C not increasing with M at time")
        #         print(ixt)

        # # Check monotonicity of consumption:
        # np.all(np.diff(policyC[:,:,ixt], axis=1) >= 0)
        # np.all(np.diff(policyC[:,:,ixt], axis=0) >= 0)
        # np.where(np.diff(policyC[:,:,ixt], axis=0) < 0)
        # np.where(np.isnan(policyC[:,:,ixt]))
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
        # slope1 = (policyCprime[1:(nM+1),nK-1,ixt] - policyCprime[1:(nM+1),nK-2,ixt])/(endk[nK-1,ixt] - endk[nK-2,ixt])
        # slope2 = (policyCprime[1:(nM+1),nK,ixt] - policyCprime[1:(nM+1),nK-1,ixt])/(endk[nK,ixt] - endk[nK-1,ixt])
        # delta = np.where(slope1 != 0, (slope2 - slope1) / slope1, 0)
        # extrapslope = np.where(slope1 != 0, (1 + delta) * slope2, slope2)

        # slope1c = (policyC[1:(nM+1),nK-1,ixt] - policyC[1:(nM+1),nK-2,ixt])/(endk[nK-1,ixt] - endk[nK-2,ixt])
        # slope2c = (policyC[1:(nM+1),nK,ixt] - policyC[1:(nM+1),nK-1,ixt])/(endk[nK,ixt] - endk[nK-1,ixt])
        # deltac = np.where(slope1c != 0, (slope2c - slope1c) / slope1c, 0)
        # extrapslopec = np.where(slope1c != 0, (1 + deltac) * slope2c, slope2c)
        # extrapslopec[extrapslopec<0] = 0

        # # alternatively just use slope to extrapolate linearly
        # extrapslope = -slope2

        # quad points of K
        Kvec = endk[1:,ixt][:,np.newaxis] * quad
        # mask for extrapolation
        # above_mask = Kvec > endk[1:,ixt].max() 

        # we will also make a dense grid around the good ixk points on which to do EGM
        ixk1 = np.argmax(valid_indices)
        ixk2 = len(valid_indices) - np.argmax(valid_indices[::-1]) - 1
        egmkmin = np.full(1,endk[ixk1+1,ixt])
        egmkmax = np.full(1,endk[ixk2+1,ixt])
        egmk[:,ixt] = GetGrid(egmkmin, egmkmax, nK, "power",1.5).flatten()
        egmkindex = np.full(nK,np.nan)
        for ixk in range(nK):
            egmkindex[ixk] = np.abs(endk[:, ixt] - egmk[ixk, ixt]).argmin()
        subegmmask[:,:,ixt] = egmmask[1:(nM+1), egmkindex.astype(int), ixt] + cpath[1:(nM+1), egmkindex.astype(int), ixt]

        # Get value grid
        minterp = policyM[:,ixt+1]
        kinterp = endk[:,ixt+1]
        # Create a 2D interpolation function
        nextm = R*(np.tile(policyM[:,ixt],(nK+1,1)).T - policyC[:,:,ixt]) + np.tile(endk[:,ixt],(nM+2,1))*policyH[:,:,ixt]
        nextk = np.tile(endk[:,ixt],(nM+2,1)) * (1+growth[ixt])
        nextk= np.clip(nextk,np.min(kinterp),np.max(kinterp))
        interpolator = RegularGridInterpolator((minterp, kinterp), EVgrid[:,:,ixt+1],
                                            method='linear', bounds_error=False, fill_value=None)
        Vgrid[:,:,ixt] = (policyC[:,:,ixt]**(1-gamma))/(1-gamma) - (alpha/eta)* policyH[:,:,ixt]**eta + beta * interpolator((nextm,nextk))
        Kvecfull = endk[:,ixt][:,np.newaxis] * quad

        for ixm in range(nM+2):
            v_vec = np.interp(Kvecfull, endk[:,ixt], Vgrid[ixm,:,ixt])
            EVgrid[ixm,:,ixt] = np.mean(v_vec, axis=1)

        # Set up grids for expected marginal u of consumption
        for ixm in range(nM):
            # print(ixm)
            # interpolate v
            # cprime_vec = np.interp(Kvec, endk[1:,ixt], policyCprime[ixm+1,1:,ixt])

            c_vec = np.interp(Kvec, endk[1:,ixt], policyC[ixm+1,1:,ixt])

            # Apply the linear extrapolation formula for values beyond the limit
            # Knew = Kvec[above_mask]
            # Kold = endk[nK,ixt]
            # Cold = policyC[ixm+1,nK,ixt]
            # c_vec[above_mask] = extrapslopec[ixm] * (Knew - Kold) + Cold
            # c_vec[c_vec > policyM[ixm+1,ixt]] = policyM[nM+1,ixt]
            # # Now get the expectation
            ECprime[ixm+1,1:,ixt] = np.mean(c_vec**(-gamma), axis=1)

            # Now get the expectation
            # ECprime[ixm+1,1:,ixt] = np.mean(cprime_vec, axis=1)
            # then interpolate EC onto egm grid
            egmgrid[ixm,:,ixt] = np.interp(egmk[:,ixt], endk[1:,ixt], ECprime[ixm+1,1:,ixt])

        ECprime[:,0,ixt] = policyC[:,0,ixt]**(-gamma)

    #### END BACKWARDS RECURSION ####
    return policyC, policyH, endk, policyM