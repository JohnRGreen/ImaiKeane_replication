def simulate(policyC, policyH, endk, policyM, params_sim):
    # hyper-parameters for simulation
    startage= params_sim["startage"]
    numSims = int(params_sim["numSims"])
    simlifespan = params_sim["simlifespan"]
    simseed = params_sim["simseed"]
    length = simlifespan - startage + 1

    # Set the seed
    np.random.seed(simseed)

    # set up data grids
    c = np.full((length, numSims), np.nan)  # consumption
    h = np.full((length, numSims), np.nan)  # labor supply
    m = np.full((length, numSims), np.nan)  # market resources
    a = np.full((length, numSims), np.nan)  # assets
    k = np.full((length, numSims), np.nan)  # human capital
    y = np.full((length, numSims), np.nan)  # human capital

    # parameters for asset and k distributions
    A_mean      = params_sim['A_mean']
    V_A         = params_sim['V_A']
    HCt0_mean   = params_sim['HCt0_mean']
    HCt0_sd     = params_sim['HCt0_sd']
    sigma       = params_sim['sigma']
    growth      = params_sim['growth']
    # get shocks
    mu          = -(1/2)*sigma**2 # mean of wage shock
    e1 = np.random.normal(mu, sigma, size=(length, numSims))
    e1 = np.exp(e1)

    # draw the initial conditions for assets
    sdA = V_A/4
    A_lb = (0- A_mean) / sdA
    initial_assets = stats.truncnorm.rvs(a=A_lb, b=np.inf, loc=A_mean, scale=sdA, size=numSims)
    # initial_assets = np.random.normal(HCt0_mean, HCt0_sd, numSims)
    # np.mean(initial_assets)
    # Now do wages
    # initial_assets = np.random.normal(A_mean, V_A, numSims)
    HC_lb = (.02 - HCt0_mean) / HCt0_sd
    initial_HC = stats.truncnorm.rvs(a=HC_lb, b=np.inf, loc=HCt0_mean, scale=HCt0_sd, size=numSims)
    initial_HC = np.clip(initial_HC/1e2, .005, np.max(endk[:,0]))

    # Now that we've got our interpolators, simulate all of our paths
    k[0, :] = initial_HC
    # k[0, :] = 1e-5

    for ixt in range(length):
        # print(ixt)
        # ixt = 0

        # calculate market resources
        if ixt == 0:
            m[ixt, :] = initial_assets + (k[ixt, :]*(2040/1e2))

        # calculate consumption
        minterp = policyM[:,ixt].flatten()
        kinterp = endk[:, ixt].flatten()
        cinterp = policyC[:, :, ixt]
        hinterp = policyH[:, :, ixt]

        # interpn((minterp,kinterp), hinterp,(1,.02), bounds_error = False)

        # interpolate:
        c[ixt, :] = interpn((minterp,kinterp), cinterp,(m[ixt,:],k[ixt,:]), bounds_error = False)
        
        if ixt<(length-1):
            h[ixt, :] = interpn((minterp,kinterp), hinterp,(m[ixt,:],k[ixt,:]), bounds_error = False)
            y[ixt,:] = k[ixt, :]*h[ixt,:]

        # calculate end of period savings
        a[ixt, :] = m[ixt, :] - c[ixt, :]

        # next period human capital (just constant at this point)
        if ixt<(length-1):
            # which determine next period market resources
            m[ixt+1, :] = a[ixt, :]*R + (k[ixt, :]*h[ixt,:])
            # draw a shock
            k[ixt+1, :] = k[ixt, :] *(1+growth[ixt])* e1[ixt, :]
            k[ixt+1, :] = np.clip(k[ixt+1, :], 0, np.max(endk[:,ixt+1]))

    # want to output as a nice data frame
    # Now want to format as panel data
    age = np.tile(np.arange(startage, simlifespan+1), numSims)  # Assuming 46 time periods
    ID =  np.repeat(np.arange(1, numSims+1), length)  # Assuming 1000 individuals

    # flatten variables
    c_long = c.flatten(order='F')
    a_long = a.flatten(order='F')
    k_long = k.flatten(order='F')
    h_long = h.flatten(order='F')
    y_long = y.flatten(order='F')

    # create a data frame
    data = {
    'age': age,
    'ID': ID,
    'c': c_long,
    'a': a_long,
    'k': k_long,
    'h': h_long,
    'y': y_long
    }
    data = pd.DataFrame(data)

    # np.mean(a,1)
    # np.mean(c,1)
    return data

    