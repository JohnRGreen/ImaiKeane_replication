def getPolicy(choice,state,minterp,kinterp,value,ub):
    # choice = [1.00000000e-04, 1.52156276e+03]
    # state  = [1, .02]
    # value = value[:,:,ixt+1]
    # state = [A_grid[ixa,ixt],k_grid[ixk,ixt]]
    # print(choice)
    mnow  = state[0]
    know  = state[1]

    thisc = choice[0]
    thish = choice[1]

    nextm = R*(mnow - thisc) + thish*know
    nextk = know*(1+growth)
    nextv = interpn((minterp,kinterp), value,(nextm,nextk), bounds_error=False,fill_value = ub)
    thisv = thisc**(1-gamma)/(1-gamma) - alpha*thish**eta + beta*nextv
    # v = interpn((nextm,nextk), value, (minterp,kinterp),fill_value = ub)
    # print(thisv)
    return -thisv