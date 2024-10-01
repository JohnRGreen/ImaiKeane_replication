def getH(hchoice,thisk,alpha,eta,beta,interpolator,thisgrowth):
    # hchoice = 10
    # hchoice=sol.x
    # thisk=.01
    nextm = hchoice*thisk
    # print(nextm)
    # nextev = np.interp(nextm, minterp, evaluemat)
    nextk = thisk*(1+thisgrowth)
    # nextk = np.clip(nextk, 0, kinterp[kinterp.shape[0]-1])
    ecprime = interpolator((nextm,nextk))
    crit = hchoice-((beta/alpha)*thisk*ecprime)**(1/(eta-1))
    # nextev = interpn((minterp,kinterp), evaluemat,(nextm,nextk), method='linear', bounds_error=False, fill_value = ub)
    
    # nextev = interpn((minterp,kinterp), evaluemat,(minterp[1],kinterp[1]), bounds_error=False,fill_value = ub)
    # evaluemat[1,1]
    # interpn((minterp,kinterp), evaluemat,(minterp[13],kinterp[18]), bounds_error=False,fill_value = ub)
    # interpolate_evalue(minterp[13],kinterp[18])
    # interpolator((minterp[13],kinterp[18]))
    # evaluemat[13,18]
    # interpn((minterp,kinterp), evaluemat,(minterp[18],kinterp[13]), bounds_error=False,fill_value = ub)
    # interpolate_evalue(minterp[18],kinterp[13])
    # interpolator((minterp[18],kinterp[13]))
    # evaluemat[18,13]

    # v = interpn((nextm,nextk), value, (minterp,kinterp),fill_value = ub)
    # print(thisv)
    return  crit**2


    # def plot_interp(thism, thisK, minterp, kinterp, evalue, ub, gamma, alpha, eta, beta, interpolator):
    #     m_range = np.linspace(0,nextm+1, 1000)
    #     V = np.array([interpolator((m,thisK)) for m in m_range])
    #     plt.plot(m_range, V, label=f'K={thisK:.4f}')

    #     plt.xlabel('H')
    #     plt.ylabel('Value')
    #     plt.title(f'Objective Function for Different K Values (M=)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # # Call this function for a specific state point
    # plot_interp(thism, nextk, minterp, kinterp, evalue[:,:,ixt+1], ub, gamma, alpha, eta, beta, interpolator)
