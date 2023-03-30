#https://stackoverflow.com/questions/75727494/how-can-i-code-vuongs-statistical-test-in-python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer

def vuong_test(mod1, mod2, correction=True):
    '''
    mod1, mod2 - non-nested logitstic regression fit results from statsmodels
    '''
    # number of observations and check of models
    N = mod1.nobs
    N2 = mod2.nobs
    if N != N2:
        raise ValueError('Models do not have the same number of observations')
    # extract the log-likelihood for individual points with the models
    m1 = mod1.model.loglikeobs(mod1.params)
    m2 = mod2.model.loglikeobs(mod2.params)
    # point-wise log likelihood ratio
    m = m1 - m2
    # calculate the LR statistic
    LR = np.sum(m)
    # calculate the AIC and BIC correction factors -> these go to zero when df is same between models
    AICcor = mod1.df_model - mod2.df_model
    BICcor = np.log(N)*AICcor/2
    # calculate the omega^2 term
    omega2 = np.var(m, ddof=1)
    # calculate the Z statistic with and without corrections
    Zs = np.array([LR,LR-AICcor,LR-BICcor])
    Zs /= np.sqrt(N*omega2)
    # calculate the p-value
    ps = []
    msgs = []
    for Z in Zs:
        if Z>0:
            ps.append(1 - norm.cdf(Z))
            msgs.append('model 1 preferred over model 2')
        else:
            ps.append(norm.cdf(Z))
            msgs.append('model 2 preferred over model 1')
    # share information
    print('=== Vuong Test Results ===')
    labs = ['Uncorrected']
    if AICcor!=0:
        labs += ['AIC Corrected','BIC Corrected']
    for lab,msg,p,Z in zip(labs,msgs,ps,Zs):
        print('  -> '+lab)
        print('    -> '+msg)
        print('    -> Z: '+str(Z))
        print('    -> p: '+str(p))