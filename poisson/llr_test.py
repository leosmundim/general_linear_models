from scipy.stats import chi2

def llr_test (model_ref, model_test):
    
    if model_ref == model_test:

        stats_chi2 = -2*(model_ref.llnull - model_test.llf)
        
        return print(f"""LogLike Modelo Nulo: {model_ref.llnull:.2f}
LogLike Modelo: {model_test.llf:.2f}
Estatistica Chi2: {stats_chi2:.2f}
P-value Chi2: {1-chi2.cdf(stats_chi2, df=model_test.df_model)}""")
            
    else:
        
        stats_chi2 = -2*(model_ref.llf - model_test.llf)
        
        return print(f"""LogLike Modelo Ref: {model_ref.llf:.2f}
LogLike Modelo: {model_test.llf:.2f} 
Estatistica Chi2: {stats_chi2:.2f}
P-value Chi2: {1-chi2.cdf(stats_chi2, df=model_test.df_model)}""")