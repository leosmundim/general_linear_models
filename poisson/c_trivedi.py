# Função Teste_Cameron_Trivedi
def c_trivedi(var_y, fitted_values):
    
    from scipy.stats import chi2
    
    y_star = ((var_y - fitted_values)**2 - var_y) / fitted_values
    df = pd.DataFrame([y_star, var_y, fitted_values]).T
    
    modelo_aux = sm.OLS.from_formula(formula='y_star ~ 0 + fitted_values', data=df).fit()
    
    return print(f""" Teste de Superdispersão - Cameron e Trivedi 1990
    
    t-test score: {round(modelo_aux.tvalues.item(),4)}, p-value = {modelo_aux.f_pvalue.round(4)}
    **Se p-value < 0.05, existe superdispersão**""")