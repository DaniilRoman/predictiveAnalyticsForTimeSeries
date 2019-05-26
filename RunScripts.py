from scipy.optimize import minimize

from CrossValidation import timeseriesCVscore
from HoltWinters import HoltWinters


def run_triple_exponential_smoothing(data):
    # data = data[:-20] # leave some data for testing

    # initializing model parameters alpha, beta and gamma
    x = [0, 0, 0]

    # Minimizing the loss function
    opt = minimize(timeseriesCVscore, x0=x,
                   args=(data),
                   method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
                  )
    # , mean_squared_log_error

    # Take optimal values...
    alpha_final, beta_final, gamma_final = opt.x
    # alpha_final, beta_final, gamma_final = (0.025290320233789743, 0.0737327205106203, 0.0)
    # print("alpha_final: ",alpha_final)
    # print("beta_final: ", beta_final)
    # print("gamma_final: ", gamma_final)

    # print(alpha_final, beta_final, gamma_final)

    model = HoltWinters(data, slen = 46,
                        alpha = alpha_final,
                        beta = beta_final,
                        gamma = gamma_final,
                        n_preds = 50, scaling_factor = 3)
    model.triple_exponential_smoothing()
    return model

