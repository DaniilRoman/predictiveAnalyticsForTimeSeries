from scipy.optimize import minimize

from src.mlModels.CrossValidation import timeseriesCVscore
from src.classicApproach.HoltWinters import HoltWinters
from src.mlModels.MLModels import *


def runTripleExponentialSmoothing(data):
    data = data[:-20] # leave some data for testing

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
    print(alpha_final, beta_final, gamma_final)

    # ...and train the model with them, forecasting for the next 50 hours
    model = HoltWinters(data, slen = 24,
                        alpha = alpha_final,
                        beta = beta_final,
                        gamma = gamma_final,
                        n_preds = 50, scaling_factor = 3)
    model.triple_exponential_smoothing()
    return model

def runLegReg():
    mlModels = MLModels(pd.DataFrame(), "time")
    mlModels.runLenRegAndPlot()

