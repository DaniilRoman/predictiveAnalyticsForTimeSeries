from HoltWinters import *
from RunScripts import run_triple_exponential_smoothing
from TsUtils import *

import numpy as np                               # vectors and matrices
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
sns.set()

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')





model = run_triple_exponential_smoothing(candy.IPG3113N)
model.plotHoltWinters(candy.IPG3113N, plot_intervals=True, plot_anomalies=True)



