from src.classicApproach.RunScripts import runTripleExponentialSmoothing
from src.TsUtils import *

import seaborn as sns   # more plots
import warnings  # `do not disturbe` mode

sns.set()
warnings.filterwarnings('ignore')





model = runTripleExponentialSmoothing(candy.IPG3113N)
model.plotHoltWinters(candy.IPG3113N, plot_intervals=True, plot_anomalies=True)



