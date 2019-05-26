import pandas as pd

from src.Config import candy_file, waves_file
from src.SARIMA.SARIMA import SARIMA
from src.SARIMA.SARIMA_plot import plotSARIMA, plotWithAutocorrelation, plotDoubleDiff, plotDiff


def run_SARIMA(filePath, columnName):
    data = pd.read_csv(filePath)
    model = SARIMA(data, columnName)
    model.initParamList()
    result_table = model.optimizeSARIMA()
    # result_table = {"parameters": [[3, 3, 0, 1]]}
    model.runBestModel(result_table)

    #time to ploting
    plotSARIMA(model.series, model.bestModel, model.s, model.d, 4)
    plotWithAutocorrelation(model.series, 12)
    plotDiff(model.series)
    plotDoubleDiff(model.series)


# run_SARIMA('../../data/notebooks/network.csv', 'date')
# run_SARIMA("../../data/notebooks/data.csv")

data = pd.read_csv('../../data/notebooks/network.csv')


