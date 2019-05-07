from src.Config import waves_file, candy_file
import pandas as pd

from src.classicApproach.HoltWinters import HoltWinters


def run_HoltWinters(data):
    # data = data[:-2]  # leave some data for testing

    model = HoltWinters(data, slen=4,
                        alpha=0.2,#0.1 (норм результат: 0,2)
                        beta=0.6,#0.2 (норм результат: 0,6)
                        gamma=0.8,#0.3 (норм результат: 0,8)
                        n_preds=4, scaling_factor=3)#scalar = 3
    model.triple_exponential_smoothing()
    model.plotHoltWinters(data, plot_intervals=False, plot_anomalies=False)
    # model.plotPredictedDeviation()

def getData(filePath, columnName = None):
    data = pd.read_csv(filePath)
    series = pd.DataFrame(data.copy())
    if (columnName == None):
        newColumnNames = []
        newOldColumnNames = {}
        for i in range(len(series.columns)):
            key = "y" + str(i)
            newColumnNames.append(key)
            newOldColumnNames[key] = series.columns[i]
        series.columns = newColumnNames
        series = series.y1
    else:
        series = series[columnName]
    return series

# run_HoltWinters(getData("../../"+candy_file))
run_HoltWinters(getData("../../"+waves_file))
# run_HoltWinters(getData("../../data/notebooks/data.csv"))

