import warnings
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


#
# if __name__ == '__main__':

def oldApproachForPrepareData(d):
    columns = list(d.columns)
    columns.pop()
    columns.append('target')
    d.columns = columns

    y = d.target
    d.drop('target', axis=1, inplace=True)
    columns = list(d.columns)
    columns.pop()
    columns.append(0)
    d.columns = columns

    d = d.stack()

    d.index.rename(['id', 'time'], inplace=True)
    d = d.reset_index()
    return d

def newApproachForPrepareData(d, timeStr):
    time = d[timeStr]
    d.drop(timeStr, axis=1, inplace=True)
    d = getY(d)
    d['id'] = range(1, len(d)+1)
    d['time'] = time
    print(d.head())
    return d

def getY(data):
    series = pd.DataFrame(data.copy())
    newColumnNames = []
    newOldColumnNames = {}
    for i in range(len(series.columns)):
        key = "y" + str(i)
        newColumnNames.append(key)
        newOldColumnNames[key] = series.columns[i]
    series.columns = newColumnNames
    series = pd.DataFrame(series.y0)
    series.columns = ["y"]
    print(series.head())
    return series


def extractFeatures(fileName, columnWithDate):
    d = pd.read_csv(fileName+'.csv')
    print(d.head())

    #TODO for speed
    # d = d.loc[len(d)*0.8:]

    d = newApproachForPrepareData(d, columnWithDate)

    y = d['y']

    print(len(d))
    print(d.head())

    # doesn't work too well
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = extract_features(d, column_id="id", column_sort="time")

    f['y'] = y
    # f['id'] = range(0, len(f))

    print(f.head())

    impute(f)

    print(f.head())

    assert f.isnull().sum().sum() == 0

    fileToFeatures = fileName+'_features.csv'
    f.to_csv(fileToFeatures, index=None)
    return f, fileToFeatures