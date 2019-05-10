import pandas as pd

class DataHolder:
    def __init__(self):
        pass

    def getSeries(self, left, right):
        series = pd.read_csv('../../data/notebooks/network.csv')
        oldSeries = series.r_asn
        series = oldSeries[left:right]
        return series, oldSeries