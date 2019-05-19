import pandas as pd


def getSeries(left, right):
    # series = pd.read_csv('../../data/notebooks/lte.csv')
    # oldSeries = series.Traffic
    series = pd.read_csv('../../data/notebooks/network.csv')
    oldSeries = series.r_asn
    series = oldSeries[left:right]
    return series, oldSeries
