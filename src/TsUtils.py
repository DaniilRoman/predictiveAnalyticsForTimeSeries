import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import numpy as np


waves = pd.read_csv('../../data/waves_price.csv', index_col=['Date'], parse_dates=['Date'])
candy = pd.read_csv('../../data/candy_production.csv', index_col=['observation_date'], parse_dates=['observation_date'])
slice_waves = pd.DataFrame(waves['Close'], index=waves.index, columns=['Close'])

def plot(values, name):
    plt.figure(figsize=(12, 6))
    plt.plot(values)
    plt.title(name + ' watched')
    plt.grid(True)
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

