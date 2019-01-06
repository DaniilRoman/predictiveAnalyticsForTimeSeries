from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from SARIMA import SARIMA
from TsUtils import candy, waves
import pandas as pd

# sarima = SARIMA(waves)
# sarima.plotWithAutocorrelation(lags=5)

# series = pd.DataFrame(waves['Close'], index=waves.index, columns=['Close'])
#
# print(series.head())
# series = extract_features(series, column_id="Close")
# print(series.head())


from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
from tsfresh import extract_features
from tsfresh import extract_relevant_features

timeseries, y = load_robot_execution_failures()

print(timeseries.head())

extraction_settings = ComprehensiveFCParameters()

features_filtered_direct = extract_relevant_features(timeseries, y,
                                                     column_id='id', column_sort='time',
                                                     default_fc_parameters=extraction_settings)
