from src.Config import candy_file, waves_file, pathToCandyFile
from src.mlModels.ExtractFeatures import extractFeatures
import pandas as pd

from src.mlModels.MLModels import MLModels

from src.mlModels.SelectFeatures import selectFeatures
#
# extractFeatures("waves_price", "Date")
#
#
# featuresFile = '../../data/waves_price_features.csv'
# features = pd.read_csv(featuresFile)
#
#
# selectFeatures(features, "waves_price")
#
model = MLModels(waves_file, 'Date')
#
model.runLenRegAndPlot()


model.applyRidgeCVRegularization()