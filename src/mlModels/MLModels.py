import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import numpy as np                               # vectors and matrices

from sklearn.preprocessing import StandardScaler
import seaborn as sns                            # more plots

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from src.mlModels.ExtractFeatures import extractFeatures
from src.TsUtils import mean_absolute_percentage_error
from src.mlModels.SelectFeatures import selectFeatures


class MLModels:
    def __init__(self, pathToFile, timeStr):
        self.timeStr = timeStr
        self.pathToFile = pathToFile
        self.tscv = TimeSeriesSplit(n_splits=5)

    def prepareData(self):
        train = pd.read_csv('../../data/waves_price_train.csv')
        test = pd.read_csv('../../data/waves_price_test.csv')
        self.X_train = train.drop('y', axis=1)
        self.y_train = train.y

        self.X_test = test.drop('y', axis=1)
        self.y_test = test.y

        return self.X_train, self.X_test, self.y_train, self.y_test

    def plotModelResults(self, model, X_train, X_test, plot_intervals=False, plot_anomalies=False):
        """
            Plots modelled vs fact values, prediction intervals and anomalies

        """
        # if X_train == None: X_train = self.X_train
        # if X_test == None: X_test= self.X_test
        self.prediction = model.predict(X_test)

        plt.figure(figsize=(15, 7))
        plt.plot(self.prediction, "g", label="prediction", linewidth=2.0)
        plt.plot(self.y_test, label="actual", linewidth=2.0)

        if plot_intervals:
            cv = cross_val_score(model, X_train, self.y_train,
                                 cv=self.tscv,
                                 scoring="neg_mean_absolute_error")
            mae = cv.mean() * (-1)
            deviation = cv.std()

            scale = 1.96
            lower = self.prediction - (mae + scale * deviation)
            upper = self.prediction + (mae + scale * deviation)

            plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
            plt.plot(upper, "r--", alpha=0.5)

            if plot_anomalies:
                anomalies = np.array([np.NaN] * len(self.y_test))
                anomalies[self.y_test < lower] = self.y_test[self.y_test < lower]
                anomalies[self.y_test > upper] = self.y_test[self.y_test > upper]
                plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        error = mean_absolute_percentage_error(self.prediction, self.y_test)
        plt.title("Mean absolute percentage error {0:.2f}%".format(error))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("../../data/images/ML/LinReg/" + "waves_Predicted.png")


    def plotCoefficients(self, model):
        """
            Plots sorted coefficient values of the model
        """

        coefs = pd.DataFrame(model.coef_, self.X_train.columns)
        coefs.columns = ["coef"]
        coefs["abs"] = coefs.coef.apply(np.abs)
        coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

        plt.figure(figsize=(15, 7))
        coefs.coef.plot(kind='bar')
        plt.grid(True, axis='y')
        plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
        plt.savefig("../../data/images/ML/LinReg/" + "waves_Coef.png")

    def runLenRegAndPlot(self):
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = self.prepareData()

        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(self.X_train_scaled, y_train)

        self.plotModelResults(lr, X_train=self.X_train_scaled, X_test=self.X_test_scaled, plot_intervals=True, plot_anomalies=True)
        self.plotCoefficients(lr)
        plt.savefig("../../data/images/ML/LinReg/" + "waves_LenReg.png")
        plt.show()

    def plotCorrelMatrix(X_train):
        plt.figure(figsize=(10, 8))
        sns.heatmap(X_train.corr())

    def applyRidgeCVRegularization(self):
        self.applyCVRegularization(RidgeCV)

    def applyLassoCVRegularization(self):
        self.applyCVRegularization(LassoCV)

    def applyCVRegularization(self, method):
        regularizationModel = method(cv=self.tscv)
        regularizationModel.fit(self.X_train_scaled, self.y_train)

        self.plotModelResults(regularizationModel,
                         X_train=self.X_train_scaled,
                         X_test=self.X_test_scaled,
                         plot_intervals=True, plot_anomalies=True)
        self.plotCoefficients(regularizationModel)
        plt.savefig("../../data/images/ML/LinReg/" + "waves_Regularization.png")

    def applyXGBoost(self):
        xgb = XGBRegressor()
        xgb.fit(self.X_train_scaled, self.y_train)
        return xgb

    def extractFeatures(fileName, timeStr):
        features, pathToFeatures = extractFeatures(fileName, timeStr)
        train, test, pathToTrain, pathToTest = selectFeatures(features, fileName)
        return train, test