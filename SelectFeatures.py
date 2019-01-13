import pandas as pd
from tsfresh import extract_features, select_features

from Config import *

#

# workaround for multiprocessing on windows
# if __name__ == '__main__':
def selectFeatures(pathToFile):
    print("loading {}".format(pathToFile))
    features = pd.read_csv(pathToFile)

    print(features.head())
    print(features.y.head())

    validation_split_i = int(len(features)*0.8)

    train_x = features.iloc[:validation_split_i].drop('y', axis=1)
    test_x = features.iloc[validation_split_i:].drop('y', axis=1)

    train_y = features.iloc[:validation_split_i].y
    test_y = features.iloc[validation_split_i:].y

    print("selecting features...")
    print(train_x)
    for i in train_x.columns:
        print(i)
    print(train_y)
    print('indexxxxxxxxxxx')
    print(train_x.index)
    print(type(train_x))
    print(type(train_y))
    train_features_selected = select_features(train_x, train_y, fdr_level=fdr_level, chunksize=100, ml_task='regression')

    print(train_features_selected.head())

    print("selected {} features.".format(len(train_features_selected.columns)))

    train = train_features_selected.copy()
    train['y'] = train_y

    test = test_x[train_features_selected.columns].copy()
    test['y'] = test_y


    print("saving {}".format(train_file))
    train.to_csv('data/candy_prodaction_train.csv', index=None)

    print("saving {}".format(test_file))
    test.to_csv('data/candy_prodaction_test.csv', index=None)