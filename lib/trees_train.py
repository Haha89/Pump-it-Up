# -*- coding: utf-8 -*-

import pandas as pd
from utils import preprocessing_data
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    PATH_DATA = "../data/"

    train_lab = pd.read_csv(PATH_DATA + 'train_labels.csv')["status_group"]
    train_val = pd.read_csv(PATH_DATA + 'train_values.csv')

    train_set = pd.concat([train_val, train_lab], axis=1)
    train_set = preprocessing_data(train_set)

    train_set, test_set = train_test_split(train_set, test_size=0.2)

    train_target = train_set['status_group'].values
    train = train_set.drop('status_group', axis=1)

    test_target = test_set['status_group'].values
    test_val = test_set.drop('status_group', axis=1)

    clf = lgb.LGBMClassifier( max_bin=1300,
                             learning_rate=0.0085, num_leaves=150,
                             num_iterations=1150
                             # , objective="multiclass",
                             # boosting="rf", bagging_freq=1, bagging_fraction=0.8
                             )
    clf.fit(train.values, train_target)
    y_pred = clf.predict(test_val)
    
    print(confusion_matrix( test_target, y_pred))
    correct = 0
    for i, el in enumerate(test_target):
        if el == y_pred[i]:
            correct += 1
    ratio = correct/len(test_target)*100
    print(f"Accuracy: {correct}/{len(test_target)} ({ratio}%)")
