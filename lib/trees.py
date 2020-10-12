# -*- coding: utf-8 -*-

"""Testing with lightGBM """

import pandas as pd
from utils import preprocessing_data
import lightgbm as lgb

if __name__ == "__main__":

    PATH_DATA = "../data/"
    train_lab = pd.read_csv(PATH_DATA + 'train_labels.csv')["status_group"]
    train_val = pd.read_csv(PATH_DATA + 'train_values.csv')

    train_set = pd.concat([train_val, train_lab], axis=1)
    train_set = preprocessing_data(train_set)

    train_target = train_set['status_group'].values
    train = train_set.drop('status_group', axis=1)

    test_val = pd.read_csv(PATH_DATA + 'test_values.csv')
    test_val = preprocessing_data(test_val, test=True)

    # add extra columns missings in the test set
    for col in train.columns:
        if col not in test_val.columns:
            test_val[col] = 0
    test_val = test_val[train.columns]

    clf = lgb.LGBMClassifier(max_bin=1300, learning_rate=0.0085,
                             num_leaves=150, num_iterations=1150)
    clf.fit(train.values, train_target)

    y_pred = clf.predict(test_val)

    submission = pd.read_csv(PATH_DATA + 'SubmissionFormat.csv')
    labels = ["non functional", "functional needs repair", "functional"]
    submission['status_group'] = list(map(lambda x: labels[x], y_pred))
    submission.to_csv("../data/submission.csv", index=False)
