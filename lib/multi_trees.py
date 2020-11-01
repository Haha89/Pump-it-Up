# -*- coding: utf-8 -*-

import pandas as pd
from utils import preprocessing_data, lr_decay
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from numpy import argmax, zeros, array
import warnings
warnings.filterwarnings("ignore")

PATH_DATA = "../data/"
NB_FOLDS = 8

if __name__ == "__main__":
    # create_village_region_files(PATH_DATA)
    train_lab = pd.read_csv(PATH_DATA + 'train_labels.csv')["status_group"]
    train_val = pd.read_csv(PATH_DATA + 'train_values.csv')

    train_set = pd.concat([train_val, train_lab], axis=1)
    train_set = preprocessing_data(train_set)

    test_val = pd.read_csv(PATH_DATA + 'test_values.csv')
    test_val = preprocessing_data(test_val, test=True)

    params = {'colsample_bytree': 0.6765, 'max_bin': 2013,
              'min_child_samples': 122, 'min_child_weight': 10.0,
              'num_leaves': 442, 'reg_alpha': 0.1, 'reg_lambda': 10,
              'subsample': 0.28234, "random_state": 314,
              'n_jobs': 4, 'n_estimators': 5000, "max_depth": 30,
              'learning_rate': 0.0085, 'num_iterations': 2150}

    kf = StratifiedKFold(n_splits=NB_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_set)

    dataset_target = train_set['status_group']
    dataset_values = train_set.drop('status_group', axis=1)

    for col in dataset_values.columns:
        if col not in test_val.columns:
            test_val[col] = 0
    test_val = test_val[dataset_values.columns]
    preds = zeros((len(test_val), 3), dtype=float)

    i = 0
    for train_index, val_index in kf.split(dataset_values, dataset_target):

        i += 1
        val_target = dataset_target.iloc[val_index]
        val_val = dataset_values.iloc[val_index]

        train_target = dataset_target.iloc[train_index]
        train_val = dataset_values.iloc[train_index]

        # add extra columns missings in the test set
        for col in train_val.columns:
            if col not in val_val.columns:
                val_val[col] = 0
        val_val = val_val[train_val.columns]

        clf = lgb.LGBMClassifier()
        clf.set_params(**params)
        clf.fit(train_val.values, train_target,
                callbacks=[lgb.reset_parameter(learning_rate=lr_decay)])
        y_pred = clf.predict(val_val)
        correct = len([i for i, j in zip(y_pred, val_target) if i == j])
        ratio = correct/len(val_target)*100
        print(f"Accuracy f-{i}: {ratio:.3f}")
        preds += array(clf.predict_proba(test_val))

    preds = argmax(preds, axis=1)
    submission = pd.read_csv(PATH_DATA + 'SubmissionFormat.csv')
    labels = ["non functional", "functional needs repair", "functional"]
    submission['status_group'] = list(map(lambda x: labels[x], preds))
    submission.to_csv(PATH_DATA + "submission.csv", index=False)
