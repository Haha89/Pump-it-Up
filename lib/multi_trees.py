# -*- coding: utf-8 -*-


import pandas as pd
from utils import preprocessing_data, lr_decay
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from numpy import argmax, zeros, array
import warnings
warnings.filterwarnings("ignore")

PATH_DATA = "../data/"
NB_FOLDS = 5
SUBMIT = False

if __name__ == "__main__":
    # create_village_region_files(PATH_DATA)
    train_lab = pd.read_csv(PATH_DATA + 'train_labels.csv')["status_group"]
    train_val = pd.read_csv(PATH_DATA + 'train_values.csv')

    train_set = pd.concat([train_val, train_lab], axis=1)
    train_set = preprocessing_data(train_set)

    params = {'colsample_bytree': 0.6765, 'max_bin': 2013,
              'min_child_samples': 122, 'min_child_weight': 10.0,
              'num_leaves': 442, 'reg_alpha': 0.1, 'reg_lambda': 10,
              'subsample': 0.28234, "random_state": 314,
              'metric': 'None', 'n_jobs': 4, 'n_estimators': 5000,
              'learning_rate': 0.0085, 'num_iterations': 2150}

    kf = StratifiedKFold(n_splits=NB_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_set)

    dataset_target = train_set['status_group']
    dataset_values = train_set.drop('status_group', axis=1)

    i = 0
    for train_index, test_index in kf.split(dataset_values, dataset_target):

        i += 1
        test_target = dataset_target.iloc[test_index]
        test_val = dataset_values.iloc[test_index]

        train_target = dataset_target.iloc[train_index]
        train_val = dataset_values.iloc[train_index]

        # add extra columns missings in the test set
        for col in train_val.columns:
            if col not in test_val.columns:
                train_val = train_val.drop(col, axis=1)
        test_val = test_val[train_val.columns]

        clf = lgb.LGBMClassifier()
        clf.set_params(**params)
        clf.fit(train_val.values, train_target,
                callbacks=[lgb.reset_parameter(learning_rate=lr_decay)])
        y_pred = clf.predict(test_val)

        correct = len([i for i, j in zip(y_pred, test_target) if i == j])
        ratio = correct/len(test_target)*100
        print(f"Accuracy f-{i}: {correct}/{len(test_target)} ({ratio:.2f}%)")
        clf.booster_.save_model(f'{PATH_DATA}model/tree_{i}.txt')

    # Submission
    if SUBMIT:
        test_val = pd.read_csv(PATH_DATA + 'test_values.csv')
        test_val = preprocessing_data(test_val, test=True)

        for col in dataset_values.columns:
            if col not in test_val.columns:
                test_val[col] = 0
        test_val = test_val[dataset_values.columns]

        preds = zeros((len(test_val), 3), dtype=float)
        for fold in range(1, NB_FOLDS+1):
            clf = lgb.Booster(model_file=PATH_DATA+f'model/tree_{fold}.txt')
            preds += array(clf.predict(test_val))

        preds = argmax(preds, axis=1)

        submission = pd.read_csv(PATH_DATA + 'SubmissionFormat.csv')
        labels = ["non functional", "functional needs repair", "functional"]
        submission['status_group'] = list(map(lambda x: labels[x], preds))
        submission.to_csv("../data/submission.csv", index=False)
