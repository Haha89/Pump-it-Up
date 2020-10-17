# -*- coding: utf-8 -*-

from numpy import power
import pandas as pd
from utils import preprocessing_data, create_village_region_files
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SUBMIT = False
PATH_DATA = "../data/"


def lr_decay(current_iter):
    return max(1e-3, 0.1 * power(.995, current_iter))


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

    clf = lgb.LGBMClassifier()
    clf.set_params(**params)

    if not SUBMIT:
        train_set, test_set = train_test_split(train_set, test_size=0.2)
        test_target = test_set['status_group'].values
        test_val = test_set.drop('status_group', axis=1)

    else:
        test_val = pd.read_csv(PATH_DATA + 'test_values.csv')
        test_val = preprocessing_data(test_val, test=True)

    train_target = train_set['status_group'].values
    train = train_set.drop('status_group', axis=1)

    # add extra columns missings in the test set
    for col in train.columns:
        if col not in test_val.columns:
            train = train.drop(col, axis=1)
    test_val = test_val[train.columns]

    clf.fit(train.values, train_target,
            callbacks=[lgb.reset_parameter(learning_rate=lr_decay)])
    y_pred = clf.predict(test_val)

    if not SUBMIT:
        correct = len([i for i, j in zip(y_pred, test_target) if i == j])
        ratio = correct/len(test_target)*100
        print(f"Accuracy: {correct}/{len(test_target)} ({ratio}%)")
        print(confusion_matrix(test_target, y_pred))
        feat_imp = pd.Series(clf.feature_importances_, index=train.columns)
        feat_imp.nlargest(30).plot(kind='barh', figsize=(10, 20))

    else:
        submission = pd.read_csv(PATH_DATA + 'SubmissionFormat.csv')
        labels = ["non functional", "functional needs repair", "functional"]
        submission['status_group'] = list(map(lambda x: labels[x], y_pred))
        submission.to_csv("../data/submission.csv", index=False)
