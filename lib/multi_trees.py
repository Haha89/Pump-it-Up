# -*- coding: utf-8 -*-


import pandas as pd
from utils import preprocessing_data, lr_decay
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

PATH_DATA = "../data/"
NB_FOLDS = 5


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
              'learning_rate': 0.0085, 'num_iterations': 100} #  2150

    kf = StratifiedKFold(n_splits=NB_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_set)

    train_target = train_set['status_group']
    train = train_set.drop('status_group', axis=1)
    
    print("Train shape", train.shape)
    print("Train target shape",train_target.shape)
    i = 0
    for train_index, test_index in kf.split(train, train_target):
        
        print(train_index.shape)
        print(test_index.shape)
        print("----")
        i += 1
        test_target = train_target.iloc[test_index]
        test_val = train.iloc[test_index]

        train_target = train_target.iloc[train_index]
        train = train.iloc[train_index]

        clf = lgb.LGBMClassifier()
        clf.set_params(**params)

        # add extra columns missings in the test set
        for col in train.columns:
            if col not in test_val.columns:
                train = train.drop(col, axis=1)
        test_val = test_val[train.columns]

        clf.fit(train.values, train_target,
                callbacks=[lgb.reset_parameter(learning_rate=lr_decay)])
        y_pred = clf.predict(test_val)

        correct = len([i for i, j in zip(y_pred, test_target) if i == j])
        ratio = correct/len(test_target)*100
        print(f"Accuracy fold {i}: {correct}/{len(test_target)} ({ratio}%)")
