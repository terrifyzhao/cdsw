#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2020/06/28
# @Author : humaohai
# @File   : xgb.py
# @desc   : 

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def xgb_fit(data_train_x,data_train_y,save=False):
    x = pd.read_csv(data_train_x)
    y = pd.read_csv(data_train_y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    xgbt = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    )
    clf = GridSearchCV(xgbt, parameters)
    clf.fit(x_train, y_train)
    print("训练集预测结果：")
    print(clf.best_score_(x_train))
    print("测试集预测结果：")
    print(clf.best_score_(x_test))
    if save:
        pickle.dump(clf.best_estimator_, open(r'lr.pkl', 'wb'))


