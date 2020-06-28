import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def lr_fit(data_train_x,data_train_y,save=False):
    x = pd.read_csv(data_train_x)
    y = pd.read_csv(data_train_y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters)
    clf.fit(x_train, y_train)
    print("训练集预测结果：")
    print(clf.best_score_(x_train))
    print("测试集预测结果：")
    print(clf.best_score_(x_test))
    if save:
        pickle.dump(clf.best_estimator_, open(r'lr.pkl', 'wb'))
