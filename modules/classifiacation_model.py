import os
from math import sqrt

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier


class ClassificationModel:

    def __init__(self, name, X_train, y_train, X_test, y_test):
        self._name = name
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

    def Evaluation_matrix(self, y_test, y_pred, type):

        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i in range(len(y_test)):
            if y_test[i] == 0:
                if y_pred[i] == 0:
                    TN+=1
                else:
                    FP+=1
            else:
                if y_pred[i] == 0:
                    FN+=1
                else:
                    TP+=1
        accuracy_score = (TP+TN)/(TP+FN+FP+TN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = (2*precision*recall)/(precision+recall)
        if type == 'accuracy_score':
            return accuracy_score
        elif type == 'f1_score':
            return f1_score

    def build_model(self, model):

        print('Progress: building '+str(model)+' model')
        if model == 'randomforest':
            clf = RandomForestClassifier(min_samples_split=5, min_samples_leaf= 5,max_depth=10,bootstrap= True, random_state= 42, n_estimators=200)
        elif model == 'xgb':
            clf = XGBClassifier(learning_rate = 0.08 ,booster='gbtree', min_child_weight = 13, max_depth = 12, random_state=42)
        elif model == 'lgbm':
            clf = LGBMClassifier()
        
        clf.fit(self._X_train, self._y_train)

        y_pred_train = clf.predict(self._X_train)
        y_pred = clf.predict(self._X_test)

        accuracy_score_train = self.Evaluation_matrix(self._y_train, y_pred_train, 'accuracy_score')
        f1_score_train = self.Evaluation_matrix(self._y_train, y_pred_train, 'f1_score')
        accuracy_score = self.Evaluation_matrix(self._y_test, y_pred, 'accuracy_score')
        f1_score = self.Evaluation_matrix(self._y_test, y_pred, 'f1_score')

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/model'):
            os.mkdir(path+'/model')

        if not os.path.isdir(path+'/model/'+str(model)):
            os.mkdir(path+'/model/'+str(model))

        print('Progress: exporting result of '+str(model)+' model')

        df = pd.DataFrame(columns = ['train','test'], index = ['accuracy','f1_score'])
        df['train'] = [accuracy_score_train, f1_score_train]
        df['test'] = [accuracy_score, f1_score]
        dfi.export(df, path + '/model/'+str(model)+'/evaluation.png', max_cols = -1, max_rows = -1)
