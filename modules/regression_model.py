import os
from math import sqrt

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RegressionModel:

    def __init__(self, name, X_train, y_train, X_test, y_test):
        self._name = name
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

    def build_OLS_model(self):
        self._X_train = sm.add_constant(self._X_train)
        self._X_test = sm.add_constant(self._X_test)
        self._model = sm.OLS(self._y_train, self._X_train).fit()
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/model'):
            os.mkdir(path+'/model')

        if not os.path.isdir(path+'/model/OLS'):
            os.mkdir(path+'/model/OLS')

        print('Progress: building OLS model')

        for i in range(3):
            df = pd.DataFrame(self._model.summary().tables[i])
            dfi.export(df, path + '/model/OLS/model_summary{}.png'.format(str(i+1)), max_cols = -1, max_rows = -1)

        return self._model

    def build_XGBregressor_model(self):
        self._xgb_model = xgb.XGBRegressor(
        n_estimators = 200,
        reg_alpha = 0,
        reg_lambda = 1,
        booster = 'gbtree',
        learning_rate = 0.03,
        gamma = 0.1,
        subsample = 0.4,
        colsample_bytree = 1,
        max_depth = 7
        )

        print('Progress: building XGBregressor model')

        self._xgb_model.fit(self._X_train, self._y_train)

        plt.figure(figsize=(8,5))
        xgb.plot_importance(self._xgb_model)

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/model'):
            os.mkdir(path+'/model')

        if not os.path.isdir(path+'/model/XGBboost_regressor'):
            os.mkdir(path+'/model/XGBboost_regressor')

        plt.savefig(path+'/model/XGBboost_regressor/feature_importance.png')

        return self._xgb_model
    
    def evaluation_matrix(self, model, target_scaler):
        actual = target_scaler.inverse_transform(np.array(self._y_test).reshape(-1,1))
        predict = target_scaler.inverse_transform(np.array(model.predict(self._X_test)).reshape(-1,1))
        actual_train = target_scaler.inverse_transform(np.array(self._y_train).reshape(-1,1))
        predict_train = target_scaler.inverse_transform(np.array(model.predict(self._X_train)).reshape(-1,1))

        score_df = pd.DataFrame(columns = ['Train','Test'], index = ['MSE','RMSE','MAE'])

        score_df['Train'] = [
            mean_squared_error(actual_train, predict_train),
            sqrt(mean_squared_error(actual_train, predict_train)),
            mean_absolute_error(actual_train, predict_train)
            ]

        score_df['Test'] = [
            mean_squared_error(actual, predict),
            sqrt(mean_squared_error(actual, predict)),
            mean_absolute_error(actual, predict)
        ]

        path = './output/'+str(self._name)

        print('Progress: exporting Evaluation Matrix')
        
        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/model'):
            os.mkdir(path+'/model')

        if str(type(model)).split()[1] == "'statsmodels.regression.linear_model.RegressionResultsWrapper'>":            
            if not os.path.isdir(path+'/model/OLS'):
                os.mkdir(path+'/model/OLS')
            dfi.export(score_df, path + '/model/OLS/EvaluationMaxtrix.png', max_cols = -1, max_rows = -1)

        else:
            if not os.path.isdir(path+'/model/XGBboost_regressor'):
                os.mkdir(path+'/model/XGBboost_regressor')
            dfi.export(score_df, path + '/model/XGBboost_regressor/EvaluationMaxtrix.png', max_cols = -1, max_rows = -1)

    
    def show_prediction(self, model,scaler, scale):
        scale = scale
        actual = scaler.inverse_transform(np.array(self._y_test).reshape(-1,1))
        predict = scaler.inverse_transform(np.array(model.predict(self._X_test)).reshape(-1,1))

        plt.figure(figsize = (18,5))
        plt.plot(actual[:scale], label = 'actual')
        plt.plot(predict[:scale], label = 'predict')
        plt.ylabel('(혈청지오티)ALT')
        plt.legend()

        print('Progress: exporting Prediction')

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/model'):
            os.mkdir(path+'/model')

        if str(type(model)).split()[1] == "'statsmodels.regression.linear_model.RegressionResultsWrapper'>":            
            if not os.path.isdir(path+'/model/OLS'):
                os.mkdir(path+'/model/OLS')
            plt.savefig(path+'/model/OLS/prediction'+str(scale)+'.png')

        else:
            if not os.path.isdir(path+'/model/XGBboost_regressor'):
                os.mkdir(path+'/model/XGBboost_regressor')
            plt.savefig(path+'/model/XGBboost_regressor/prediction'+str(scale)+'.png')
