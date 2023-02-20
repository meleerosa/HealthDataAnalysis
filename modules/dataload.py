import os
import warnings

import dataframe_image as dfi
import pandas as pd

warnings.filterwarnings('ignore')
from math import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rc('font', family='Malgun Gothic') # For Windows

class Dataload:
    def __init__(self, name):
        self._name = name

    def load_data(self, data = None):
        self._data = data

        print('Progress: loading data')

        if self._name == 'raw':
            self._data = pd.read_csv('./data/data.csv',encoding='utf-8')

        return self._data

    def data_info(self):
        dataFeatures = []
        dataType     = []
        null         = []
        nullPCT      = []
        unique       = []
        minValue     = []
        maxValue     = []
        uniqueSample = []

        for item in list(self._data):
            dataFeatures.append(item)

        for item in dataFeatures:
            dataType.append(self._data[item].dtype.name)
            
        for item in dataFeatures:
            null.append(len(self._data[self._data[item].isnull() == True]))

        for item in dataFeatures:
            nullPCT.append(round(len(self._data[self._data[item].isnull() == True])/len(self._data[item])*100,2))
            
        for item in dataFeatures:
            minValue.append(self._data[item].min())

        for item in dataFeatures:
            maxValue.append(self._data[item].max())

        for item in dataFeatures:
            unique.append(self._data[item].nunique())

        for item in dataFeatures:
            uniqueSample.append(self._data[item].unique()[0:2])

        self._data_info = pd.DataFrame({
            'dataFeatures' : dataFeatures,
            'dataType' : dataType,
            'null' : null,
            'nullPCT':nullPCT,
            'unique' : unique,
            'minValue' : minValue,
            'maxValue' : maxValue,
            'uniqueSample':uniqueSample
        })

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        print('Progress: exporting data information')

        dfi.export(self._data_info, path + '/data_info.png', max_cols = -1, max_rows = -1)
    
    def data_describe(self):
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        print('Progress: exporting data describe')

        dfi.export(self._data.describe().T, path + '/data_describe.png', max_cols = -1, max_rows = -1)

    def plot_distribution(self):
        count = 1

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/distribution'):
            os.mkdir(path+'/distribution')

        for col in self._data.columns:
            
            print('Progress: saving distribution plot files({}{}{}) in directory {}'.format(count,'/',len(self._data.columns),path+'/distribution'))

            count += 1
            coef = 1.5
            iqr = self._data[col].quantile(.75) - self._data[col].quantile(.25)
            u = self._data[col].quantile(.75) + coef*iqr
            l = self._data[col].quantile(.25) - coef*iqr

            plt.figure(figsize=(18,5))
            plt.title(col)
            plt.subplot(1,3,1)
 
            sns.boxplot(self._data[col])
            plt.plot([-1,1],[u,u],c = 'red', linestyle = '--', label = 'upper')
            plt.plot([-1,1],[l,l],c = 'red', linestyle = '--', label = 'lower')
            plt.legend()

            plt.subplot(1,3,2)
            sns.distplot(self._data[col], bins = 100) 
            plt.title(col)

            plt.subplot(1,3,3)
            sns.violinplot(self._data[col]) 
            plt.title(col)
            plt.plot([-1,1],[u,u],c = 'red', linestyle = '--', label = 'upper')
            plt.plot([-1,1],[l,l],c = 'red', linestyle = '--', label = 'lower')
            plt.legend()
            plt.savefig(path+'/distribution/'+col+'.png')
        plt.close()

    def show_vif(self):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(self._data.values, i) for i in range(self._data.shape[1])]
        vif["features"] = self._data.columns 

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        print('Progress: exporting vif data')
        
        dfi.export(vif, path + '/vif.png', max_cols = -1, max_rows = -1)


    def countplot(self, col):
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/countplot'):
            os.mkdir(path+'/countplot')

        plt.figure(figsize = (12,10))
        sns.countplot(x = col, data = self._data)
        plt.savefig(path+'/countplot/countplot.png')

        print('Progress: saving countplot files in directory {}'.format(path+'/countplot'))

        plt.close()

    def plot_difference(self, col):

        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/difference_distribution'):
            os.mkdir(path+'/difference_distribution')

        for i in range(len(self._data.columns)):
            print('Progress: saving distribution of difference plot files({}{}{}) in directory {}'.format(i,'/',len(self._data.columns),path+'/difference_distribution'))
            sns.boxplot(x = col, y = self._data.columns[i], data = self._data)
            plt.title('정상그룹 vs 이상그룹 - '+str(self._data.columns[i]))
            plt.savefig(path+'/difference_distribution/'+str(self._data.columns[i])+'.png')
            plt.close()

    # 차이 검정
    def difference_test(self, col):
        normal = self._data[self._data[col] == 1]
        abnormal = self._data[self._data[col] != 1]
        diff_True = []
        diff_False = []
        for col in self._data.columns:
            t_test = stats.ttest_ind(normal[col],abnormal[col])
            if t_test[1] < 0.05:
                diff_True.append(col)
            else:
                diff_False.append(col)
        print('정상 그룹과 이상 그룹 간 차이가 있는 변수:',diff_True)
        print('\n')
        print('정상 그룹과 이상 그룹 간 차이가 없는 변수:',diff_False)
        return diff_True