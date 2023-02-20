import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class Preprocess:
    def __init__(self,data):
        self._data = data

    def select_columns(self, name , columns):
        self._name = name
        self._data = self._data[columns]
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        with open(path+"/columns.pkl","wb") as f:
            print('Progress: exporting column information')
            pickle.dump(columns, f)
    

    def dropna(self):
        self._data = self._data.dropna()


    def drop_anomalies(self):
        # 특별한 이상치 처리(전체 dataset에 적용)

        if '청력(우)' in self._data.columns:
            idx = self._data[self._data['청력(우)'] == 3].index
            self._data.drop(index = idx, inplace= True)

        if '청력(좌)' in self._data.columns:
            idx = self._data[self._data['청력(좌)'] == 3].index
            self._data.drop(index = idx, inplace= True)

        if ('시력(우)' in self._data.columns):
            self._data.loc[(self._data['시력(우)'] == 9.9),'시력(우)'] = 0
        
        if '시력(좌)' in self._data.columns:
            self._data.loc[(self._data['시력(좌)'] == 9.9),'시력(좌)'] = 0

        if '허리둘레' in self._data.columns:
            idx = self._data[(self._data['허리둘레'] == 999.0) | (self._data['허리둘레'] == 680.0)].index
            self._data.drop(index = idx, inplace= True)
            
        self._data = self._data.reset_index(drop = True)
        # if '요단백' in self._data.columns:
        #     self._data.loc[(self._data['요단백'] == 1),'요단백'] = 0
        #     self._data.loc[(self._data['요단백'] != 0),'요단백'] = 1


    def split_dataset(self, test_size = 0.2, stratify = None):
        if stratify != None: 
            train, test = train_test_split(
                self._data, test_size=test_size, random_state=42, shuffle = True, stratify=self._data[stratify]
                )
        else:
            train, test = train_test_split(self._data, 
                                            test_size=test_size, 
                                            random_state=42, 
                                            shuffle = True, 
                                        )

        return train, test


    def drop_outliers(self, train, coef):

        # 통계적 이상치 처리(train set에만 적용)

        outlier_list = ['체중(5Kg 단위)', '허리둘레', '시력(좌)',
        '시력(우)','수축기 혈압','이완기 혈압','식전혈당(공복혈당)', '총 콜레스테롤',
        '트리글리세라이드', 'HDL 콜레스테롤', 'LDL 콜레스테롤', '혈색소', '혈청크레아티닌',
        '(혈청지오티)AST','감마 지티피']

        for col in train.columns :
            if col in outlier_list:
                iqr = train[col].quantile(.75)-train[col].quantile(.25)
                u = train[col].quantile(.75) + coef*iqr
                l = train[col].quantile(.25) - coef*iqr
                idx = train[(train[col] > u) | (train[col] < l)].index
                train = train.drop(index = idx)

        train = train.reset_index(drop = True)

        return train

    def scale_dataset(self, X_train, y_train, X_test, y_test, scale_type = 'minmax'):
        if scale_type == 'minmax':
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()

        elif scale_type == 'robust':
            feature_scaler = RobustScaler()
            target_scaler = RobustScaler()
        
        elif scale_type == 'standard':
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()

        for col in X_train.columns:
            X_train[col] = feature_scaler.fit_transform(X_train[col].values.reshape(-1,1))
            X_test[col] = feature_scaler.transform(X_test[col].values.reshape(-1,1))
        
        y_train = target_scaler.fit_transform(np.array(y_train).reshape(-1,1))
        y_test = target_scaler.transform(np.array(y_test).reshape(-1,1))

        return feature_scaler, target_scaler, X_train, y_train, X_test, y_test

    def make_Xy(self, train, test, y):
        X_train = train.drop(columns = [y])
        y_train = train[y]
        X_test = test.drop(columns = [y])
        y_test = test[y]
        
        return X_train, y_train, X_test, y_test

    def save_data(self):
        self._data.to_csv('./data/saved_data.csv', index = False)

    def clf_encoding(self):
        self._data.loc[(self._data['요단백'] == 1),'요단백'] = 0
        self._data.loc[(self._data['요단백'] != 0),'요단백'] = 1

    def oversampling(self, X_train, y_train):
        print('Progress: sampling model')
        sm = BorderlineSMOTE(random_state= 42)
        X_train_res, y_train_res = sm.fit_resample(X_train,y_train)

        return X_train_res, y_train_res

    def undersampling(self, X_train, y_train, type = 'random', random_state = 42):
        print('Progress: sampling data')
        if type == 'tomek':
            tl = TomekLinks()
            X_train_un, y_train_un = tl.fit_resample(X_train, y_train)
        elif type == 'random':
            X_train_un, y_train_un = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)

        return X_train_un, y_train_un