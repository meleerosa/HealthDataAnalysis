import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rc('font', family='Malgun Gothic') # For Windows

class Analysis:
    def __init__(self, name, data):
        self._name = name
        self._data = data
    
    def corr_heatmap(self):
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/heatmap'):
            os.mkdir(path+'/heatmap')

        corr = self._data.corr()

        plt.figure(figsize = (12,10))
        sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':10}, annot = True, fmt = '.2f')
        plt.savefig(path+'/heatmap/heatmap.png')

        print('Progress: saving heatmap plot files in directory {}'.format(path+'/heatmap'))

        plt.close()

    def corr_scatter(self):
        path = './output/'+str(self._name)

        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path+'/scatter'):
            os.mkdir(path+'/scatter')
        
        count = 1
        for col in self._data.columns:
            plt.figure(figsize =(6,5))
            plt.scatter(self._data[col],self._data['(혈청지오티)ALT'], s = 6)
            plt.title('상관계수: '+str(self._data.corr()['(혈청지오티)ALT'][col]))
            plt.xlabel(col)
            plt.ylabel('(혈청지오티)ALT')
            plt.savefig(path+'/scatter/'+col+'.png')

            print('Progress: saving scatter plot files({}{}{}) in directory {}'.format(count,'/',len(self._data.columns),path+'/scatter'))
            
            count += 1
        plt.close()
