# written by Zilin
# this file is transfer the windows with statistics to numpy dataset and the label?
# the dataset and label is fpr LSTM algorithm

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
#import matplotlib.pyplot as plt
#import seaborn as sns

class NumpyGenerator:
    def __init__(self,windows,kind,imb=False):
        self.windows = windows
        self.df = pd.DataFrame()
        self.dataset = np.array(self.df)
        self.label = np.empty((0,))
        self.dataset_smo = None
        self.label_smo = None

        self.kind = kind
        self.imb = imb

    def process_windows(self):
        for wnd in self.windows:
            stat = wnd.stat
            #self.df = self.df.append(stat,ignore_index=True)
            self.df = pd.concat([self.df, pd.DataFrame(stat, index=[0])])
            self.dataset = np.array(self.df)
            label = wnd.get_label(self.kind)
            self.label = np.concatenate((self.label,[label]),axis=0)

        #add the balancing part directly
        smo = SMOTE()

        if len(self.dataset) > 1 and self.imb==True :
            self.dataset_smo, self.label_smo = smo.fit_resample(self.dataset,self.label)

    '''
    def draw_corr(self):
        # add the label index to dataframe and calculate the correlation
        df = self.df
        df['class'] = self.label
        corr = df.corr()
        ax = plt.subplots(figsize= (40,40))
        ax = sns.heatmap(corr, vmax=.8, square=True, annot=True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.savefig('picture.jpg')
    '''












