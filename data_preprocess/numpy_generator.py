# written by Zilin
# this file is transfer the windows with statistics to numpy dataset and the label?
# the dataset and label is fpr LSTM algorithm

import pandas as pd
import numpy as np

class NumpyGenerator:
    def __init__(self,windows,kind):
        self.windows = windows
        self.df = pd.DataFrame()
        self.dataset = np.array(self.df)
        self.label = np.empty((0,))

        self.kind = "attack"

    def process_windows(self):
        for wnd in self.windows:
            stat = wnd.stat
            #self.df = self.df.append(stat,ignore_index=True)
            self.df = pd.concat([self.df, pd.DataFrame(stat, index=[0])])
            self.dataset = np.array(self.df)
            label = wnd.get_label("attack")
            self.label = np.concatenate((self.label,[label]),axis=0)





