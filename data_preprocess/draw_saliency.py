import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor

def draw_saliency():
    data = np.load('r_data.npy',allow_pickle=True)
    label = np.load('r_label.npy',allow_pickle = True)
    label = label.reshape(data.shape[0],1)
    print("shape of data:",data.shape,"shape of label:", label.shape)

    # combine
    combination = np.append(data,label,axis=1)
    print("shape of combine:",combination.shape)

    fe = FeatureExtractor(None)
    #fe.add_features()
    features = fe.features

    pd_index = []
    for f in features:
        pd_index.append(f.name)
    pd_index.append("class")
    print("pd_index:",pd_index)

    df = pd.DataFrame(combination, columns=pd_index)
    print(df.corr(method='spearman'))

    df.to_csv("data_label.csv")


draw_saliency()
