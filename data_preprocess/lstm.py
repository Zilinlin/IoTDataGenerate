# written by Hyunwoo, modified by Zilin
# this is for LSTM training

import sys
import copy
import logging
import numpy as np
from algorithm import Algorithm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

TIME_STEP = 2
THRESHOLD = 0.5

class Lstm(Algorithm):
    def __init__(self, name):
        super().__init__(name)
        #self.data = data
        #self.label = label

    # Please implement the following functions
    # Concerning dataset, refer to the class TrainingSet
    def learning(self,features,dataset,label,kind):
        #dataset = copy.deepcopy(windows.get_dataset(kind))
        #dataset = np.array(dataset)
        self.scale = StandardScaler().fit(dataset)
        #self.scale = StandardScaler().fit(dataset)
        dataset = self.scale.transform(dataset)
        fallback = False

        # just use the simpler method
        try:
            dataset = dataset.reshape((dataset.shape[0], TIME_STEP, int(dataset.shape[1] / TIME_STEP)))
        except:
            fallback = True
            dataset = dataset.reshape((dataset.shape[0], 1, dataset.shape[1]))

        #tmp = windows.get_labels(kind)
        #labels = []
        #for l in tmp:
        #    labels.append([l])
        labels = label

        # this is the number of features
        #features = 5
        #features = len(windows.get_feature_names())

        self.classifier[kind] = Sequential()
        if fallback:
            self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu', input_shape=(1, features)))
        else:
            self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu', input_shape=(TIME_STEP, int(features / TIME_STEP))))
        self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu'))
        self.classifier[kind].add(Dense(1, activation='sigmoid'))
        self.classifier[kind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("the info of dataset," ,dataset.shape)
        try:
            self.classifier[kind].fit(dataset, labels, epochs=50, validation_split=0.1, verbose=2)
            if fallback:
                logging.info("{} {} classifier is generated with the time step 1".format(self.get_name(), kind))
                print("classifier is generated with time step 1")
            else:
                logging.info("{} {} classifier is generated with the time step {}".format(self.get_name(), kind, TIME_STEP))
                print("classifire is generates with time step",TIME_STEP)
        except:
            self.classifier[kind] = None
            logging.info("{} {} classifier is not generated".format(self.get_name(), kind))


    def cal_fitness(self, dataset, label, kind):
        test = np.array(dataset)
        test = self.scale.transform(test)
        fallback = False
        try:
            test = test.reshape((test.shape[0], TIME_STEP, int(test.shape[1] / TIME_STEP)))
        except:
            fallback = True
            test = test.reshape((test.shape[0],1,test.shape[1]))

        pred = list(self.classifier[kind].predict(test))
        return pred


    def detection(self, dataset, label, kind):
        #logging.debug("window.get_code(): {}".format(window.get_code()))
        #label = window.get_label(kind)
        #test = window.get_code().copy()
        test = np.array(dataset)
        test = self.scale.transform(test)
        fallback = False
        try:
            test = test.reshape((test.shape[0], TIME_STEP, int(test.shape[1] / TIME_STEP)))
        except:
            fallback = True
            test = test.reshape((test.shape[0], 1, test.shape[1]))

        pred = list(self.classifier[kind].predict(test))

        predicts = []
        for p in pred:
            ret = (p[0] > THRESHOLD).astype("int32")
            predicts.append(ret)
        pred = np.array(predicts)
        pred = pred.reshape((pred.shape[0]),)

        if fallback:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: 1".format(label, pred, ret))
        else:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: {}".format(label, pred, ret, TIME_STEP))

        '''
        #calculate the accuracy
        count_same = 0
        fp=0
        tp=0
        fn=0
        tn=0
        others=0
        for i in range(len(label)):
            o = label[i]
            d=pred[i]
            if label[i] == pred[i]:
                count_same += 1
            # calculate FP TP...
            if o==0 & d==1:
                fp+=1
            elif o==1 & d==1:
                tp += 1
            elif o==1 & d==0:
                fn += 1
            elif o==0 & d==0:
                tn += 1
            else:
                others +=1
                print("o:",o,",d:",d)
        '''
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

        acc = (tn+tp)/len(label)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        print("accuracy:", acc,"precision:", precision, "recall:", recall, "f1:",f1)
        print("fp:",fp,",tp:",tp,",fn:",fn,",tn:",tn)
        return pred, acc
