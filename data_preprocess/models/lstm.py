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
from sklearn.utils import shuffle
import sklearn.metrics as metrics 
import tensorflow as tf
import sklearn as sk
from models.utils import recall_th_99, precision_th_99
from keras.callbacks import EarlyStopping
TIME_STEP = 2
THRESHOLD = 0.5

class Lstm(Algorithm):
    def __init__(self, name):
        super().__init__(name)
        #self.data = data
        #self.label = label

    def add_dataset(self, dataset):
        self.dataset = dataset
    # Please implement the following functions
    # Concerning dataset, refer to the class TrainingSet
    def learning(self,features,dataset,label,kind, epochs=50, patience=None):
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
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        self.classifier[kind].compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

        print("the info of dataset," ,dataset.shape)

        es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
        try:
            print("shuffle is enabled for training")
            dataset, labels = shuffle(dataset,labels)
            if patience is None:
                print("Early stopping: NO.")
                self.classifier[kind].fit(dataset, labels, epochs=epochs, validation_split=0.1, verbose=2)
            else:
                print("Early stopping: YES.")
                self.classifier[kind].fit(dataset, labels, epochs=epochs, validation_split=0.1, verbose=2, callbacks=[es])
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


    def predict(self, dataset, kind):
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
        fpr, tpr, thresholds_roc = metrics.roc_curve(label, np.array(pred).squeeze(), pos_label=1)
        precision, recall, thresholds_pr= metrics.precision_recall_curve(label, np.array(pred).squeeze(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auprc = metrics.auc(recall, precision)
        
        precision99 = precision_th_99(np.array(pred).squeeze(), label)
        recall99 = recall_th_99(np.array(pred).squeeze(), label)

        predicts = []
        for p in pred:
            ret = (p[0] > THRESHOLD).astype("int32")
            logging.info("direct calculation number:",p[0],"predict label:",ret)
            predicts.append(ret)
        pred = np.array(predicts)
        pred = pred.reshape((pred.shape[0]),)

        if fallback:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: 1".format(label, pred, ret))
        else:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: {}".format(label, pred, ret, TIME_STEP))

        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

        acc = (tn+tp)/len(label)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        #print("auc:",auc, "accuracy:", acc,"precision:", precision, "recall:", recall, "f1:",f1)
        #print("fp:",fp,",tp:",tp,",fn:",fn,",tn:",tn)

        metrics_dic = { 'auc': auc,
                        'accuracy': acc, 
                        'precision': precision, 
                        "recall": recall,
                        "f1": f1,
                        "fp": fp,
                        "tp": tp,
                        "fn": fn,
                        "tn": tn,
                        "auprc": auprc,
                        "precision99": precision99,
                        "recall99": recall99
                        }
        return pred, acc, metrics_dic
