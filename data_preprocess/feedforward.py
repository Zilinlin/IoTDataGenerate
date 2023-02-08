import sys
import copy
import logging
import numpy as np
from algorithm import Algorithm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.metrics import Recall, Precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler

THRESHOLD=0.5

class Feedforward(Algorithm):
    def __init__(self, name):
        super().__init__(name)

    # Please implement the following functions
    # Concerning dataset, refer to the class TrainingSet
    def learning(self, features, dataset, labels, kind):
        #dataset = copy.deepcopy(windows.get_dataset(kind))
        dataset = np.array(dataset)
        self.scale = StandardScaler().fit(dataset)
        dataset = self.scale.transform(dataset)
        dataset = dataset.reshape((dataset.shape[0], 1, dataset.shape[1]))

        # this is the number of features
        features = features
        #features = len(windows.get_feature_names())
        #tmp = windows.get_labels(kind)
        #labels = []
        #for l in tmp:
        #    labels.append([l])

        # this kind is modified by Zilin
        # kind is directly assigned as "attack"
        #kind = "attack"
        self.classifier[kind] = Sequential()
        self.classifier[kind].add(Dense(128, activation='relu', input_shape=(1,features), input_dim=features))
        self.classifier[kind].add(Dense(128, activation='relu'))
        self.classifier[kind].add(Dense(1, activation='sigmoid'))
        self.classifier[kind].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        try:
            self.classifier[kind].fit(dataset, labels, epochs=20, batch_size=1, verbose=1)
            logging.info("{} {} classifier is generated".format(self.get_name(), kind))
        except:
            self.classifier[kind] = None
            logging.info("{} {} classifier is not generated".format(self.get_name(), kind))




    # modified by zilin
    def detection(self, dataset, label,kind):
        #label = window.get_label(kind)
        #test = window.get_code().copy()
        #kind = "attack"
        test = np.array(dataset)
        test = self.scale.transform(test)
        test = test.reshape((test.shape[0], 1, test.shape[1]))
        #print("the test data:",test)
        #print("the shape of test data,",test.shape)

        # modified by Zilin
        #pred = list(self.classifier[kind].predict(test)[0][0])
        pred = list(self.classifier[kind].predict(test))

        #print("pred [0]", pred[0])
        predicts = []
        # just add the probability of class 0 (benign)
        for p in pred:
            ret = (p[0] > THRESHOLD).astype("int32")
            predicts.append(ret)
        pred = np.array(predicts)
        pred = pred.reshape((pred.shape[0]),)
        #print("the prediction result",pred)
        #print("the shape os prediction results",pred.shape)
        #val = pred[0]
        # pred.insert(0, 1-val)
        logging.debug("label: {}, pred: {}, ret: {}".format(label, pred, ret))

        # calculate the accuracy of detection
        count_same = 0
        for i in range(len(label)):
            if label[i] == pred[i]:
                count_same += 1
        acc = count_same/len(label)
        print("the accuracy of detection",acc)

        return pred,acc
