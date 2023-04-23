#written by Zilin Shen
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import numpy as np
from packet_capturer_daniel import PacketCapturer
from window_manager import WindowManager
from feature_extractor import FeatureExtractor
from numpy_generator import NumpyGenerator
from label_packet import generate_label_data, label_packets
from lstm import LSTM
from random_perturb import random_perturb
from pso import PSO

# the hyperparameters of window manager
window_size = 1
move_size = 0.1 #how long the movement of window
swnd = True
# the class of detector
# attack, infection,reconnaissance
kind = "reconnaissance"
# the chosen algorithm "feedforward" or "lstm"
algorithm = "lstm"

# get the dataset from
# because of the wrong generate_dataset, training is test,test is training
test_dataset_name = "../set_1/test.pcap"
training_dataset_name = "../set_1/training.pcap"
test_label_name = "../set_1/test.label"
training_label_name = "../set_1/training.label"

training_packet_capturer = PacketCapturer(None,training_dataset_name)
training_packet_capturer.pcap2packets()
train_packets = training_packet_capturer.packets

test_packet_capturer = PacketCapturer(None,test_dataset_name)
test_packet_capturer.pcap2packets()
test_packets = test_packet_capturer.packets

print("length of train packets,",len(train_packets))
print("length of test packets,", len(test_packets))

# ----------------start labeling the train_packets and test_packets-------------------
train_label, train_ts = generate_label_data(training_label_name)
test_label, test_ts = generate_label_data(test_label_name)
print("the shape of train_label,",train_label.shape)
print("the shape of test_label,",test_label.shape)

# filter the packets with label, drop the packet without label
train_packets = label_packets(train_packets, train_label,train_ts)
print('length of training packets after labeling',len(train_packets))
test_packets = label_packets(test_packets, test_label,test_ts)
print("length of testing packets after labeling",len(test_packets))

# -----------start randomly perturbing packets--
#random_perturb(test_packets)

# ----------------start generating the windows with packets--------------
train_window_manager = WindowManager(train_packets,window_size,swnd,move_size)
#train_window_manager.add_packets(train_packets)
train_window_manager.process_packets()
train_windows = train_window_manager.windows
print("successfully get training windows, the number of windows is ", len(train_window_manager.windows))

test_window_manager = WindowManager(test_packets,window_size,swnd,move_size)
#test_window_manager.add_packets(test_packets)
test_window_manager.process_packets()
test_windows = test_window_manager.windows
print("successfully get testing window, the number of windows is ",len(test_window_manager.windows))


# ---------------start extracting features------------------- #
test_fe = FeatureExtractor(test_windows)
#test_fe.add_features()
test_fe.process_windows()
print(test_windows[0].stat,type(test_windows[0].stat))

train_fe = FeatureExtractor(train_windows)
#train_fe.add_features()
train_fe.process_windows()

#------------------ start processing the windows to numpy data---------------#
test_data_generator = NumpyGenerator(test_windows,kind)
test_data_generator.process_windows()
print(test_data_generator.df,test_data_generator.dataset,test_data_generator.label)

# the second parameter is the kind of detector
train_data_generator = NumpyGenerator(train_windows,kind,True)
train_data_generator.process_windows()
print(train_data_generator.df,train_data_generator.dataset,train_data_generator.label)

# draw the correlation picture of training dataset
#train_data_generator.draw_corr()

# test each label of each packet
#for p in train_packets:
#    print(p.get_label())


# -----------get the training and testing data of numpy format
# the training data, use balancing dataset and label
data = train_data_generator.dataset
label = train_data_generator.label

np.save('r_data.npy',data)
np.save('r_label.npy',label)
# the testing data
test_data = test_data_generator.dataset
test_label = test_data_generator.label



# -----------------start learning with some algorithm----------------------

from lstm import Lstm
lstm = Lstm("lstm")

from feedforward import Feedforward
fforward = Feedforward("Feedforward")

features_len = train_fe.features_len()

if algorithm == "lstm":
    lstm.learning(features_len, data,label,kind)

    lstm.detection(test_data, test_label, kind)
    # ------- prepare for pso algorithm ------
    # first find a 100th window which label is reconnaissance
    # because the calculation result of
    window_count = 0
    range_speed = [(-0.02,0.02),(-2,2)]
    range_pop = [(0,0.4),(0,40)]
    learning_rate = [0.001, 0.1]

    mani_windows = []

    start_time = time.process_time()
    for i in range(len(test_windows)):
        if test_label[i] == 1:
            window_count += 1
            if window_count >= 300:
                break

            if window_count >=200:
                print("current manipulating window number:",i)
                pso_window = test_windows[i]
                # add this to limit current running time
                pso = PSO(learning_rate, 100, pso_window, range_speed,range_pop, lstm)
                mani_windows.append(pso.get_window())
        #else:
        #    mani_windows.append(test_windows[i])

    # process the mani windows
    end_time = time.process_time()

    mani_fe = FeatureExtractor(mani_windows)
    mani_fe.process_windows()
    mani_data_generator = NumpyGenerator(mani_windows,kind)
    mani_data_generator.process_windows()
    mani_data = mani_data_generator.dataset
    mani_label = mani_data_generator.label
    print("start testing the manipulated windows")
    lstm.detection(mani_data, mani_label, kind)

    print("500 windows manipulation time:", str(end_time - start_time))


    # start testing with LSTM
    #lstm.detection(test_data,test_label,kind)

elif algorithm == "feedforward":
    fforward.learning(features_len,data,label,kind)

    print("----------------start testing-------------")
    print("the test data label",test_label)
    fforward.detection(test_data,test_label,kind)

'''
# to calculate more details about test_data and test_label
print("the total number of training data:", len(label))
count_positive = 0
count_negative = 0
count_others = 0
for i in range(len(label)):
    if label[i] ==0:
        count_negative += 1
    elif label[i] == 1:
        count_positive += 1
    else:
        print(label[i])
        count_others += 1
print("count_positive:",count_positive)
print("count_negative:",count_negative)
print("count_others:",count_others)
'''

