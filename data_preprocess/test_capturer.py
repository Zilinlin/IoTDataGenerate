#written by Zilin Shen
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from packet_capturer_shen import PacketCapturer
from window_manager import WindowManager
from feature_extractor import FeatureExtractor
from numpy_generator import NumpyGenerator
from label_packet import generate_label_data, label_packets
from lstm import LSTM


# the hyperparameters of window manager
window_size = 1
swnd = True
# the class of detector
# attack, infection,
kind = "reconnaissance"

# get the dataset from
# because of the wrong generate_dataset, training is test,test is training
test_dataset_name = "../set_1/test_new.pcap"
training_dataset_name = "../set_1/training_new.pcap"
test_label_name = "../set_2/test.label"
training_label_name = "../set_2/training.label"

training_packet_capturer = PacketCapturer(None,training_dataset_name)
training_packet_capturer.pcap2packets()
train_packets = training_packet_capturer.packets

test_packet_capturer = PacketCapturer(None,test_dataset_name)
test_packet_capturer.pcap2packets()
test_packets = test_packet_capturer.packets

print("length of train packets,",len(train_packets))
print("length of test packets,", len(test_packets))

# ----------------start labeling the train_packets and test_packets-------------------
train_label = generate_label_data(training_label_name)
test_label = generate_label_data(test_label_name)
print("the shape of train_label,",train_label.shape)
print("the shape of test_label,",test_label.shape)

# filter the packets with label, drop the packet without label
train_packets = label_packets(train_packets, train_label)
test_packets = label_packets(test_packets, test_label)



# ----------------start generating the windows with packets--------------
train_window_manager = WindowManager(train_packets,window_size,swnd)
#train_window_manager.add_packets(train_packets)
train_window_manager.process_packets()
train_windows = train_window_manager.windows
print("successfully get training windows, the number of windows is ", len(train_window_manager.windows))

test_window_manager = WindowManager(test_packets,window_size,swnd)
#test_window_manager.add_packets(test_packets)
test_window_manager.process_packets()
test_windows = test_window_manager.windows
print("successfully get testing window, the number of windows is ",len(test_window_manager.windows))


# ---------------start extracting features------------------- #
test_fe = FeatureExtractor(test_windows)
test_fe.add_features()
test_fe.process_windows()
print(test_windows[0].stat,type(test_windows[0].stat))

train_fe = FeatureExtractor(train_windows)
train_fe.add_features()
train_fe.process_windows()

#------------------ start processing the windows to numpy data---------------#
test_data_generator = NumpyGenerator(test_windows,kind)
test_data_generator.process_windows()
print(test_data_generator.df,test_data_generator.dataset,test_data_generator.label)

# the second parameter is the kind of detector
train_data_generator = NumpyGenerator(train_windows,kind)
train_data_generator.process_windows()
print(train_data_generator.df,train_data_generator.dataset,train_data_generator.label)

# test each label of each packet
#for p in train_packets:
#    print(p.get_label())




# ----------------start learning with FeedForward--------------
data = train_data_generator.dataset
label = train_data_generator.label

from feedforward import Feedforward
print(data,label)
fforward = Feedforward("Feedforward")

features_len = train_fe.features_len()
fforward.learning(features_len,data,label,kind)


print("----------------start testing-------------")
test_data = test_data_generator.dataset
test_label = test_data_generator.label
print("the test data label",test_label)
fforward.detection(test_data,test_label,kind)
#print("return,",ret)
#print("prediction:",pred)

