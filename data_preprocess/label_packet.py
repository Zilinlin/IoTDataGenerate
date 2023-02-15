# written by Zilin
# this part is for labeling the packets

from packet import Packet
import logging
import numpy as np

# change the label file to label.npy
def generate_label_data(file_name):
    f = open(file_name,'r')
    data = f.readlines()
    label_data = np.empty((0,))
    print("the count of label lines",len(data))
    for line in data:
        odom = line.split(',')
        la = int(odom[-1])
        label_data = np.concatenate((label_data,[la]),axis=0)
    print(label_data.shape)
    return label_data

# label the packets with label_data
def label_packets(pkts,label_data):
    new_packets = []
    length_label = label_data.shape[0]
    for pkt in pkts:
        serial_number = pkt.get_serial_number()
        if serial_number < length_label:
            label = label_data[serial_number]
            pkt.set_label(label)
            new_packets.append(pkt)
        else:
            break
    return new_packets


# these lines is for testing
generate_label_data('../set_1/training.label')

