#written by Zilin Shen

from packet_capturer_shen import PacketCapturer


# get the dataset from
test_dataset_name = "../set_1/test_new.pcap"
training_dataset_name = "../set_1/training_new.pcap"

training_packet_capturer = PacketCapturer(None,training_dataset_name)
train_packets = training_packet_capturer.pcap2packets()
test_packet_capturer = PacketCapturer(None,test_dataset_name)
test_packets = test_packet_capturer.pcap2packets()

