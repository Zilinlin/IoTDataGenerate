# this file is for test the dataset process
# including dpkt scapy

import dpkt

def process_dpkt(file_name):
    f = open(file_name, 'rb')
    pcap = dpkt.pcap.Reader(f)

    packet_count = 0
    print("start processing file,",file_name)
    for ts, buf in pcap:
        packet_count = packet_count +1
        print("count of packet ++:",packet_count)

        ether = dpkt.ethernet.Ethernet(buf)
        length = len(buf)
        ip = ether.data

        protocol = ip.p
        trans = None


    print("the number of all packets:", packet_count)


process_dpkt("../set_1/training_new.pcap")
process_dpkt("../set_1/test_new.pcap")

