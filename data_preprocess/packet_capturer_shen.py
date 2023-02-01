# written by zilin shen

import dpkt
from packet import Packet
import logging
import socket
#from scapy.all import *

class PacketCapturer:
    def __init__(self,label,file_name):
        self.label = label
        self.packets=[]
        self.file_name = file_name

    def inet_to_str(addr):
        try:
            return socket.inet_ntop(socket.AF_INET, addr)
        except:
            return socket.inet_ntop(socket.AF_INET6, addr)

    # this function can transfer the pacp file to the list of packets
    def pcap2packets(self):
        # read file with dpkt library
        f = open(self.file_name, 'rb')
        pcap = dpkt.pcap.Reader(f)

        #packets = rdpcap(self.file_name)
        pkt_count = 0

        # for counting the number of error packets
        none_count = 0

        for ts, buf in pcap:
        #for buf in packets:

            pkt_count += 1

            ether = dpkt.ethernet.Ethernet(buf)

            if not isinstance(ether.data, dpkt.ip.IP):
                logging.debug("Non IP Packet type not supported {} ".format(ether.data.__class__.__name__))
                continue
            length = len(buf)
            ip = ether.data
            #print("type of ip", type(ip))
            #df = bool(ip.off & dpkt.ip.IP_DF)
            #mf = bool(ip.off & dpkt.ip.IP_MF)
            #offset = bool(ip.off & dpkt.ip.IP_OFFMASK)

            protocol = ip.p
            trans = None

            if protocol == 1:
                logging.debug("ICMP: {} -> {}".format(PacketCapturer.inet_to_str(ip.src), PacketCapturer.inet_to_str(ip.dst)))

            elif protocol == 6:
                if not isinstance(ip.data,dpkt.tcp.TCP):
                    print("None")
                    none_count += 1
                    continue
                    #return None
                tcp = ip.data
                sport = tcp.sport
                dport = tcp.dport
                logging.debug("TCP/IP: {}:{} -> {}:{} (len={})".format(PacketCapturer.inet_to_str(ip.src),sport,PacketCapturer.inet_to_str(ip.dst),dport, ip.len))
                trans = tcp
            elif protocol == 17:
                if not isinstance(ip.data, dpkt.udp.UDP):
                    #return None
                    print("None")
                    none_count += 1
                    continue
                udp = ip.data
                sport = udp.sport
                dport = udp.dport
                logging.debug("UDP/IP: {}:{} -> {}:{} (len = {})".format(PacketCapturer.inet_to_str(ip.src),sport, PacketCapturer.inet_to_str(ip.dst),dport,ip.len))
                trans = udp
            else:
                logging.error("Not supported protocol")
                #return None
                none_count += 1
                print("None")
                continue
            # TODO: header is not finished
            # the original code use pcap_pkthdr in libpcap/winpcap
            # just use empty thing as header
            header = ''
            packet = Packet(ts, header, ether, ip, trans, length, pkt_count)

            self.packets.append(packet)
            print('reading pacp file finished, the packet count is:', pkt_count)
            print('the error packet count is: ',none_count)
