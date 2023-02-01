import dpkt
from packet import Packet
import logging
import socket
import ctypes
import time
from scapy.all import Ether, rdpcap


class timeval(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long),
            ('tv_usec', ctypes.c_long)]


class pcap_pkthdr(ctypes.Structure):
    _fields_ = [("ts", timeval), 
            ("caplen", ctypes.c_uint), 
            ("len", ctypes.c_uint)]

def inet_to_str(addr):
    try:
        return socket.inet_ntop(socket.AF_INET, addr)
    except:
        return socket.inet_ntop(socket.AF_INET6, addr)

class PacketCapturer:
    def __init__(self,label,file_name):
        self.label = label
        self.packets=[]
        self.file_name = file_name
        self.pkt_count = 0

    # this function can transfer the pacp file to the list of packets

    def pcap2packets(self):

        packets_captured = rdpcap(self.file_name)
        hbuf = pcap_pkthdr()

        for packet in packets_captured:
            packet = bytes(packet)
            #pbuf = pcap_next(pcap.handle, ctypes.byref(hbuf))
            self.pkt_count += 1
            header = pcap_pkthdr()
            header.len = hbuf.len
            header.caplen = hbuf.caplen
            header.ts.tv_sec = hbuf.ts.tv_sec
            header.ts.tv_usec = hbuf.ts.tv_usec
            self.pp(header, packet)
    
        logging.info("PACKET:Quit Packet Capturer")


    def pp(self, header, packet):
        ts = time.time()
    #    num, header, packet = pcap.queue.pop(0)
    #    logging.info("PACKET:Packet Number: {}".format(num))

        pkt = self.parse_packet(ts, header, packet)
        if pkt:
            if pkt.get_label() != -1:
                self.packets.append(pkt)
                if self.pkt_count % 100 == 0:
                    print('reading pacp file finished, the packet count is:', self.pkt_count)
        else:
            self.pkt_count += 1


    def parse_packet(self, ts, header, packet):
        eth = dpkt.ethernet.Ethernet(packet)

        if not isinstance(eth.data, dpkt.ip.IP):
            logging.debug("Non IP Packet type not supported {}".format(eth.data.__class__.__name__))
            return None

        length = len(packet)
        ip = eth.data
        df = bool(ip.off & dpkt.ip.IP_DF)
        mf = bool(ip.off & dpkt.ip.IP_MF)
        offset = bool(ip.off & dpkt.ip.IP_OFFMASK)

        protocol = ip.p
        trans = None

        if protocol == 1:
            logging.debug("ICMP: {} -> {}".format(inet_to_str(ip.src), inet_to_str(ip.dst)))

        elif protocol == 6:
            if not isinstance(ip.data, dpkt.tcp.TCP):
                #logging.error("TCP Parsing Error")
                return None
            tcp = ip.data
            sport = tcp.sport
            dport = tcp.dport
            logging.debug("TCP/IP: {}:{} -> {}:{} (len={})".format(inet_to_str(ip.src), sport, inet_to_str(ip.dst), dport, ip.len))
            trans = tcp
            key = "{}:{}:{}:{}".format(inet_to_str(ip.src), sport, inet_to_str(ip.dst), dport)

        elif protocol == 17:
            if not isinstance(ip.data, dpkt.udp.UDP):
                #logging.error("UDP Parsing Error")
                return None
            udp = ip.data
            sport = udp.sport
            dport = udp.dport
            logging.debug("UDP/IP: {}:{} -> {}:{} (len={})".format(inet_to_str(ip.src), sport, inet_to_str(ip.dst), dport, ip.len))
            trans = udp
            key = "{}:{}:{}:{}".format(inet_to_str(ip.src), sport, inet_to_str(ip.dst), dport)

        else:
            logging.error("Not supported protocol")
            return None

        return Packet(ts, header, eth, ip, trans, length, self.pkt_count)