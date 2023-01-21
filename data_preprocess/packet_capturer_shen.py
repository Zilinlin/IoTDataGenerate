# written by zilin shen

import dpkt
from packet import Packet

class PacketCapturer:
    def __init__(self,label,file_name):
        self.label = label
        self.packets=[]
        self.file_name = file_name

    # this function can transfer the pacp file to the list of packets
    def pcap2packets(self):
        f = open(self.file_name, 'rb')
        pcap = dpkt.pcap.Reader(f)

        pkt_count = 0

        for ts, buf in pcap:
            pkt_count += 1

            ether = dpkt.ethernet.Ethernet(buf)

            if not isinstance(ether.data, dpkt.ip.IP):
                logging.debug("Non IP Packet type not supported {} ".format(ether.data.__class__.__name__))
            length = len(buf)
            ip = ether.data
            df = bool(ip.off & dpkt.ip.IP_DF)
            mf = bool(ip.off & dpkt.ip.IP_MF)
            offset = bool(ip.off & dpkt.ip.IP_OFFMASK)

            protocol = ip.p
            trans = None

            if protocol == 1:
                logging.debug("ICMP: {} -> {}".format(inet_to_str(ip.src), inet_to_str(ip.dst)))

            elif protocol == 6:
                if not isinstance(ip.data,dpkt.tcp.TCP):
                    return None
                tcp = ip.data
                sport = tcp.sport
                dport = tcp.dport
                logging.debug("TCP/IP: {}:{} -> {}:{} (len={})".format(inet_to_str(ip.src),sport,inet_to_str(ip.dst),dport, ip.len))
                trans = tcp
            elif protocol == 17:
                if not isinstance(ip.data, dpkt.udp.UDP):
                    return None
                udp = ip.data
                sport = udp.sport
                dport = udp.dport
                logging.debug("UD/IP: {}:{} -> {}:{} (len = {})".format(inet_to_str(ip.src),sport, inet_to_str(ip.dst),dport,ip.len))
                trans = udp
            else:
                logging.error("Not supported protocol")
                return None
            # TODO: header is not finished
            # the original code use pcap_pkthdr in libpcap/winpcap
            # just use empty thing as header
            header = ''
            packet = Packet(ts,header,ether,ip,trans, length)

            self.packets.append(packet)
            print('reading pacp file finished, the packet count is:', pkt_count)
