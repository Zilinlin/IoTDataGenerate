# written by Hyunwoo, rewritten by Zilin

from window import Window
from packet import Packet
import copy


class WindowManager:
    def __init__(self,packets_queue):
        print("the length of packets_queue",len(packets_queue))
        self.queue = [] #the packets to be precessed
        self.windows = [] # the output is several windows

        #self.add_packets(packets_queue)
        #for pkt in packets_queue:
        #    self.queue.append(pkt)

    def add_packets(self,packets):
        for pkt in packets:
            self.queue.append(pkt)

    def add_packet(self, packet):
        print("self.queue add packet")
        self.queue.append(packet)

    # process packets in self.queue, and output windows
    def process_packets(self):
        while True:
            #print("iteration is running .........")
            if(len(self.queue) == 0):
                #print("the length of self.queue",len(self.queue))
                break

            pkt = self.queue.pop(0)

            proto,saddr,sport,daddr,dport = pkt.get_each_flow_info()
            #print("pkt info: ",proto,saddr,sport,daddr,dport)
            wnd = Window(proto,saddr,sport,daddr,dport)
            #print("wnd successfully")
            found = False
            for window in self.windows:
                #print("find corresponding window")
                if window.is_corresponding_flow(wnd):
                    found = True
                    #print("found is true")
                    #print("wnd",wnd)
                    #print("window:",window)
                    window.add_packet(pkt)
                    del wnd
                    break
            if not found:
                wnd.add_packet(pkt)
                #print("the number of window ++")
                self.windows.append(wnd)

