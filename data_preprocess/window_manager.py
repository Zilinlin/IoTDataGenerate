# written by Hyunwoo, rewritten by Zilin

from window import Window
from packet import Packet

class WindowManager:
    def __init__(self):
        self.queue = [] #the packets to be precessed
        self.windows = [] # the output is several windows

    def add_packet(self, packet):
        self.queue.append(packet)

    # process packets in self.queue, and output windows
    def process_packets(self):
        while True:
            if len(self.queue) ==0:
                break

            pkt = self.queue.pop(0)

            try:
                proto,saddr,sport,daddr,dport = pkt.get_each_flow_info()
                wnd = Window(proto,saddr,sport,daddr,dport)

                found = False
                for window in self.windows:
                    if window.is_corresponding_flow(wnd):
                        found = True
                        window.add_packet(pkt)
                        del wnd
                        break
                if not found:
                    wnd.add_packet(pkt)
                    self.windows.append(wnd)
            except:
                continue

