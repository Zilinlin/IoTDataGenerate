# written by Hyunwoo, rewritten by Zilin

from window import Window
from packet import Packet
import logging

import copy


class WindowManager:
    def __init__(self,packets_queue,period,swnd,move_size):
        logging.info("the length of packets_queue",len(packets_queue))

        # the queue is all packets
        self.queue = packets_queue #the packets to be precessed
        self.windows = [] # the output is several windows

        self.period = period # the period of window
        self.sliding_window = swnd # which is True/False, if use sliding window
        self.move_size = move_size
        #self.add_packets(packets_queue)
        #for pkt in packets_queue:
        #    self.queue.append(pkt)

    def add_packets(self,packets):
        for pkt in packets:
            self.queue.append(pkt)

    def add_packet(self, packet):
        logging.info("self.queue add packet")
        self.queue.append(packet)

    def process_packets(self):
        if self.sliding_window:
            packets = sorted(self.queue, key = lambda x: x.get_timestamp())

            # the start and end time of these packets
            first_time = packets[0].get_timestamp()
            last_time = packets[-1].get_timestamp()

            # start sliding window
            start_time = first_time
            end_time = first_time + self.period

            while end_time <= last_time:
                logging.info("the start time of sliding window,", start_time)
                logging.info("the end time of sliding window,", end_time)
                temp_packets = self.divide_packets(start_time, end_time)
                self.process_partial_packets(temp_packets)

                start_time = start_time + self.move_size
                end_time = start_time + self.period
                logging.info("current number of windows,",len(self.windows))

            #handle the last remaining part
            '''
            if start_time < last_time:
                end_time = last_time
                logging.info("the start time of sliding window:",start_time)
                logging.info("the end time of sliding window:",end_time)
                temp_packets = self.divide_packets(start_time, end_time)
                self.process_partial_packets(temp_packets)

            logging.info("first time:",first_time,"last time:",last_time)
            '''
        else:
            self.process_partial_packets(self.queue)


    # get the packets between the time interval start_time and end_time
    def divide_packets(self,start_time,end_time):
        packets = sorted(self.queue, key=lambda x: x.get_timestamp())

        # if there are packets during the time interval
        found = False
        start_index = -1
        for i in range(len(packets)):
            if packets[i].get_timestamp() > start_time:
                start_index = i
                break
        for j in range(len(packets)):
            if packets[j].get_timestamp() > end_time:
                end_index = j
                break
        sliding_packets = packets[i:j]

        return sliding_packets



    # process packets in self.queue, and output windows
    # process the packets of some time interval
    def process_partial_packets(self,packets):

        if len(packets) == 0:
            return

        partial_windows = []

        for pkt in packets:

            #logging.info("iteration is running .........")

            proto,saddr,sport,daddr,dport = pkt.get_each_flow_info()
            #logging.info("pkt info: ",proto,saddr,sport,daddr,dport,pkt.get_serial_number())
            wnd = Window(proto,saddr,sport,daddr,dport,self.period)
            #logging.info("wnd successfully")
            found = False
            for window in partial_windows:
                #logging.info("find corresponding window")
                if window.is_corresponding_flow(wnd):
                    found = True
                    #logging.info("found is true")
                    #logging.info("wnd",wnd)
                    #logging.info("window:",window)
                    window.add_packet(pkt)
                    del wnd
                    break
            if not found:
                wnd.add_packet(pkt)
                #logging.info("the number of window ++")
                partial_windows.append(wnd)

        # then add the windows of this time interval to self.windows
        logging.info("the number of packets of each partial_windows",len(packets))
        logging.info("the number of windows of each partial_windows",len(partial_windows))
        for wnd in partial_windows:
            self.windows.append(wnd)

