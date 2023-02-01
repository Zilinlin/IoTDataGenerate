# wrritten by Hyunwoo Lee, rewritten by Zilin Shen
from flow import Flow
import logging
import numpy as np

# Window is the basic part in IoTEDEF
class Window:
    def __init__(self,protocol,saddr,sport,daddr,dport):
        self.flow = {}
        self.flow["forward"] = Flow(protocol,saddr,sport,daddr,dport)
        self.flow["backward"] = Flow(protocol,daddr,dport,saddr,sport)
        self.packets = {}
        self.packets["forward"] = []
        self.packets["backward"] = []

        # the start time and end time of window
        self.window_start_time = None
        self.window_end_time = None

        self.label = {}
        self.label["attack"] = 0
        self.label["infection"] = 0
        self.label["reconnaissance"] = 0

        self.stat = {} # the statistics from feature to value

    # to justice wether a pkt belongs to a flow
    def pkt_ifin_flow(self,flow,pkt):
        protocol, saddr, sport, daddr, dport = pkt.get_each_flow_info()

        if flow.get_protocol() == protocol and flow.get_saddr() == saddr and flow.get_sport() == sport and flow.get_daddr() == daddr and flow.get_dport() == dport:
            return True
        else:
            return False

    def add_packet(self, pkt):
        protocol, saddr, sport, daddr, dport = pkt.get_each_flow_info()

        if self.pkt_ifin_flow(self.flow["forward"],pkt):
            self.packets["forward"].append(pkt)
        elif self.pkt_ifin_flow(self.flow["backward"], pkt):
            self.packets["backward"].append(pkt)

        # this part is about the relationship between packet label and window label.
        # if we only get two kinds of datasets
        if pkt.get_label() ==1:
            self.label["attack"] = 1
            logging.debug("Window is set to {} (attack)".format(self.label["attack"]))
        elif pkt.get_label() == 2:
            self.label["infection"] =1
            logging.debug("Window is set to {} (infection)".format(self.label["infection"]))
        elif pkt.get_label() == 3:
            self.label["reconnaissance"] =1
            logging.debug("Window is set to {} (reconnaissance)".format(self.label["reconnaissance"]))

    def get_packets(self,direction):
        return self.packets[direction]

    def get_feature_value(self, feature):
        return self.stat[feature]

    #add the current value to currently existing number
    def add_feature_value(self,feature,value):
        if feature not in self.stat:
            self.stat[feature]=0
        self.stat[feature] = self.stat[feature] + value

    def set_times(self,start_time, end_time):
        self.window_start_time = start_time
        self.widow_end_time = end_time

    def get_flow(self,direction):
        return self.flow[direction]

    def is_corresponding_flow(self,window):
        b1 = self.flow["backward"]
        f1 = self.flow["forward"]
        b2 = window.get_flow("backward")
        f2 = window.get_flow("forward")

        ret1 = b1.is_corresponding_flow(b2)
        ret2 = b1.is_corresponding_flow(f2)
        ret3 = f1.is_corresponding_flow(b2)
        ret4 = f1.is_corresponding_flow(f2)

        return ret1 or ret2 or ret3 or ret4

    def get_label(self,kind=None):
        if kind:
            return self.label[kind]
        else:
            return self.label


