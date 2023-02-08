# written by zilin
from features.flow.backward_iat_max import BackwardIatMax
from features.flow.backward_iat_mean import BackwardIatMean
from features.flow.backward_iat_min import BackwardIatMin
from features.flow.backward_iat_std import BackwardIatStd
from features.flow.backward_iat_total import BackwardIatTotal
#from features.flow.all import *
import os
# import all the files in one folder
folder = "features/flow"
for entry in os.scandir(folder):
    if entry.is_file():
        name = entry.name[:-3]
        string = f'from features.flow.{name} import *'
        exec (string)


class FeatureExtractor:
    def __init__(self,windows):
        self.features = []
        self.windows = windows

    def add_windows(self,windows):
        for wnd in windows:
            self.windows.append(wnd)

    def features_len(self):
        return len(self.features)

    def add_features(self):
        b_iat_max = BackwardIatMax("b_iat_max")
        b_iat_mean = BackwardIatMean("b_iat_mean")
        b_iat_min = BackwardIatMin("b_iat_min")
        b_iat_std = BackwardIatStd("b_iat_std")
        b_iat_total = BackwardIatTotal("b_iat_total")

        self.features.append(b_iat_max)
        self.features.append(b_iat_mean)
        self.features.append(b_iat_min)
        self.features.append(b_iat_std)
        self.features.append(b_iat_total)

        b_pkt_len_max = BackwardPacketLengthMax("b_pkt_len_max")
        b_pkt_len_mean = BackwardPacketLengthMean("b_pkt_len_mean")
        b_pkt_len_min = BackwardPacketLengthMin("b_pkt_len_min")
        b_pkt_len_std = BackwardPacketLengthStd("b_pkt_len_std")

        self.features.append(b_pkt_len_max)
        self.features.append(b_pkt_len_mean)
        self.features.append(b_pkt_len_min)
        self.features.append(b_pkt_len_std)

        bpkt_per_second = BpktsPerSecond("bpkt_per_second")
        flow_ack = FlowAck("flow_ack")
        flow_cwr = FlowCwr("flow_cwr")
        flow_ece = FlowEce("flow_ece")
        flow_fin = FlowFin("flow_fin")
        self.features.extend([flow_ack,flow_cwr,flow_ece,flow_fin,bpkt_per_second])

        f_i_max = FlowIatMax("f_i_max")
        f_i_mean = FlowIatMean("f_i_mean")
        f_i_std = FlowIatStd("f_i_std")
        f_i_min = FlowIatMin("f_i_min")
        f_i_total = FlowIatTotal("f_i_total")
        self.features.extend([f_i_max,f_i_mean,f_i_std,f_i_min,f_i_total])
        fpps = FlowPacketsPerSecond('fpps')
        f_pro = FlowProtocol('f_pro')
        f_psh = FlowPsh('f_psh')
        f_rst = FlowRst('f_rst')
        f_syn = FlowSyn('f_syn')
        f_urg = FlowUrg('f_urg')
        self.features.extend([f_pro,f_psh,f_rst,f_syn,f_urg,fpps])
        fo_i_max = ForwardIatMax('fo_i_max')
        fo_i_min = ForwardIatMin('fo_i_min')
        fo_i_mean = ForwardIatMin('fo_i_mean')
        fo_i_std = ForwardIatStd('fo_i_std')
        fo_i_total = ForwardIatTotal('fo_i_total')
        self.features.extend([fo_i_max,fo_i_min,fo_i_mean,fo_i_std,fo_i_total])
        fo_pl_max = ForwardPacketLengthMax('fo_pl_max')
        fo_pl_min = ForwardPacketLengthMin('fo_pl_min')
        fo_pl_std = ForwardPacketLengthStd('fo_pl_std')
        fo_pl_mean = ForwardPacketLengthMean('fo_pl_mean')
        fp_per_second = FpktsPerSecond('fp_per_second')
        self.features.extend([fo_pl_max,fo_pl_min,fo_pl_std,fo_pl_mean,fp_per_second])
        tb_p = TotalBackwardPackets('tb_p')
        t_bhlen = TotalBhlen('t_bhlen')
        t_fhlen = TotalFhlen('t_fhlen')
        tf_p = TotalForwardPackets('tf_p')
        tl_bp = TotalLengthOfBackwardPackets('tl_bp')
        tl_fp = TotalLengthOfForwardPackets('tl_fp')
        self.features.extend([tb_p,t_bhlen, t_fhlen,tf_p,tl_bp, tl_fp])



    def process_windows(self):
        for wnd in self.windows:
            self.extract_feature(wnd)

    # extract feature of one window
    def extract_feature(self,window):
        for f in self.features:
            #print('the name of this feature',f.get_name)
            f.extract_feature(window)

