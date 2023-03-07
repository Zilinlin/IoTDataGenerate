# written by Zilin
# this file is to randomly perturb the time of packets

import random

# the perturbation will only delay the packets from attacker
# only delay the reconnaissance packet
def random_perturb_time(packets):
    for pkt in packets:
      label = pkt.get_label()
      if label == 3:
          ts = pkt.get_timestamp()
          print("old timestamp:",ts)
          ts = ts * 10
          print("new timestamp:",ts)
          pkt.set_timestamp(ts)




