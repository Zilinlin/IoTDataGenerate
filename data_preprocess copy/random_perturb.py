# written by Zilin
# this file is to randomly perturb the time of packets

import random

# the perturbation will delay the packets from attacker, and also add the packet length
# only delay the reconnaissance packet
def random_perturb(packets):
    for pkt in packets:
      label = pkt.get_label()
      if label == 3:
          ts = pkt.get_timestamp()
          print("old timestamp:",ts)
          ts = ts * 1.25
          print("new timestamp:",ts)
          pkt.set_timestamp(ts)

          length = pkt.get_packet_length()
          print("old length:",length)
          length = length + 10
          print("new length:",length)
          pkt.set_packet_length(length)




