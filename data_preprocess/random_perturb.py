# written by Zilin
# this file is to randomly perturb the time of packets

import random

# add a random time delay of [0,0.4) for each packet
def random_perturb_time(packets):
    for pkt in packets:
      ts = pkt.get_timestamp()
      ts += 0.4*random.random()
      pkt.set_timestamp(ts)




