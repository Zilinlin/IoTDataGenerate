import os

def getfiles():
    pcap_file_names = os.listdir(r'../iot_intrusion_dataset')
    print(pcap_file_names)
    return pcap_file_names


def get_csv_dataset(pcap_file_name):
    input_file_path = "../iot_intrusion_dataset/" + pcap_file_name
    output_file_path = "../csv_iot_intrusion_dataset/" + pcap_file_name + '_.csv'


    frame_Features = "-e frame.time_delta -e frame.time_relative -e frame.len "
    flow_Features = "-e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e ip.proto -e ip.ttl "
    tcp_Features = "-e tcp.flags -e tcp.time_delta -e tcp.len -e tcp.ack -e tcp.connection.fin -e tcp.connection.rst -e tcp.connection.syn -e tcp.flags.ack -e tcp.flags.fin -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.urg -e tcp.hdr_len -e tcp.payload -e tcp.window_size_value -e tcp.checksum "

    #mqtt_Features = "-e  mqtt.clientid -e mqtt.clientid_len -e mqtt.conack.flags -e mqtt.conack.val -e mqtt.conflag.passwd -e mqtt.conflag.qos -e mqtt.conflag.reserved -e mqtt.conflag.retain -e  mqtt.conflag.willflag -e mqtt.conflags -e mqtt.dupflag -e mqtt.hdrflags -e mqtt.kalive -e mqtt.len -e mqtt.msg -e mqtt.msgtype -e mqtt.qos -e mqtt.retain -e mqtt.topic -e mqtt.topic_len -e mqtt.ver -e mqtt.willmsg_len "

    others = "-E header=y -E separator=, -E quote=d -E occurrence=f "



    allFeatures = frame_Features + flow_Features + tcp_Features + others

    command = 'tshark -r '+ input_file_path + ' -T fields ' + allFeatures + '> '+ output_file_path

    print(f"--- Input File: {input_file_path} ---")

    print('--Processing File--')

    print("=== Extracting Features and Generating CSV===")

    os.system(command)

    print("--- Done ---")


pcap_file_names = getfiles()

for pcap_file_name in pcap_file_names:
    get_csv_dataset(pcap_file_name)
