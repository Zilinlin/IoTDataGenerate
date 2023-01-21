from packet_capturer_shen import PacketCapturer

file_name = "../set_1/test.pcap"


packet_capturer = PacketCapturer(None,file_name)
packet_capturer.pcap2packets()
