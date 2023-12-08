import struct

# Class for creating pipeline connection
class PipeLineClientModel:
    def __init__(self, pipe_address: str):
        self.pipe_address = pipe_address

    def setUpPipeLineConnection(self):
        try:
            self.pipeLine = open(self.pipe_address, 'r+b', 0)
            return True
        except Exception as e:
            print("[PipeLine] Error while connecting :: %s" % e)
            return False

    def send_Packet_To_Unity(self, packet):
        try:
            encoded_packet = packet.encode('utf-8')
            self.pipeLine.write(struct.pack('I', len(encoded_packet)) + encoded_packet)
            self.pipeLine.seek(0)  
            return True
        except Exception as e:
            print("[PipeLine] Error while sending :: " + str(e))
            return False

