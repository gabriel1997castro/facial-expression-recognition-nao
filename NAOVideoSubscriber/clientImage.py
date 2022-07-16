from email.headerregistry import Address
import cv2, socket
import numpy as np
import base64
import pickle

BUFF_SIZE = 65000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
HOST = '127.0.0.1'
portHost = 8081
message = 'AZUL'

client_socket.sendto(message.encode('utf-8'), (HOST, portHost))

while True:
    packet,_ = client_socket.recvfrom(BUFF_SIZE)

    if len(packet) < 100:
        frame_info = pickle.loads(packet)

        if frame_info:
            nums_of_packs =  frame_info["packs"]

            for index in range(nums_of_packs):
                data, address = client_socket.recvfrom(BUFF_SIZE)

                if index == 0:
                    buffer = data
                else:
                    buffer += data
            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape(frame.shape[0], 1)
            # data = base64.b64decode(packet, ' /')
            # npdata = np.fromstring(data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            cv2.imshow('Recebendo a imagem', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                client_socket.close()
                break