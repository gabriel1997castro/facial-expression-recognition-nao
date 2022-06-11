import cv2, socket
import numpy as np
import base64

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
HOST = '127.0.0.1'
portHost = 8081
message = 'AZUL'

client_socket.sendto(message.encode('utf-8'), (HOST, portHost))

while True:
    packet,_ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet, ' /')
    npdata = np.fromstring(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, 1)
    cv2.imshow('Recebendo a imagem', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        client_socket.close()
        break