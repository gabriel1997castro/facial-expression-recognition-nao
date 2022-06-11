import socket, cv2, base64, imutils

host = '127.0.0.1'
portHost = 8081

vid = cv2.VideoCapture('video1.mkv')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)

BUFF_SIZE = 65536

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

server_socket.bind((host, portHost))

while True:
    print
    msg, addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ', addr)
    teste = 400
    while(vid.isOpened()):
        ret, frame = vid.read()
        frame = imutils.resize(frame, width=teste)
        encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message, addr)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            server_socket.close()
            break
