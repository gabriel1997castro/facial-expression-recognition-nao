import socket, cv2, base64, imutils, math
import pickle

host = '127.0.0.1'
portHost = 8081

vid = cv2.VideoCapture(0) # Pela webcam
# vid = cv2.VideoCapture('video1.mkv') # Testes para transferencia de video
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

BUFF_SIZE = 65000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

server_socket.bind((host, portHost))

while True:
    print
    msg, addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ', addr)
    while(vid.isOpened()):
        ret, frame = vid.read()

        HORIZONTAL_FLIP_INDEX = 1

        frame = cv2.flip(frame, HORIZONTAL_FLIP_INDEX) # FLIPPER AS IMAGENS
        encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if encode:
            buffer = buffer.tobytes()
            buffer_size = len(buffer)
        
        num_packs = 1
        if buffer_size > BUFF_SIZE:
            num_packs = math.ceil(float(buffer_size)/BUFF_SIZE)
        
        frame_info = {"packs":int(num_packs)}

        server_socket.sendto(pickle.dumps(frame_info), addr)

        left = 0
        right = BUFF_SIZE

        for index in range(int(num_packs)):
            data = buffer[left:right]
            left = right
            right += BUFF_SIZE

            server_socket.sendto(data, addr)

        # message = base64.b64encode(buffer)
        # server_socket.sendto(message, addr)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            server_socket.close()
            break
