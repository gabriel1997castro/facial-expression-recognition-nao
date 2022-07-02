import socket, cv2, base64
import numpy as np
from PIL import Image
from naoqi import ALProxy
import qi
import vision_definitions

host = '127.0.0.1'
portHost = 8081
NAOIP = "169.254.187.174"
NAOPORT = 9559

vid = cv2.VideoCapture('video1.mkv')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)

BUFF_SIZE = 65536

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

server_socket.bind((host, portHost))

alVideo = ALProxy("ALVideoDevice", NAOIP, NAOPORT)
# alVideo = 
# alvalue = ALProxy("ALValue", NAOIP, NAOPORT)

session = qi.Session()
session.connect("tcp://" + NAOIP + ":" + str(NAOPORT))

video_service = session.service("ALVideoDevice")
resolution = vision_definitions.kVGA
colorSpace = vision_definitions.kRGBColorSpace
fps = 30
SUBSCRIBE_NAME = "NAO_CAM"

nameId = video_service.subscribe(SUBSCRIBE_NAME, resolution, colorSpace, fps)

while True:
    print
    msg, addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ', addr)
    teste = 400
    while(vid.isOpened()):

        camNAO = video_service.getImageRemote(nameId)
        if camNAO is None:
            print('Erro: NAO is none!')
            break

        WIDTH_INDEX = 0
        HEIGTH_INDEX = 1
        IMAGE_ARRAY_INDEX = 6

        imageWidth = camNAO[WIDTH_INDEX]
        imageHeight = camNAO[HEIGTH_INDEX]
        array = camNAO[IMAGE_ARRAY_INDEX]

        frame = str(bytearray(array))

        frame = Image.frombytes("RGB", (imageWidth, imageHeight), frame, 'raw', 'BGR', 0, -1)
        frame = np.asarray(frame)

        VERTICAL_FLIP_INDEX = 0
        HORIZONTAL_FLIP_INDEX = 1

        frame = cv2.flip(frame, VERTICAL_FLIP_INDEX) # FLIPPER AS IMAGENS
        frame = cv2.flip(frame, HORIZONTAL_FLIP_INDEX)


        encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        message = base64.b64encode(buffer)
        server_socket.sendto(message, addr)

        # cv2.imshow('Envia', frame) #Utilizar para debugger
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            video_service.unsubscribe(nameId)
            server_socket.close()
            break
