import socket, cv2, math, base64
import numpy as np
from PIL import Image
from naoqi import ALProxy
import qi
import vision_definitions
import pickle


host = '127.0.0.1'
portHost = 8081
NAOIP = '192.168.1.62'
NAOPORT = 9559

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

tts = session.service("ALTextToSpeech")
tts.setLanguage("Brazilian")

nameId = video_service.subscribe(SUBSCRIBE_NAME, resolution, colorSpace, fps)

while True:
    msg, addr = server_socket.recvfrom(BUFF_SIZE)
    print('GOT connection from ', addr)
    teste = 400
    while True:

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
        cv2.imshow('Envia', frame) #Utilizar para debugger
        # server_socket.sendto(message, addr)
        pred = server_socket.recvfrom(BUFF_SIZE)
        
        emotion = pred[0]
        print(emotion)

        if emotion == 'Sad':
            emotion = 'Triste'

        if emotion == 'Happy':
            emotion = 'Feliz'

        if emotion == 'Neutral':
            emotion = 'Neutro'

        if emotion == 'Angry':
            emotion = 'Raiva'

        tts.say(emotion)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            video_service.unsubscribe(nameId)
            server_socket.close()
            break
