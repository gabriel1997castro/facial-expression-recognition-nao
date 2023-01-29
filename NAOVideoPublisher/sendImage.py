import socket, cv2, base64, math
import pickle
import time
from datetime import timedelta

host = '127.0.0.1'
portHost = 8081

fileName = "angry0123.txt"

vid = cv2.VideoCapture("./AngryFER/Angry.avi") # Por video
# vid = cv2.VideoCapture(0) # Pela webcam
# vid = cv2.VideoCapture('video1.mkv') # Testes para transferencia de video
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

BUFF_SIZE = 65000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

server_socket.bind((host, portHost))

timer = []
timerSave = 0

fps = vid.get(cv2.CAP_PROP_FPS)
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
duration = round(frame_count/fps)

while True:
    msg, addr = server_socket.recvfrom(BUFF_SIZE)

    arrayEmotions = []
    arrayDataEmotion = []

    print('GOT connection from ', addr)
    pTime = time.time()
    while(vid.isOpened()):
        ret, frame = vid.read()
        # frame = cv2.imread("./affect_neutral_origin.png")
        # frame = cv2.resize(frame, (640,580))

        if True:

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
            cv2.imshow('Envia', frame)

            pred = server_socket.recvfrom(BUFF_SIZE)

            pred = pred[0]

            print("emotions: ", pred)
            emotion = pred.split(":")[0]
            dataEmotions = pred.split(":")[1]
            print(emotion)

            if emotion == 'Angry':
                emotion = 'Raiva'

            if emotion == 'Disgust':
                emotion = 'Desgosto'

            if emotion == 'Fear':
                emotion = 'Medo'

            if emotion == 'Happy':
                emotion = 'Feliz'

            if emotion == 'Neutral':
                emotion = 'Neutro'

            if emotion == 'Surprise':
                emotion = 'Surpresa'

            arrayEmotions.append(emotion)
            arrayDataEmotion.append(dataEmotions)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)

            timerSave = timerSave + 333 
            # timer.append(timerSave)
            pTime = cTime
            print("emotion: ", emotion)
            print("fps: ", fps)
            print("timer: ", timerSave)

            timer.append(timerSave)





            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):

                text_file = open(fileName, "wt")
                print("len: ", len(arrayEmotions))

                for index in range(len(arrayEmotions)):
                    n = text_file.write("time: ")
                    timeVideo = timedelta(milliseconds=timer[index])

                    n = text_file.write(str(timeVideo).split(":")[2])
                    n = text_file.write("   emotion: ")
                    n = text_file.write(str(arrayEmotions[index]))
                    n = text_file.write("   Array Predict: ")
                    n = text_file.write(str(arrayDataEmotion[index]))
                    n = text_file.write("\n")
                text_file.close()

                server_socket.sendto("shutdown", addr)

                server_socket.shutdown(socket.SHUT_RDWR)
                server_socket.close()
                break

        else:
            text_file = open(fileName, "wt")
            print("len: ", len(arrayEmotions))

            for index in range(len(arrayEmotions)):
                n = text_file.write("time: ")

                timeVideo = timedelta(milliseconds=timer[index])

                n = text_file.write(str(timeVideo).split(":")[2])
                n = text_file.write("   emotion: ")
                n = text_file.write(str(arrayEmotions[index]))
                n = text_file.write("   Array Predict: ")
                n = text_file.write(str(arrayDataEmotion[index]))
                n = text_file.write("\n")
            text_file.close()

            server_socket.sendto("shutdown", addr)

            server_socket.shutdown(socket.SHUT_RDWR)
            server_socket.close()
