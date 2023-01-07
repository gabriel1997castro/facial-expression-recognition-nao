from inspect import CORO_CREATED
import cv2, socket
from cv2 import repeat
import numpy as np
import time
import mediapipe.python.solutions.face_detection_test
import pickle
from pointsFace import lipsOutter
import mediapipe as mp

from keras.models import model_from_json
from tensorflow.keras.models import Sequential

BUFF_SIZE = 65000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
HOST = '127.0.0.1'
portHost = 8081
message = 'AZUL'

client_socket.sendto(message.encode('utf-8'), (HOST, portHost))

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.results = None
        self.imgRGB = None
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mediapipe.python.solutions.drawing_utils
        self.mpFaceMash = mediapipe.python.solutions.face_mesh
        self.faceMash = self.mpFaceMash.FaceMesh(max_num_faces=10)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # self.face = mediapipe.python.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face = mp.solutions.mediapipe.python.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMash.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_FACE_OVAL,
                                           self.drawSpec, self.drawSpec)
                #print(faceLms)
        return img

    def findOnlyFaceMesh(self, img, draw=False):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMash.process(self.imgRGB)
        #print(self.results.multi_face_landmarks)

        contorno = img
        contorno.flags.writeable = False
        contorno = cv2.cvtColor(contorno, cv2.COLOR_BGR2RGB)

        results = self.face.process(contorno)

        if results.detections:
            for detection in results.detections:
                # print("detection: ", detection.location_data.relative_bounding_box)

                data = detection.location_data.relative_bounding_box

                h, w, c = contorno.shape
                xleft = data.xmin*w
                xleft = int(xleft)
                xtop = data.ymin*h
                xtop = int(xtop)
                xright = data.width*w + xleft
                xright = int(xright)
                xbottom = data.height*h + xtop
                xbottom = int(xbottom)

                print("left: ", xleft)
                print("top: ", xtop)
                print("right: ", xright)
                print("bottom: ", xbottom)

                contorno = contorno[xtop:xleft, xbottom:xright]
                # self.mpDraw.draw_detection(contorno, detection)
        
        empty = np.zeros(img.shape, dtype='uint8')
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_FACE_OVAL,
                                            self.drawSpec, self.drawSpec)
                    self.mpDraw.draw_landmarks(empty, faceLms, self.mpFaceMash.FACEMESH_FACE_OVAL,
                                            self.drawSpec, self.drawSpec)
            face = []
            for id, lm in enumerate(faceLms.landmark):
                # print('LM = ', lm)
                # print('AZUL', id)
                ih, iw = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                # if id % 2 == 0:
                # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.putText(empty, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.rectangle(contorno, )
                if lipsOutter.get(id):
                    img = cv2.circle(img, (x,y), 1, (255, 255, 255), -1)
                    empty = cv2.circle(empty, (x,y), 1, (255, 255, 255), -1)
                face.append([x,y])
        return empty, img, contorno

def main():
    pTime = 0
    detector = FaceMeshDetector()
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                empty, img, contorno = detector.findOnlyFaceMesh(frame)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

                cv2.imshow('Key Points', img)
                cv2.imshow('Fundo escuro', empty)


                ### Aplicando o treinamento

                model = Sequential()
                model = model_from_json(open("model.json", "r").read())
                model.load_weights('model.h5')



                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    client_socket.close()
                    break

if __name__ == '__main__':
    main()