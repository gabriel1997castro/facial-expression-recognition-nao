from inspect import CORO_CREATED
import cv2, socket
from cv2 import repeat
import numpy as np
import time
import mediapipe.python.solutions.face_detection_test
import pickle

BUFF_SIZE = 65000
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
HOST = '127.0.0.1'
portHost = 8081
message = 'AZUL'

client_socket.sendto(message.encode('utf-8'), (HOST, portHost))

lipsOutter = {
61: True,
185: True,
40: True,
39: True,
37: True,
0: True,
267: True,
269: True,
270: True,
409: True,
291: True,
375: True,
321: True,
405: True,
314: True,
17: True,
84: True,
181: True,
91: True,
146: True,
61: True
}

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
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                if lipsOutter.get(id):
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    img = cv2.circle(img, (x,y), 1, (0, 255, 0), -1)
                    # cv2.putText(empty, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    empty = cv2.circle(empty, (x,y), 1, (0, 255, 0), -1)
                face.append([x,y])
                
            face.append(face)
        return empty, img

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

                empty, img = detector.findOnlyFaceMesh(frame)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

                cv2.imshow('Key Points', img)
                cv2.imshow('Fundo escuro', empty)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    client_socket.close()
                    break

if __name__ == '__main__':
    main()