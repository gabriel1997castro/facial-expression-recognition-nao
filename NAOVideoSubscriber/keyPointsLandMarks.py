import cv2, socket
import numpy as np
import base64
import time
import mediapipe.python.solutions.face_detection_test

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
HOST = '127.0.0.1'
portHost = 8081
message = 'NAO'

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

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMash.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_FACE_OVAL,
                                           self.drawSpec, self.drawSpec)
                # print(faceLms)
        return img

    def findOnlyFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMash.process(self.imgRGB)
        # print(self.results.multi_face_landmarks)
        empty = np.zeros(img.shape, dtype='uint8')
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # print('LMS')
                print('LMS', faceLms)
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_TESSELATION,
                                           self.drawSpec, self.drawSpec)
                self.mpDraw.draw_landmarks(empty, faceLms, self.mpFaceMash.FACEMESH_TESSELATION,
                                           self.drawSpec, self.drawSpec)
        return empty, img


def main():
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        packet, _ = client_socket.recvfrom(BUFF_SIZE)
        print('eric')
        data = base64.b64decode(packet, ' /')
        npdata = np.fromstring(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, 1)
        empty, frame = detector.findOnlyFaceMesh(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow('Recebendo a imagem', frame)
        cv2.imshow('Recebendo a vazia', empty)
        key = cv2.waitKey(1) & 0xFFe
        if key == ord('q'):
            client_socket.close()
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
