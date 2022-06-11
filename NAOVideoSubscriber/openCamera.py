import os
import sys
import time
from naoqi import ALProxy

IP = "169.254.53.248"
PORT = 9559
try:
  photoCaptureProxy = ALProxy("ALPhotoCapture", IP, PORT)
  # photoCaptureProxy.setResolution(2)
  # photoCaptureProxy.setPictureFormat("png")
  # photoCaptureProxy.takePicture("/home/nao/", "PhotoPycharme")
except e:
  print("Error when creating ALPhotoCapture proxy:")
  print(e)
  exit(1)