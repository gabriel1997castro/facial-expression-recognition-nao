import os
import sys
import time
from naoqi import ALProxy

IP = "169.254.96.12"
PORT = 9559
try:
  photoCaptureProxy = ALProxy("ALPhotoCapture", IP, PORT)
  photoCaptureProxy.setResolution(2)
  photoCaptureProxy.setPictureFormat("png")
  photoCaptureProxy.takePicture("/home/nao/", "testeFunfa")
except Exception, e:
  print "Error when creating ALPhotoCapture proxy:"
  print str(e)
  exit(1)