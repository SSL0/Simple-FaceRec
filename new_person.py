import dlib, os
from skimage import io
from scipy.spatial import distance
import numpy as np
import cv2

def addNewPerson(name):
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    DIR = './train'
    nextid = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    #cv2.imshow('frame', frame)
    cv2.imwrite('./train/{}_{}.jpg'.format(nextid + 1, name), frame)
    print("ID: " + str(nextid + 1))
    cap.release()
    cv2.destroyAllWindows()


addNewPerson(input("Set name of new person: "))
