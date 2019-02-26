import dlib, os
from skimage import io
from scipy.spatial import distance
import numpy as np
import cv2

path = './train'

def mainFunc():
    print("Start Capture")
    id = [] #Array with the names of people entered in the database

    for f in os.listdir(path):
        name = str(f).replace(".jpg", "")
        id.append(name)
    id.sort()  

    print("All base: ", end='')
    print(id)
    

    sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') 
    facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat') #Load the trained models

    #Part for extracting the descriptor (the thing that helps recognize faces) from each photo
    detector = dlib.get_frontal_face_detector()
    face_descriptor = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]  
    imagePaths.sort()  
    for imagePath in imagePaths:
        img = io.imread(imagePath)

        dets = detector(img, 1)

        for k, d in enumerate(dets):
            shape = sp(img, d)

        face_descriptor.append(facerec.compute_face_descriptor(img, shape))
    #END 

    cap = cv2.VideoCapture(0) #Turn on webcam with opencv
    
    while(True):
        ret, img = cap.read() #We take the image from the webcam
        dets_webcam = detector(img, 1)
        for k, d in enumerate(dets_webcam):
            shape = sp(img, d)
        flag = True
        face_descriptor2 = facerec.compute_face_descriptor(img, shape)
        for i in range(0, len(face_descriptor)):
            a = distance.euclidean(face_descriptor[i], face_descriptor2)
            if a < 0.6:
                print(id[i])
                flag = False
        if flag == True:
            print('Unknown')    

    cap.release()
    cv2.destroyAllWindows()

mainFunc() 
