import dlib, os
from skimage import io
from scipy.spatial import distance
import numpy as np
import cv2

path = './train'

def mainFunc():
    print("Start Capture")
    id = [] #Массив с именами людей внесенных в базу

    for f in os.listdir(path):
        name = str(f).replace(".jpg", "")
        id.append(name)
    id.sort()  

    print("All base: ", end='')
    print(id)
    

    sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat') 
    facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat') #Загружаем обученные модели

    #Часть для извлечения дескриптора(вещи которая помогает распознавать лица) из каждой фотографии 
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

    cap = cv2.VideoCapture(0) #Включаем вебку при помощи opencv
    
    while(True):
        ret, img = cap.read() #Берем изображение с вебки
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
