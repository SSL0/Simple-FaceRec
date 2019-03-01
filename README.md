# WebCam_FaceRecognition
Best face recognition on python use library dlib with web_cam

# !Attention! To use this programm you need Python version 3.6

# Installing the necessary components
Pre-trained model download links(Extract it in project folder):

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    
**Download modules:**

 pip3 install opencv-python 
 
 pip3 install scikit-image
 
 pip3 install scipy
        
**Now you need to install dlib library:**

`git clone https://github.com/davisking/dlib.git`

`cd dlib`

`python3 setup.py install`
    
    

# Add new person
In the project folder create a directory named "train": `mkdir train`. Next, run the file **new_person.py**, then put the person in front of the camera that you want to add for recognition, the program will ask for his name. 

# Recognition
Run the file "main.py", the console will display the names of people who are in front of the camera.

**P.S Sorry for english!) I used google translate.**
