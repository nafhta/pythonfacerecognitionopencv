import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('trainer-haar-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);    
    if (len(faces) == 0):
        return None, None

    cv2.imshow("Training on image...", img)
    cv2.waitKey(100)
    
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data():
    
    data_folder_path = "training-data" 
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
           
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = image
            
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data()
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.save("trainer.yml");