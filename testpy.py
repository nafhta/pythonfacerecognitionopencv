import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

subjects = ["", "El gran santiago", ""]
labels_recorded = []
conn = sqlite3.connect("records.sqlite")
c = conn.cursor()

def recordDetection(label):
    if(label not in labels_recorded):
        labels_recorded.append(label)
        try:
            c.execute("INSERT INTO records (id,identifier, site, cam_code) VALUES (NULL, '" + label + "', 'Door office 1' , 'CodeCAM')")
            conn.commit()
            #conn.close()
        except sqlite3.IntegrityError:
            print('ERROR: ID already exists in PRIMARY KEY column {}'.format(id_column))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    return cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def detect_face(img):
    gray = img
    #cascada entrenada para detectar rostros frontales.
    face_cascade_profile = cv2.CascadeClassifier('trainer-haar-files/haarcascade_profileface.xml')
    face_cascade_frontal = cv2.CascadeClassifier('trainer-haar-files/haarcascade_frontalface_alt.xml')
    
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    detected = False;

    if (len(faces_frontal) > 0):
        return faces_frontal
        detected = True

    if (detected is not True):
        faces_profile_right = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        if (len(faces_profile_right) > 0):
            return faces_profile_right
            detected = True

    if (detected is False):
        return None

def predict(test_img):
    img = test_img.copy()
    faces = detect_face(img)
    
    if(faces is not None): 
        for face in faces:
            rect = (x, y, w, h) = face
            label = face_recognizer.predict(img[y:y+w, x:x+h])
            #label_text = subjects[label[0]] + " " + str(label[1])
            label_text = subjects[label[0]]
            draw_rectangle(img, rect)
            draw_text(img, label_text, rect[0], rect[1]-5)
            recordDetection(label_text)
    
    return img

#stest_img1 = cv2.imread("test-data/test1.jpg")
#predicted_img1 = predict(test_img1)
#create a figure of 2 plots (one for each test image)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.setThreshold(40)
face_recognizer.read("trainer.yml")

cap = cv2.VideoCapture(1)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Resolution : " + str(w) + "x" + str(h), 0, 0)

while(True):
    ret, frame = cap.read()
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predicted_img1 = predict(test_img1)
   
    cv2.imshow("frame", predicted_img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()