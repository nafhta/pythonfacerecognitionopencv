import numpy as np
import cv2

#solo para probar la camara y la detección de rostros.

cap = cv2.VideoCapture(1)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Resolution : " + str(w) + "x" + str(h), 0, 0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #haar trained faces
    face_cascade_profile = cv2.CascadeClassifier('trainer-haar-files/haarcascade_profileface.xml')
    face_cascade_frontal = cv2.CascadeClassifier('trainer-haar-files/haarcascade_frontalface_alt.xml')

    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    detected = False;

    if (len(faces_frontal) > 0):
        for face in faces_frontal:
            (x, y, w, h) = face
            gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected = True

    if(detected is not True):
        faces_profile_right = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        if (len(faces_profile_right) > 0):
            for face in faces_profile_right:
                (x, y, w, h) = face
                gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
                detected = True

    if(detected is not True):
        imgflipped = cv2.flip(gray,1)
        faces_profile_left = face_cascade_profile.detectMultiScale(imgflipped, scaleFactor=1.2, minNeighbors=5);
        if (len(faces_profile_left) > 0):
            for face in faces_profile_left:
                (x, y, w, h) = face
                gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()