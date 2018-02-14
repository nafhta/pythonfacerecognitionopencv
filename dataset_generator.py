import numpy as np
import cv2

cap = cv2.VideoCapture(1)

Id=input('enter your id')
sampleNum=0

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('trainer-haar-files/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('trainer-haar-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) > 0):
        (x, y, w, h) = faces[0]
        gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("training-data/" + Id + "/" + Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        sampleNum = sampleNum + 1

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()