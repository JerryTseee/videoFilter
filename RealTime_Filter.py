import cv2
import numpy as np
from utils import CFEVideoConf, image_resize

#load the related classifiers
face_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\third_party\\frontalEyes35x16.xml")
nose_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\third_party\\Nose18x15.xml")
glasses = cv2.imread("C:\\opencv\\build\\etc\\third_party\\glasses.png",-1)
mustache = cv2.imread("C:\\opencv\\build\\etc\\third_party\\mustache.png",-1)

#open the camera and capture the video
cap = cv2.VideoCapture(0)

#processing inside the while loop
while True:

    ret, original = cap.read()

    #break if no clip is retrieved
    if not ret:
        break

    #convert the color to gray
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #get the faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = original[y:y+h, x:x+h]
        cv2.rectangle(original, (x,y), (x+w, y+h), (255,255,255), 3)#draw the rectangle

        #for eye detection, and draw the rectangle
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
            roi_eyes = roi_gray[ey:ey+eh, ex:ex+ew]

            #put on the glasses
            glasses_resized = image_resize(glasses.copy(), width = ew)
            gw, gh, gc = glasses_resized.shape
            for i in range(gw):
                for j in range(gh):
                    if glasses_resized[i, j][3] != 0:
                        alpha = glasses_resized[i, j][3] / 255.0
                        roi_color[ey + i, ex + j] = (1 - alpha) * roi_color[ey + i, ex + j] + alpha * glasses_resized[i, j][:3]

        #for nose detection, and draw the rectangle
        noses = nose_detector.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 0), 3)
            roi_noses = roi_gray[ny:ny+nh, nx:nx+nw]

            #put on mustache
            mustache2 = image_resize(mustache.copy(), width=nw)
            mw, mh, mc = mustache2.shape
            for i in range(mw):
                for j in range(mh):
                    if mustache2[i, j][3] != 0:
                        alpha = mustache2[i, j][3] / 255.0
                        roi_color[ny + int(nh/2.0) + i, nx + j] = (1 - alpha) * roi_color[ny + int(nh/2.0) + i, nx + j] + alpha * mustache2[i, j][:3]

    #display
    original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
    cv2.imshow('output', original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()