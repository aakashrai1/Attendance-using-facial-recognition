# -*- coding: utf-8 -*-
"""
@author: akash
"""

import cv2
#import logging as log
from time import sleep
from predict import PredictImage

cascPath = "./cvdata/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

img_width, img_height = 224, 224
classify = PredictImage()

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(30, 30)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
            
        padding = 80
        #print(frame.shape, x, x+w, y, y+h)
        img = frame[y-padding:y+h+padding, x-padding:x+w+padding]
        
        cv2.rectangle(frame, (x-padding, y-padding), (x+w+padding, y+h+padding), (0, 255, 0), 2)
        try:
            resized_img = cv2.resize(img, (img_width, img_height))
            name = classify.predict(resized_img)
            #print(name)
            if name is not None:
                draw_text(frame, name, x, y-5)
        except:
            #print('error')
            pass

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite('test.png',img)
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



