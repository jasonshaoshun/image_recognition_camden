import time
import os
import numpy as np
import cv2
import pickle

# from object_detection_request import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "analysis")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# cap = cv2.VideoCapture('vtest.avi')

cap = cv2.VideoCapture(0)

image_id = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]
		
		# draw the rectangle around faces
    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # if people detected, draw rectangle around and process the item recognition
    	cv2.imwrite(os.path.join(image_dir, "%d.png" %image_id), frame)
        
        # if fund people similar to someone in the library, draw the name on the image
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=2:
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2

    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		cv2.imwrite(os.path.join(image_dir, "%d.png" %image_id), frame)

    	# detection = ItemRecognition("analysis/%d.png" %image_id)
    	# detection.localize_objects()

    	image_id += 1

        # detect smiles
    	# subitems = smile_cascade.detectMultiScale(roi_gray)
    	# for (ex,ey,ew,eh) in subitems:
    	# 	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
