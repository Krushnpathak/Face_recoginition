import numpy as np
import cv2
import pickle


trained_face_data = cv2.CascadeClassifier('Frontal_Face_Data_Pretrained.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
og_labels = {}
font = cv2.FONT_HERSHEY_SIMPLEX
with open("labels.pkl", 'rb') as f:#Bringing in the pickle file from other file
	og_labels =  pickle.load(f)#Gives name:id
	labels = {v:k for k,v in og_labels.items()}#Invert the dictionary

cap = cv2.VideoCapture(0)

while True:#checking for every frame
	#Capture frames
	ret , frame  = cap.read()
	frame  = cv2.flip(frame,1)
	#Grayscaled image and flipped
	gs_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#Detects Faces
	face_xy = trained_face_data.detectMultiScale(gs_img)#detect the faces

	#drawing the recatangle
	for (x,y,w,h) in face_xy:#checking for every face in the picture
		cv2.rectangle(frame , (x,y), (x+w,y+h),(0,255,0),2)#draw rectangle around the face
		roi_gray = gs_img[y:y+h , x:x+w]
		id_,conf = recognizer.predict(roi_gray)
		if conf>= 50 :
			print(id_)
			print(labels[id_])
			name = labels[id_]
			cv2.putText(frame,name,(x,y),font,1,(255,255,255),1,cv2.LINE_AA)

			#with open('/hello.csv', 'w') as f:
			#	writer = csv.writer(f)
			#	writer.writerrow(['',name])


	#Show the image
	cv2.imshow('Face Recognition',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print("Code Completed")
cap.release()
cv2.destroyAllWindows()
