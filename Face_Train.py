import cv2
import os
from PIL import Image
import numpy as np
import pickle

trained_face_data = cv2.CascadeClassifier('frontal_face_extended.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"Photos")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		 if file.endswith("jpeg"):#finding the images
				path = os.path.join(root,file)#finding the path
				label = os.path.basename(os.path.dirname(path)).replace(" ","_").lower() #making the labels of the images clear
				pil_image = Image.open(path).convert("L") #L converts the image into grayscale
				#size = (800,800)
				#final_image = pil_image.resize(size,Image.ANTIALIAS)
				image_array = np.array(pil_image,"uint8")
				print(image_array)
				print(label,path)
				if not label in label_ids:#Creating the label ids
					label_ids[label] = current_id
					current_id +=1
				id_ = label_ids[label]
				face_xy = trained_face_data.detectMultiScale(image_array)
				for (x,y,w,h) in face_xy:
					roi = image_array[y:y+h,x:x+w]#Grab the region of interest
					x_train.append(roi)#add that roi to the training
					y_labels.append(id_)#giving the label ids


with open("labels.pkl", 'wb') as f:#Making a pickle file to store the label ids
	pickle.dump(label_ids,f)

recognizer.train(x_train , np.array(y_labels))#trains the data and labels
recognizer.save("trainner.yml")
