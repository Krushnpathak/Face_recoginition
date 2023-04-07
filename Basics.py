import cv2 as cv
import numpy as np
import face_recognition


imgElon = face_recognition.load_image_file('images/Elon.webp')
imgElon = cv.cvtColor(imgElon, cv.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/Elon-test.jpg')
imgTest = cv.cvtColor(imgTest, cv.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv.rectangle(imgElon, (faceLoc[3], faceLoc[0]),
             (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
             (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
cv.putText(imgTest, f'{results} {round(faceDis[0],2)}',
           (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv.imshow('Elon Musk', imgElon)
cv.imshow('Elon Test', imgTest)
cv.waitKey(0)