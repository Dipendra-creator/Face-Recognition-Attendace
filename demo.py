from random import random
import cv2
import numpy as np
import face_recognition

imgJohnCena = face_recognition.load_image_file('Images/JohnCena.jpg')
imgJohnCena = cv2.cvtColor(imgJohnCena, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Images/profile.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgTest2 = face_recognition.load_image_file('Images/JCT3.jpg')
imgTest2 = cv2.cvtColor(imgTest2, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgJohnCena)[0]
encodeImg = face_recognition.face_encodings(imgJohnCena)[0]

faceLoc1 = face_recognition.face_locations(imgTest)[0]
encodeImg1 = face_recognition.face_encodings(imgTest)[0]

faceLoc2 = face_recognition.face_locations(imgTest2)[0]
encodeImg2 = face_recognition.face_encodings(imgTest2)[0]

cv2.rectangle(imgJohnCena, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (round(random()*255), round(random()*255), round(random()*255)), 2)
cv2.rectangle(imgTest, (faceLoc1[3], faceLoc1[0]), (faceLoc1[1], faceLoc1[2]), (round(random()*255), round(random()*255), round(random()*255)), 2)
cv2.rectangle(imgTest2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (round(random()*255), round(random()*255), round(random()*255)), 2)

result = face_recognition.compare_faces([encodeImg], encodeImg1)
faceDis = face_recognition.face_distance([encodeImg], encodeImg1)
result2 = face_recognition.compare_faces([encodeImg], encodeImg2)
faceDis2 = face_recognition.face_distance([encodeImg], encodeImg2)

cv2.putText(imgTest, f'{result[0] } {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgTest2, f'{result2[0]} {round(faceDis2[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# print(result[0], faceDis[0], result2[0], faceDis1[0])
cv2.imshow("John Cena", imgJohnCena)
cv2.imshow("John Cena Test", imgTest)
cv2.imshow("John Cena Test2", imgTest2)

cv2.waitKey(0)