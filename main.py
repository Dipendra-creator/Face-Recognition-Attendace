import csv
import os
from datetime import date
from datetime import datetime

import cv2 as cv
import face_recognition
import numpy as np

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
today = date.today()


def create_dir(parentDirectory, childDirectory):
    try:
        Path = os.path.join(parentDirectory, childDirectory)
        os.mkdir(Path)

    except FileExistsError:
        pass


def create_csv_file(Name, filename):
    create_dir('Attendance', str(today))
    with open(f'Attendance/{str(today)}/{filename}', "r+", newline="") as csvfile:
        prev_data = csvfile.readlines()
        name_list = []

        for line in prev_data:
            entry = line.split(',')
            name_list.append(entry[0])
        if Name not in name_list:
            now = datetime.now()
            data = csv.writer(csvfile)
            x = [Name, now.strftime("%H:%M:%S")]
            data.writerow(x)


for cls in myList:
    currImg = cv.imread(f"{path}/{cls}")
    images.append(currImg)
    classNames.append(os.path.splitext(cls)[0])


def find_face_loc_encodings(image):
    face_loc_list = []
    encoding_list = []

    for photo in image:
        photo = cv.cvtColor(photo, cv.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(photo)[0]
        encode_image = face_recognition.face_encodings(photo)[0]
        face_loc_list.append(face_location)
        encoding_list.append(encode_image)

    return face_loc_list, encoding_list


face_loc_known_face, encoding_known_face = find_face_loc_encodings(images)

cam = cv.VideoCapture(0)

old_name = ''
while True:
    success, img = cam.read()
    img_small = cv.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv.cvtColor(img_small, cv.COLOR_BGR2RGB)

    face_loc_curr_frame = face_recognition.face_locations(img_small)
    encode_curr_frame = face_recognition.face_encodings(img_small, face_loc_curr_frame)

    for encode_face, face_loc in zip(encode_curr_frame, face_loc_curr_frame):
        matches = face_recognition.compare_faces(encoding_known_face, encode_face)
        face_distance = face_recognition.face_distance(encoding_known_face, encode_face)

        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = classNames[match_index].upper()
            if name != old_name:
                old_name = name
                create_csv_file(old_name, 'Attendance.csv')

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1+6, y2-6), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('WebCam', img)

    # Press ESC to Exit
    if cv.waitKey(1) & 0xFF == 27:
        cam.release()
        break

cam.release()
