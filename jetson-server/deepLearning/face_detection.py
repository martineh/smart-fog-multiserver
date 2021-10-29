import cv2
import numpy as np 
import time
import sys
import os
import face_recognition
from PIL import Image, ImageDraw


config_path = os.path.dirname(__file__)


#-------------------------------------------------------------------------
#                   F A C E    D E T E C T I O N
#-------------------------------------------------------------------------

def convert_img_from_opencv_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    im_np = np.asarray(pil_img)
    return im_np

def detect_faces(image):
    #image = convert_img_from_opencv_to_PIL(img)
    #img_resize = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    #img_resize = img_resize[:, :, ::-1]
    #img_resize = image
    #cv2.imwrite("body.jpg", img_resize)
    face_locations = face_recognition.face_locations(image)
    return face_locations

#-------------------------------------------------------------------------
#                   F A C E    I D E N T I F I C A T I O N
#-------------------------------------------------------------------------
known_face_encodings = []
known_face_names     = []

def load_know_people():
    config_path = os.path.dirname(__file__)

    ids_db = []
    fd_faces = open(config_path+"/faces_database/faces.db", "r")
    for line in fd_faces:
        if line[0] != "#":
            sp = line.split(";")
            ids_db.append((sp[0], sp[1]))
    fd_faces.close()

    for id_ in ids_db:
        # Load a sample picture and learn how to recognize it.
        id_image = face_recognition.load_image_file(config_path + "/faces_database/"+id_[0])
        id_face_encoding = face_recognition.face_encodings(id_image)[0]

        known_face_encodings.append(id_face_encoding)
        known_face_names.append(id_[1][:-1])    
    
load_know_people()

def identify_faces(unknown_image, face_locations):
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    names = []
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        names.append(name)
                
    return names


#-------------------------------------------------------------------------
#                           W R A P P E R S
#-------------------------------------------------------------------------

def face_detector(image):
    locations = detect_faces(image)
    if len(locations) <= 0:
        return None
    
    top, right, bottom, left = locations[0]
    
    return image[top:bottom, left:right]

def face_identify(image):
    top, right, bottom, left = (0, image.shape[1], image.shape[0], 0)
    image = image[top:bottom - 1, left:right - 1]        
    names = identify_faces(image, [(0, image.shape[1], image.shape[0], 0)] )

    return names
