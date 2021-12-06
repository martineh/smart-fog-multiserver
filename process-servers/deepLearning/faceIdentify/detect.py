import cv2
import numpy as np 
import time
import sys
import os
import face_recognition
from pathlib import Path
from PIL import Image, ImageDraw


FILE = Path(__file__).resolve()

#-------------------------------------------------------------------------
#                   F A C E    I D E N T I F I C A T I O N
#-------------------------------------------------------------------------
class faceIdentity:
    def __init__(self, database):
        self.known_face_encodings = []
        self.known_face_names     = []        

        ids_db = []
        fd_faces = open(database + "/faces.db", "r")
        for line in fd_faces:
            if line[0] != "#":
                sp = line.split(";")
                ids_db.append((sp[0], sp[1]))
        fd_faces.close()

        for id_ in ids_db:
            # Load a sample picture and learn how to recognize it.
            id_image = face_recognition.load_image_file(database + "/"+id_[0])
            id_face_encoding = face_recognition.face_encodings(id_image)[0]            
            self.known_face_encodings.append(id_face_encoding)
            self.known_face_names.append(id_[1][:-1])    

    def identify(self, unknown_image):
        top, right, bottom, left = (0, unknown_image.shape[1], unknown_image.shape[0], 0)
        unknown_image  = unknown_image[top:bottom - 1, left:right - 1]
        face_locations = [(0, unknown_image.shape[1], unknown_image.shape[0], 0)] 
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        pil_image = Image.fromarray(unknown_image)
        results = []
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                results.append(self.known_face_names[best_match_index])
        return results


#-------------------------------------------------------------------------
#                   F A C E    D E T E C T I O N
#-------------------------------------------------------------------------

def convert_img_from_opencv_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    im_np = np.asarray(pil_img)
    return im_np

def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

def face_detector(image):
    locations = detect_faces(image)
    if len(locations) <= 0:
        return None    
    top, right, bottom, left = locations[0]    
    return image[top:bottom, left:right]

