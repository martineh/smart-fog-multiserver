#import bodyDetector.detector   as bd
import weaponDetector.detector as wd
import faceDetector.detector   as fd
import faceIdentify.detector   as fi
import cv2
import sys

from PIL import Image, ImageDraw

def weapon_detector(img):
    #img = cv2.resize(img, (640, 480))    
    #Fase-2: Weapon detection (yolo)
    weaponsDetected = wd.detect_weapons(img, debug=True)
    
    return weaponsDetected

def face_detector_and_identify(img):
    #Fase-3: If a weapon detected, face detection (face_recognition)
    locations = fd.detect_faces(img, debug=True)
    if len(locations) >= 1:
        for face_location in locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            face_image = img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.show()

    #Fase-4: Identify the faces
    if len(locations) >= 1:
        names = fi.identify_faces(img, locations)
        print("People Identify: {}".format(names)) 
        
    return names
