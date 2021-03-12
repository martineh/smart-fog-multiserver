import cv2
import os
import numpy as np
from PIL import Image
import face_recognition
from PIL import Image, ImageDraw

def detect_faces_other(img, debug):
    # Load the cascade
    config_path = os.path.dirname(__file__)
    face_cascade = cv2.CascadeClassifier(config_path + '/haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

    return len(faces) >= 1

def convert_img_from_opencv_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # For reversing the operation:
    im_np = np.asarray(pil_img)
    return im_np

def detect_faces(img, debug):
    image = convert_img_from_opencv_to_PIL(img)
    face_locations = face_recognition.face_locations(image)

    return face_locations
    #print(face_locations)

    #print("I found {} face(s) in this photograph.".format(len(face_locations)))
    #crops = []
    #for face_location in face_locations:
        # Print the location of each face in this image
        #top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # You can access the actual face itself     
        #face_image = image[top:bottom, left:right]
        #pil_image = Image.fromarray(face_image)
        #crops.append(pil_image)
        
    #return crops
