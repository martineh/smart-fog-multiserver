import detection as dL
import sys
import cv2

#import face_recognition
#image = face_recognition.load_image_file(sys.argv[1])
#face_locations = face_recognition.face_locations(image)

#print (face_locations)
#sys.exit(-1)

image = cv2.imread(sys.argv[1])
info, bodies = dL.detect_weapon_to_body(image)

if bodies == None:
    print("Nothing Detected")
    
#image = cv2.resize(image, (800, 600))
#face = dL.face_detector(image)
#names = dL.face_identify(face)
#print("People Identify: {}".format(names))
#sys.exit(-1)

#info, body_list = dL.body_weapon_detector(image)
print(info)
for body in bodies:
    print("Body detected")
    face = dL.face_detector(body)
    if face is not None:
        cv2.imshow("img", face)
        cv2.waitKey(0)
        names = dL.face_identify(face)
        print("People Identify: {}".format(names))
    else:
        print("Face Not Detected")


