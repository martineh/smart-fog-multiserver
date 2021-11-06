import weapon_detection as wD
#import face_detection  as fD

import sys
import cv2
import time
#import face_recognition
#image = face_recognition.load_image_file(sys.argv[1])
#face_locations = face_recognition.face_locations(image)

#print (face_locations)
#sys.exit(-1)

print ("************ START DETECTION ******************")
t0 = time.time()
image = cv2.imread(sys.argv[1])
info, bodies = wD.detect_weapon_to_body(image)
t1 = time.time()

if bodies == None:
    print("Nothing Detected")

print ("Timing %0.2f" %(t1 - t0))
print ("************ STOP DETECTION ******************")

#image = cv2.resize(image, (800, 600))
#face = dL.face_detector(image)
#names = dL.face_identify(face)
#print("People Identify: {}".format(names))
#sys.exit(-1)

#info, body_list = dL.body_weapon_detector(image)
'''
print(info)
for body in bodies:
    print("Body detected")
    face = fD.face_detector(body)
    if face is not None:
        cv2.imshow("img", face)
        cv2.waitKey(0)
        names = fD.face_identify(face)
        print("People Identify: {}".format(names))
    else:
        print("Face Not Detected")

'''

#


'''
def pixel_centered(box):
    x, y, w, h = box

    return (x + int(w/2), y + int(h/2))

def body_weapon_detector(image):
    bodies_with_weapons = []
    bodies_ids_with_weapons = []
    #Detect weapons and calculate the center pixel from each box
    weapon_center_pixels = []
    weapon_idxs, weapon_boxes = wd.detect_weapons(image, debug=True)
    for i in range(len(weapon_boxes)):
        if i in weapon_idxs:
            c_pixel = pixel_centered(weapon_boxes[i])
            weapon_center_pixels.append(c_pixel)

    if len(weapon_center_pixels) <= 0:
        return bodies_with_weapons

    #If weapons detected.
    #   Detect bodies and calculate the center pixel from each box
    body_center_pixels   = []
    body_idxs, body_boxes = bd.detect_body(image)
    if len(body_idxs) > 0:
        for i in body_idxs.flatten():
            c_pixel = pixel_centered(body_boxes[i])
            body_center_pixels.append(c_pixel)

    if len(body_center_pixels) <= 0:
        return bodies_with_weapons

    #If weapon and body detected.
    #   Calculate the euclidian distance form each weapon to each body
    #   and select the minimum distance
    weapon_id = 0
    for w_pixel in weapon_center_pixels:
        min_distance = sys.maxsize
        body_id = 0
        body_selected = 0
        for b_pixel in body_center_pixels:
            diff = (w_pixel[0] - b_pixel[0], w_pixel[1] - b_pixel[1])
            distance = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])            
            if distance < min_distance:
                min_distance = distance
                body_selected = body_id
            body_id += 1
        #x, y, w, h = weapon_boxes[weapon_id]
        #cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)        
        #cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)
        weapon_id += 1
        if body_selected not in bodies_ids_with_weapons:
            x, y, w, h = body_boxes[body_selected]
            crop_image = image[y:y+h, x:x+w]
            bodies_with_weapons.append(crop_image)
            bodies_ids_with_weapons.append(body_selected)
        
    return bodies_with_weapons

def body_detector(image):
    idxs, boxes = bd.detect_body(image)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)
    
    return image
    
def weapon_detector(image):
    idxs, boxes = wd.detect_weapons(image, debug=True)
    for i in range(len(boxes)):
        if i in idxs:
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=1)

    return image

def face_detector(img):
    #cv2.imshow("show", img)
    #cv2.waitKey(0)
    locations = fd.detect_faces(img, debug=True)
    print(locations)
    top, right, bottom, left = locations[0]

    return img[top:bottom, left:right]

def face_identify(face_image):
    top, right, bottom, left = (0, face_image.shape[1], face_image.shape[0], 0)
    face_image = face_image[top:bottom - 1, left:right - 1]        
    names = fi.identify_faces(face_image, [(0, face_image.shape[1], face_image.shape[0], 0)] )

    return names



#START TEST
image = cv2.imread(sys.argv[1])
#face = face_detector(image)
#names = face_identify(face)
#print("People Identify: {}".format(names))
#sys.exit(-1)

body_images = body_weapon_detector(image)
for body in body_images:
    body = cv2.resize(body, (640, 480))
    face = face_detector(body)
    names = face_identify(face)
    print("People Identify: {}".format(names))
'''





'''
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
'''
