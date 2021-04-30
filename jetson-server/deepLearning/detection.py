import cv2
import numpy as np 
import time
import sys
import os
import face_recognition
from PIL import Image, ImageDraw

import torch

config_path = os.path.dirname(__file__)

def display_image(image, weapon_boxes, body_boxes, final_body_boxes):
    for box, conf, cls, center in body_boxes:
        c1, c2 = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(image, c1, c2, (255, 0, 0), 3)
        cv2.putText(image, "body", (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 2, lineType=cv2.LINE_AA)

    for box in final_body_boxes:
        c1, c2 = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(image, c1, c2, (0, 255, 0), 3)
        cv2.putText(image, "body-weapon", (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 2, lineType=cv2.LINE_AA)

    for box, conf, cls, center in weapon_boxes:
        label = f'Weapon {conf:.2f}'
        c1, c2 = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(image, c1, c2, (0, 0, 255), 3)
        cv2.putText(image, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 2, lineType=cv2.LINE_AA)

    scale_percent = 100 # percent of original size 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("result.png", resized)
    #cv2.waitKey(0)


def pixel_centered(box):
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    x0, y0 = c1
    x1, y1 = c2
    width  = x1 - x0
    height = y1 - y0
    return (x0 + int(width/2), y0 + int(height/2))

def get_distance(p0, p1):
    diff = (p0[0] - p1[0], p0[1] - p1[1])
    #return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
    return torch.sqrt(diff[0]*diff[0] + diff[1]*diff[1])

def crop_image(box, image):
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    x0, y0 = c1
    x1, y1 = c2
    width  = x1 - x0
    height = y1 - y0

    return image[y0:y0+height, x0:x0+width]
#-------------------------------------------------------------------------
#                   L O A D    M O D E L S    Y O L O    V 5
#-------------------------------------------------------------------------

weapon_model = torch.hub.load('ultralytics/yolov5', 'custom',
                              path_or_model=config_path+"/models/weapons.pt")

body_model = torch.hub.load('ultralytics/yolov5', 'custom',
                              path_or_model=config_path+"/models/yolov5s.pt")

#-------------------------------------------------------------------------
#          W E A P O N    A N D    B O D Y    D E T E C T I O N
#-------------------------------------------------------------------------

def detect_weapon_to_body(image):
    info = {'gun' : 0, 'knife' : 0}    
    crops = []
    #weapon detection
    weapon_boxes = []
    weapon_results = weapon_model(image[:, :, ::-1], size=640)    
    for (img, pred) in zip(weapon_results.imgs, weapon_results.pred):
        for *box, conf, cls in pred:
            if (int(cls) in [0, 2]) and (conf > 0.5):
                if int(cls) == 0:
                    info['gun'] += 1
                else:
                    info['knife'] += 1
                center = pixel_centered(box)
                weapon_boxes.append([(int(box[0]), int(box[1]), int(box[2]), int(box[3])), conf, cls, center])

    #body detection
    body_boxes = []
    body_results = body_model(image[:, :, ::-1], size=640)    
    for (img, pred) in zip(body_results.imgs, body_results.pred):
        for *box, conf, cls in pred:
            if (int(cls) in [0, 2]) and (conf > 0.5):
                center = pixel_centered(box)
                body_boxes.append([(int(box[0]), int(box[1]), int(box[2]), int(box[3])), conf, cls, center])                

    #Not weapons detected
    if len(weapon_boxes) == 0 or len(body_boxes) == 0:
        return None, None

    #body association 
    final_bodies = []
    final_body_ids = []
    for weapon in weapon_boxes:
        w_box, w_conf, w_cls, w_center = weapon
        min_distance = sys.maxsize
        actual_body = 0
        for body in body_boxes:
            b_box, b_conf, b_cls, b_center = body
            distance = get_distance(w_center, b_center)
            if distance < min_distance:
                body_id = actual_body
                min_distance = distance
            actual_body += 1

        if body_id not in final_body_ids:
            final_body_ids.append(body_id)
            crops.append(crop_image(body_boxes[body_id][0], image))
            final_bodies.append(body_boxes[body_id][0]) #Comment
            
    display_image(image, weapon_boxes, body_boxes, final_bodies)
    
    return info, crops    


        
#-------------------------------------------------------------------------
#                   B O D Y    D E T E C T I O N
#-------------------------------------------------------------------------

def detect_body(image):
    results = body_model(image[:, :, ::-1], size=640)
    
    for (img, pred) in zip(results.imgs, results.pred):
        for *box, conf, cls in pred:
            if (int(cls) == 0) and (conf > 0.5):
                label = f'{results.names[int(cls)]} {conf:.2f}'
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(image, c1, c2, (255, 0, 0), 3)
                cv2.putText(image, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 2, lineType=cv2.LINE_AA)
    
    scale_percent = 30 # percent of original size 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("img", resized)
    cv2.waitKey(0)

def detect_body_knife(image):
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)    
    # sets the blob as the input of the network
    body_net.setInput(blob)    
    # get all the layer names
    ln = body_net.getLayerNames()
    ln = [ln[i[0] - 1] for i in body_net.getUnconnectedOutLayers()]
    #start = time.perf_counter()
    layer_outputs = body_net.forward(ln)
    #time_took = time.perf_counter() - start
    #print(f"Time took: {time_took:.2f}s")

    body_boxes, body_confidences   = [], []
    knife_boxes, knife_confidences = [], []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                if class_id == 0:
                    body_boxes.append([x, y, int(width), int(height)])
                    body_confidences.append(float(confidence))
                elif class_id == 43:
                    knife_boxes.append([x, y, int(width), int(height)])
                    knife_confidences.append(float(confidence))
                class_ids.append(class_id)
    
    body_idxs  = cv2.dnn.NMSBoxes(body_boxes, body_confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    knife_idxs = cv2.dnn.NMSBoxes(knife_boxes, knife_confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    
    return body_idxs, body_boxes, knife_idxs, knife_boxes

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


def body_weapon_detector(image):
    info = {'gun' : 0, 'knife' : 0}
    bodies_with_weapons = []
    bodies_ids_with_weapons = []
    
    #Detect weapons and calculate the center pixel from each box
    weapon_center_pixels = []
    weapon_idxs, weapon_boxes = detect_weapons(image, debug=True)
    for i in range(len(weapon_boxes)):
        if i in weapon_idxs:
            c_pixel = pixel_centered(weapon_boxes[i])
            weapon_center_pixels.append(c_pixel)
            info['gun'] += 1
    
    body_center_pixels, body_boxes   = [], []
    body_idxs, body_boxes_tmp, knife_idxs, knife_boxes = detect_body_knife(image)
    if len(knife_idxs) > 0:
        for i in knife_idxs.flatten():
            c_pixel = pixel_centered(knife_boxes[i])
            weapon_center_pixels.append(c_pixel)
            weapon_boxes.append(knife_boxes[i])
            info['knife'] += 1

    if len(weapon_center_pixels) <= 0:    
        return info, bodies_with_weapons
    #else:
    #print("Weapons Detected:")
    #for box in weapon_boxes:
    #print(box)
    #x, y, w, h = box
    #cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
    
    if len(body_idxs) > 0:
        for i in body_idxs.flatten():
            c_pixel = pixel_centered(body_boxes_tmp[i])
            body_center_pixels.append(c_pixel)
            body_boxes.append(body_boxes_tmp[i])
                
    if len(body_center_pixels) <= 0:
        return info, bodies_with_weapons
    #else:
    #print("Bodies Detected:")
    #for box in body_boxes:
    #print(box)
    #x, y, w, h = box
    #cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    
    #cv2.imshow("image", image)
    #cv2.waitKey(0)

    #If weapon and body detected.
    #   Calculate the euclidian distance form each weapon 
    #   to each body and select the minimum distance
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
        weapon_id += 1
        if body_selected not in bodies_ids_with_weapons:
            x, y, w, h = body_boxes[body_selected]
            crop_image = image[y:y+h, x:x+w]
            bodies_with_weapons.append(crop_image)
            bodies_ids_with_weapons.append(body_selected)
        
    return info, bodies_with_weapons

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
