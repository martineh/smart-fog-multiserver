import numpy as np
import cv2
import os
import imutils

def detect_bodies(img):
    config_path = os.path.dirname(__file__)
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # using a greyscale picture, also for faster detection
    
    #low_cascade = cv2.CascadeClassifier(config_path + '/haarcascade_lowerbody.xml')
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #low = low_cascade.detectMultiScale(gray, 1.1 , 3)
    #for (x,y,w,h) in low:
    #    cv2.rectangle(img, (x,y), (x+w, y+h), (12,150,100),2)
    img = imutils.resize(img, 
                       width=min(400, img.shape[1]))     
    # detect people in the image
    # returns the bounding boxes for the detected objects
    (regions, _) = hog.detectMultiScale(img, winStride=(8,8), padding=(4,4), scale=1.05 )
    #boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #print(boxes)
    for (x, y, w, h) in regions:
        # display the detected boxes in the colour picture
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', img)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

