from objectDetection.detect import YoloV5OD
from objectDetection.detect import pairing_bodies_to_objects, save_pair_results, body_crop
import sys
import cv2
import time

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    t0 = time.time()
    bodyOD   = YoloV5OD("./objectDetection/yoloV5-weights/yolov5s.pt")
    weaponOD = YoloV5OD("./objectDetection/yoloV5-weights/granada-weapons.pt")
    t1 = time.time()
    print("[*] Models Loaded in %0.2f(s)" % (t1 - t0))
    print("[*] Running inference...")
    
    t0 = time.time()
    bodies  = bodyOD.do_inference(img, class_filter=['person'])
    weapons = weaponOD.do_inference(img)
    t1 = time.time()
    
    print("[*] Inference Done in %0.2f(s)" % (t1 - t0))
    print("[*] Pairing objects and weapons and cropping images...")
    t0 = time.time()
    pairs = pairing_bodies_to_objects(weapons, bodies) 
    bodies_crop = body_crop(pairs, weapons, bodies, img)
    t1 = time.time()
    print("[*] Pairing and cropping Done in %0.2f(s)" % (t1 - t0))    
    save_pair_results(pairs, weapons, bodies, img, "results.jpg")

    for i, b in enumerate(bodies_crop):
        cv2.imwrite("crops/crop-"+str(i)+".jpg", b[0])
