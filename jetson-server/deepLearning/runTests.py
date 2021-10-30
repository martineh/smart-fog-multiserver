import sys
import cv2
import time
import argparse
import os
from os import listdir
from os.path import isfile, join

from objectDetection.detect import YoloV5OD
from objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop

ROOT_WEIGHTS = "./objectDetection/yoloV5-weights"
WEIGHTS = ["yolov5s.pt", "granada-weapons.pt"]

def get_files(opt):
    files = []
    if opt.img != "":
        files.append(opt.img)
    else:
        files = [join(opt.imgPath, f) for f in listdir(opt.imgPath) if isfile(join(opt.imgPath, f))]

    return files

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath', '-p',    type=str, default="./testImages/", help='images input path')
    parser.add_argument('--img',     '-i',    type=str, default="",              help='image file name')
    parser.add_argument('--outputPath', '-o', type=str, default="./testOutput/", help='image processing output path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_options()
    files = get_files(opt)
    
    load_t0  = time.time()
    bodyOD   = YoloV5OD(ROOT_WEIGHTS+"/" + WEIGHTS[0])
    load_t1  = time.time()
    weaponOD = YoloV5OD(ROOT_WEIGHTS+"/" + WEIGHTS[1])
    load_t2  = time.time()
    
    inf_time  = 0
    post_time = 0
    for f in files:
        img = cv2.imread(f)
        #Inferece. Body and Weapons detection
        t0 = time.time()
        bodies  = bodyOD.do_inference(img, class_filter=['person'])
        weapons = weaponOD.do_inference(img)
        t1 = time.time()
        inf_time += (t1 - t0)
        #Pairing bodies with weapons
        pairs = pairing_object_to_bodies(weapons, bodies) 
        bodies_crop = body_crop(pairs, weapons, bodies, img)
        t2 = time.time()
        post_time += (t2 - t1)
        #Save Results
        outFile = opt.outputPath + "/out-" + os.path.basename(f)
        save_pair_results(pairs, weapons, bodies, img, outFile)
        
        #for i, b in enumerate(bodies_crop):
        #cv2.imwrite("crops/crop-"+str(i)+".jpg", b[0])

        
    print("==========================================================")
    print("=              P R O C E S S     T I M I N G             =")
    print("==========================================================")
    print("Models Load Timing:")
    print("    [*] Weights '"+WEIGHTS[0]+"' Loaded in: %0.2f(s)" % (load_t1 - load_t0))
    print("    [*] Weights '"+WEIGHTS[1]+"' Loaded in: %0.2f(s)" % (load_t2 - load_t1))
    print("Process Timing for '%d' images:" % (len(files)))
    print("    [*] Inference Time in  :  %0.2f(s) " % (inf_time))
    print("    [*] Preprocess Time in :  %0.2f(s) " % (post_time))
    print("    [*] TOTAL Time in      :  %0.2f(s) " % (inf_time + post_time))
    if len(files) > 1:
        print("Process Timing for one image:")
        print("    [*] Inference Time in  :  %0.2f(s) " % (inf_time / len(files)))
        print("    [*] Preprocess Time in :  %0.2f(s) " % (post_time / len(files)))
        print("    [*] TOTAL Time in      :  %0.2f(s) " % ((inf_time + post_time) / len(files)))
    print("==========================================================")
