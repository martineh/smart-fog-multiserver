import sys
import cv2
import time
import argparse
import os
from os import listdir
from os.path import isfile, join

from objectDetection.detect import YoloV5OD
from objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop, face_crop

ROOT_WEIGHTS = "./objectDetection/yoloV5-weights/"

WEIGHTS = [ROOT_WEIGHTS+"yolov5s.pt",
           ROOT_WEIGHTS+"weapons-YOLOv5s-300epc.pt",
           ROOT_WEIGHTS+"face_detection_yolov5s.pt"]


WEAPONS_OUTPUT = "weapons-detected"
CROPS_OUTPUT   = "bodies-croped"
FACES_OUTPUT   = "faces-detected"

def prepare_dirs(opt):
    outWPath = join(opt.outputPath, WEAPONS_OUTPUT)
    if opt.faces:
        outCPath = join(opt.outputPath, FACES_OUTPUT)
    else:
        outCPath = join(opt.outputPath, CROPS_OUTPUT)
        
    os.mkdir(outWPath)    
    os.mkdir(outCPath)
    
    return (outWPath, outCPath)

def get_files(opt):
    files = []
    if opt.img != "":
        files.append(opt.img)
    else:
        files = [join(opt.imgPath, f) for f in listdir(opt.imgPath) if isfile(join(opt.imgPath, f))]

    return files

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath',    '-p', type=str,            default="./data", help='Images input path')
    parser.add_argument('--img',        '-i', type=str,            default="",       help='Image  input file name')
    parser.add_argument('--outputPath', '-o', type=str,            default="./run",  help='Image processing output path')
    parser.add_argument('--faces',      '-f', action='store_true', default=False,    help="Face detection active.")
    parser.add_argument('--verbose',    '-v', action='store_true', default=False,    help="Information about the inference.")
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = parse_options()
    outWeaponsPath, outCropsPath = prepare_dirs(opt)
    files = get_files(opt)
    
    load_t0  = time.time()
    bodyOD   = YoloV5OD(WEIGHTS[0], conf_thres=0.2)
    load_t1  = time.time()
    weaponOD = YoloV5OD(WEIGHTS[1], conf_thres=0.2)
    load_t2  = time.time()
    if opt.faces:
        faceOD   = YoloV5OD(WEIGHTS[2], conf_thres=0.2)
    load_t3  = time.time()


    tot_faces = 0    
    inf_time, post_time, face_time  = 0, 0, 0
    for f in files:
        img = cv2.imread(f)
        if opt.verbose:
            height, width = img.shape[:2]
            print("Image '%s' (%d x %d): " % (f, height, width))
        
        #Inferece. Body and Weapons detection
        t0 = time.time()
        bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
        weapons = weaponOD.do_inference(img)
        bodies = []
        for obj in bodiesKnifes:
            bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
        t1 = time.time()
        inf_time += (t1 - t0)
        
        #Pairing bodies with weapons and crop bodies
        pairs = pairing_object_to_bodies(weapons, bodies) 
        bodies_crop = body_crop(pairs, weapons, bodies, img)
        t2 = time.time()
        post_time += (t2 - t1)
        
        #Save Image Results
        outFile = outWeaponsPath + "/out-" + os.path.basename(f)
        save_pair_results(pairs, weapons, bodies, img, outFile, verbose=opt.verbose)
        
        #Save Body Crops
        cropPath = outCropsPath + "/" + os.path.splitext(os.path.basename(f))[0]
        os.mkdir(cropPath)
        tot_faces += len(bodies_crop)
        for i, b in enumerate(bodies_crop):
            body_name = cropPath + "/body-"+str(i)+".jpg"
            if opt.faces:
                t0 = time.time()
                faces  = faceOD.do_inference(b[0])
                faces_crop = face_crop(faces, b[0])
                face_time += (time.time() - t0)
                faceOD.save_results(body_name, b[0], faces)
            else:
                cv2.imwrite(body_name, b[0])
            
        
            
    print("==========================================================")
    print("=              P R O C E S S     T I M I N G             =")
    print("==========================================================")
    print("Models Load Timing:")
    print("    [*] Weights '"+os.path.basename(WEIGHTS[0])+"' Loaded in: %0.2f(s)" % (load_t1 - load_t0))
    print("    [*] Weights '"+os.path.basename(WEIGHTS[1])+"' Loaded in: %0.2f(s)" % (load_t2 - load_t1))
    if opt.faces:
        print("    [*] Weights '"+os.path.basename(WEIGHTS[2])+"' Loaded in: %0.2f(s)" % (load_t2 - load_t1))        
    print("Timing Weapons and Bodies Inference (%d Images Processed):" % (len(files)))
    print("    [*] Inference Time per image  :  %0.2f(s) " % (inf_time / len(files)))
    print("    [*] Preprocess Time per image :  %0.2f(s) " % (post_time / len(files)))
    print("    [*] TOTAL Time per image      :  %0.2f(s) " % ((inf_time + post_time) / len(files)))
    if opt.faces:
        print("Timing Faces Inference (%d Bodies Processed):" % (tot_faces))
        print("    [*] TOTAL Time per image      :  %0.2f(s) " % (face_time / tot_faces))
    print("==========================================================")
