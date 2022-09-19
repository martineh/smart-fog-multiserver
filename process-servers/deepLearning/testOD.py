import sys
import cv2
import time
import argparse
import os
from os import listdir
from os.path import isfile, join

from objectDetection.detect import YoloV5OD
from objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop, face_crop
from faceIdentify.detect    import faceIdentity

ROOT_WEIGHTS  = "./objectDetection/yoloV5-weights/"
ROOT_FACES_DB = "./faceIdentify/faces_database/"

WEIGHTS = [ROOT_WEIGHTS+"yolov5s.pt",
           ROOT_WEIGHTS+"weapons-v5s-new.pt",           
           ROOT_WEIGHTS+"face_detection_yolov5s.pt"]

#WEIGHTS_ALL = ROOT_WEIGHTS + "bodies-weapons-300epc.pt"

WEAPONS_OUTPUT = "weapons-detected"
CROPS_OUTPUT   = "bodies-croped"
FACES_OUTPUT   = "faces-detected"

def sortFiles(e):
    return int(e.split("/")[2].split(".")[0][1:])

def prepare_dirs(opt):
    outWPath = join(opt.outputPath, WEAPONS_OUTPUT)
    if opt.faces:
        outCPath = join(opt.outputPath, FACES_OUTPUT)
    else:
        outCPath = join(opt.outputPath, CROPS_OUTPUT)

    if opt.video == "":
        os.mkdir(outWPath)    
        os.mkdir(outCPath)
    
    return (outWPath, outCPath)

def get_files(opt):
    files = []
    if opt.img != "":
        files.append(opt.img)
    elif opt.video != "":
        files.append(opt.video)
    else:
        files = [join(opt.imgPath, f) for f in listdir(opt.imgPath) if isfile(join(opt.imgPath, f))]
    
    if opt.video == "":
        files.sort(key=sortFiles)

    return files

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath',    '-p', type=str,            default="./data", help='Images input path')
    parser.add_argument('--img',        '-i', type=str,            default="",       help='Image  input file name')
    parser.add_argument('--video',      '-e', type=str,            default="",       help='Video  input file name')
    parser.add_argument('--outputPath', '-o', type=str,            default="./run",  help='Image processing output path')
    parser.add_argument('--faces',      '-f', action='store_true', default=False,    help="Face detection active.")
    parser.add_argument('--verbose',    '-v', action='store_true', default=False,    help="Information about the inference.")
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = parse_options()
    outWeaponsPath, outCropsPath = prepare_dirs(opt)
    files = get_files(opt)

    #weaponsBodiesOD = YoloV5OD(WEIGHTS_ALL, conf_thres=0.3)
    load_t0  = time.time()
    bodyOD   = YoloV5OD(WEIGHTS[0], conf_thres=0.3)
    load_t1  = time.time()
    weaponOD = YoloV5OD(WEIGHTS[1], conf_thres=0.4)
    load_t2  = time.time()

    #if opt.faces:
    faceOD   = YoloV5OD(WEIGHTS[2], conf_thres=0.3)
    faceIdentity = faceIdentity(ROOT_FACES_DB)
        
    load_t3  = time.time()


    tot_bodies, tot_faces = 0, 0
    inf_time, post_time, face_time, identify_time  = 0, 0, 0, 0

    nFrames = 0
    if opt.video != "":
        f = files[0]
        print(f)
        cap = cv2.VideoCapture(f)

        outFile = opt.outputPath + "/out-video.avi"
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.          
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            #Inferece. Body and Weapons detection
            t0 = time.time()
            #weapons_bodies = weaponsBodiesOD.do_inference(img)
            bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
            weapons = weaponOD.do_inference(img)
            bodies  = []
            for obj in bodiesKnifes:
                bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
            t1 = time.time()
            inf_time += (t1 - t0)
        
            #Pairing bodies with weapons and crop bodies
            pairs = pairing_object_to_bodies(weapons, bodies) 
            t2 = time.time()
            post_time += (t2 - t1)
        
            #Save Image Results
            save_pair_results(pairs, weapons, bodies, img, "", verbose=opt.verbose, video=out)
            nFrames += 1
        out.release()
        cap.release()
    else:
        tinit = time.time()
        for idf, f in enumerate(files):
            img = cv2.imread(f)
            print("")
            print("----------------------------------------------------------")
            print("Image %d '%s': " % (idf + 1, f))
            print("----------------------------------------------------------")
            
            t0 = time.time()
            #Inferece. Body and Weapons detection
            bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
            weapons = weaponOD.do_inference(img)            
            t1 = time.time()
            inf_time += (t1 - t0)

            bodies = []
            for obj in bodiesKnifes:
                bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
        
            #Pairing bodies with weapons and crop bodies
            pairs = pairing_object_to_bodies(weapons, bodies) 
            bodies_crop = body_crop(pairs, weapons, bodies, img)

            
            #Save Body Crops
            #cropPath = outCropsPath + "/" + os.path.splitext(os.path.basename(f))[0]
            #os.mkdir(cropPath)
            
            tot_bodies += len(bodies_crop)
            tot_img_faces = 0
            tot_img_identifies = 0
            idents = ""
            for i, b in enumerate(bodies_crop):
                t0 = time.time()
                faces  = faceOD.do_inference(b[0])
                faces_crop = face_crop(faces, b[0])
                face_time += (time.time() - t0)
                tot_faces += len(faces_crop)
                tot_img_faces += len(faces_crop)
                for j, face in enumerate(faces_crop):
                    t0 = time.time()
                    identities = faceIdentity.identify(face)
                    identify_time  += (time.time() - t0)
                    tot_img_identifies += len(identities)
                    if len(identities) > 0:
                        ident = identities[0]
                    else:
                        ident = "Unknown"
                    print("      [*] Face Identify: %s " % (ident)) 
                    idents += "#F" + str(i + j) + ": " + ident + ","
            
            if tot_img_faces == 0:
                idents = "Not Faces.";

            #Save Image Results
            outFile = outWeaponsPath + "/out-" + os.path.basename(f)
            save_pair_results(pairs, weapons, bodies, img, idents, outFile)
        tend  = time.time()
        nFrames = len(files)

    print("")
    print("==========================================================")
    print("=              P R O C E S S     T I M I N G             =")
    print("==========================================================")
    print(" Total Images processed                         : %d "       % (nFrames))
    print(" Time Per Image                                 : %0.5f(s) " % ((tend - tinit) / nFrames))
    print("   [*] Inference Time Per Call (Body + weapon)  : %0.5f(s) " % (inf_time / (nFrames * 2)))
    print("       [*] Total Calls                          : %d "       % (nFrames * 2))
    print("   [*] Inference Time Per Call (Face Detection) : %0.5f(s) " % (face_time / tot_bodies))
    print("       [*] Total Calls                          : %d "       % (tot_bodies))
    print("   [*] Inference Time Per Call (Face Identify)  : %0.5f(s) " % (identify_time / tot_faces))
    print("       [*] Total Calls                          : %d "       % (tot_faces))
    print("==========================================================")
