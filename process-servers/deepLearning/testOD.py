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

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath',    '-p', type=str,            default="./data", help='Images input path')
    parser.add_argument('--img',        '-i', type=str,            default="",       help='Image  input file name')
    parser.add_argument('--outputPath', '-o', type=str,            default="./run",  help='Image processing output path')
    parser.add_argument('--video',      '-e', type=str,            default="",       help='Video  input file name')
    parser.add_argument('--faces',      '-f', action='store_true', default=False,    help="Face detection active.")
    parser.add_argument('--verbose',    '-v', action='store_true', default=False,    help="Information about the inference.")
    parser.add_argument('--webcam',     '-w', action='store_true', default=False,    help='Video from webcam/Server')
    opt = parser.parse_args()
    
    return opt

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
    
    #if opt.video == "":
    #    files.sort(key=sortFiles)

    return files

def inference_pipeline(img):
    t0 = time.time()
    
    #Weapons and Bodies Inferece. Body and Weapons detection
    #weapons_bodies = weaponsBodiesOD.do_inference(img)
    bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
    weapons       = weaponOD.do_inference(img)
    bodies        = []
    for obj in bodiesKnifes:
        bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
     
    #Pairing bodies with weapons and cropping bodies
    pairs       = pairing_object_to_bodies(weapons, bodies) 
    bodies_crop = body_crop(pairs, weapons, bodies, img)

    identities=[]
    #For each body crop the face and identify the face
    for i, b in enumerate(bodies_crop):
        faces = faceOD.do_inference(b[0])
        faces_crop = face_crop(faces, b[0])
        for j, face in enumerate(faces_crop):
            ident = faceIdentity.identify(face)
            if len(ident) > 0 : identities.append(ident[0])  
            
    #End All pipeline Inference
    inf_time = time.time() - t0

    return pairs, weapons, bodies, identities, inf_time 


def parse_identities_text(identities, pairs):
    ident_txt = "Bodies With Weapons Detected!:"
    if len(identities) > 0:
        for i, ident in enumerate(identities):
            ident_txt += " #" + str(i) + ":" + ident + ","
    else:
        if len(pairs) > 0:
            ident_txt = "Bodies With Weapons Detected! But NOT Identified. "
        else:
            ident_txt = "NOT Bodies With Weapons Identified. "

    return ident_txt

if __name__ == "__main__":
    opt = parse_options()
    #outWeaponsPath, outCropsPath = prepare_dirs(opt)
    files = get_files(opt)

    bodyOD   = YoloV5OD(WEIGHTS[0], conf_thres=0.3)
    weaponOD = YoloV5OD(WEIGHTS[1], conf_thres=0.4)
    faceOD   = YoloV5OD(WEIGHTS[2], conf_thres=0.3)

    faceIdentity = faceIdentity(ROOT_FACES_DB) 

    tot_time      = 0.0
    nFrames       = 0
    
    if opt.webcam != "":
        #cap = cv2.VideoCapture('rtsp://192.168.1.245:51503/19FFC3852035457401E0D35E0499725E_1 RTSP/1.0')
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            pairs, weapons, bodies, identities, inf_time = inference_pipeline(frame)
            
            idents_str = parse_identities_text(identities, pairs)

            #Save Image Results
            save_pair_results(pairs, weapons, bodies, frame, idents_str, display="frame")
            
            nFrames  += 1
            tot_time += inf_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif opt.video != "":
        f = files[0]
        print(f)
        cap = cv2.VideoCapture(f)

        outFile = opt.outputPath + "/out-video.avi"
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
    
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.          
        out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret : break
            pairs, weapons, bodies, identities, inf_time = inference_pipeline(frame)
            
            idents_str = parse_identities_text(identities, pairs)

            #Save Image Results
            save_pair_results(pairs, weapons, bodies, frame, idents_str, outVideo=out)
            
            nFrames  += 1
            tot_time += inf_time

        out.release()
        cap.release()
    else:
        for idf, f in enumerate(files):
            frame = cv2.imread(f)

            print("[*] Process Image %d '%s'." % (idf + 1, f))
            pairs, weapons, bodies, identities, inf_time = inference_pipeline(frame)
            idents_str = parse_identities_text(identities, pairs)
            
            #Save Image Results
            outFile = opt.outputPath + "/out-" + os.path.basename(f)
            save_pair_results(pairs, weapons, bodies, frame, idents_str, outImage=outFile)
        
            nFrames  += 1
            tot_time += inf_time

    print("")
    print("==========================================================")
    print("=              P R O C E S S     T I M I N G             =")
    print("==========================================================")
    print(" [*] Total Images processed        : %d "       % (nFrames))
    print(" [*] Inference Time                : %0.5f(s) " % (tot_time / nFrames))
    print(" [*] FPS                           : %0.5f(s) " % (1 / (tot_time / nFrames)))
    print("==========================================================")

