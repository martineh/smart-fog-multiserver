#! /usr/bin/python3
# -*- coding: utf-8 -*-

#------------------ Imports ----------------#
import socket 
import threading, queue
import numpy as np
import cv2
import time 
import sys
import os
import shutil
import argparse
from   datetime import datetime
import threading



#------------ Global Variables ------------#
bodyOD    = None
weaponOD  = None
faceOD    = None
faceIdent = None

task_queue   = queue.Queue(20)

PACK_SIZE    = 50000

FORMAT       = 'utf-8'
imgData_conn = {} #Dicctionary for each host and its image
totImg_conn  = {} #Dicctiorary for each host and the total number of image received

PRINT_LIM    = 10

#- Input variables
debug        = False
write        = False
level        = 0
deepL        = True

#- Deep Learning
bodyOD       = None
weaponOD     = None
faceOD       = None

################## D E E P    L E A R N I N G    F U N C T I O N S ######################
ROOT_WEIGHTS  = "./objectDetection/yoloV5-weights/"
ROOT_FACES_DB = "./faceIdentify/faces_database/"

WEIGHTS = [ROOT_WEIGHTS+"yolov5s.pt",
           ROOT_WEIGHTS+"weapons-YOLOv5s-300epc.pt",
           ROOT_WEIGHTS+"face_detection_yolov5s.pt"]

def load_Deep_Learning():
    #Deep learning
    from objectDetection.detect import YoloV5OD
    from objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop, face_crop

    global bodyOD
    global weaponOD
    bodyOD   = YoloV5OD(WEIGHTS[0], conf_thres=0.3)
    weaponOD = YoloV5OD(WEIGHTS[1], conf_thres=0.4)
    
def apply_Deep_Learning(img):
    global bodyOD
    global weaponOD
    
    tot_bodies, tot_faces = 0, 0
    inf_time, post_time, face_time, identify_time  = 0, 0, 0, 0

    nFrames = 0
        
    bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person', 'knife'])
    weapons = weaponOD.do_inference(img)

    bodies = []
    for obj in bodiesKnifes:
        bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
        
    #Pairing bodies with weapons and crop bodies
    pairs = pairing_object_to_bodies(weapons, bodies) 
    #bodies_crop = body_crop(pairs, weapons, bodies, img)
    
    #Save Image Results
    img_result = save_pair_results(pairs, weapons, bodies, img)

    return img_result



################## L O G    F U N C T I O N S ######################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_msg():
    now = datetime.now()
    t_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return "["+bcolors.WARNING+"LOG"+bcolors.ENDC+"]["+bcolors.BOLD+t_string+bcolors.ENDC +"] "

def timer_msg():
    now = datetime.now()
    t_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return "["+bcolors.OKBLUE+"TIMING"+bcolors.ENDC+"]["+bcolors.BOLD+t_string+bcolors.ENDC +"] "

def timing_handler(frameTot, tTot, fps):
    log = timer_msg()
    if tTot <= 0:
        tTot = 1
    print("%s%s%d%s frames processed in %s%0.3f(s)%s [%s%0.3f(fps)%s] " %
          (log,
           bcolors.BOLD, frameTot, bcolors.ENDC,
           bcolors.BOLD, tTot, bcolors.ENDC,
           bcolors.OKCYAN, fps, bcolors.ENDC),
           flush=True,
           end='\r')

    
################## S E R V E R    F U N C T I O N S ######################

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


#*- SERVER STARTER -*#
def server_start(handler, server_addr):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(server_addr)
        server.listen()
    except socket.error as exc:
        log = log_msg()
        print(log+"SERVER can't start on "+server_addr[0]+":"+str(server_addr[1])+" [" + bcolors.FAIL + "FAIL" + bcolors.ENDC+"]")
        print(exc)
        os._exit(1)
        
    log = log_msg()
    print(log+"SERVER started on "+server_addr[0]+":"+str(server_addr[1])+" [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handler, args=(conn, addr))
        thread.start()
        log = log_msg()
        print(log+"SERVER new connection. Active connections [" + bcolors.OKGREEN +
              str(threading.activeCount() - 1) + bcolors.ENDC +"]") #


#*- HANDLER -*#
def server_OD(conn, addr):
    log = log_msg()
    print(log+"CLIENT connected from Host "+addr[0]+" ("+str(addr[1])+") [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")
    
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])
    
    #first = True
    #frameTot = 0
    while True:
        length = recvall(conn, 16).decode()
        stringData = recvall(conn, int(length))
        data = np.frombuffer(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1)

        #Process Image (Neuronal Network)
        imgOD = apply_Deep_Learning(decimg)
        #imgOD = decimg
        cv2.imshow('webcam', imgOD)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    conn.close()

################## C L I E N T    F U N C T I O N S ######################

#*- CLIENT STARTER -*#
def client_start(handler, client_addr):
    if client_addr[1] != -1:
        try:
            client = socket.socket()
            client.connect(client_addr)
            log = log_msg()
            print(log+"CLIENT started on "+client_addr[0]+":"+str(client_addr[1])+" [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")
        except socket.error as exc:
            log = log_msg()
            print(log+"CLIENT can't start on "+client_addr[0]+":"+str(client_addr[1])+" [" + bcolors.FAIL + "FAIL" + bcolors.ENDC+"]")
            print(exc)
            os._exit(1)         
    else:
        client = None
        log = log_msg()
        print(log+"BACK-END running [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")

    thread = threading.Thread(target=handler, args=(client,))
    thread.start()


#*- HANDLER -*#
def img_sender(client):
    connected    = True
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    
    frameTot = 0
    t_global = 0
    t0 = time.time()
    while connected:
        frame = task_queue.get()
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        client.send( str(data.size).ljust(16).encode());
        client.send( data );
        del frame
        del data      
        task_queue.task_done()
        frameTot += 1
        if (frameTot % PRINT_LIM) == 0:
            t_tot = (time.time() - t0)
            t_global += t_tot
            timing_handler(frameTot, t_global, PRINT_LIM / t_tot)
            t0 = time.time()
            

def webcamCapture():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        task_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return

def piWebcamCapture():
    import picamera
    import picamera.array

    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as output:
            camera.resolution = (640, 480)
            camera.framerate  = 30
            while True:
	        camera.capture(output, 'bgr')
                task_queue.put(output.array)
                output.truncate(0)

    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE ERROR: %d (--client/--server)" % (sys.argv[0]))
        sys.exit(-1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--server',  '-s', action='store_true', required=False, help="Enable server")
    parser.add_argument('--client',  '-c', action='store_true', required=False, help="Enable client")
    parser.add_argument('--pi',      '-p', action='store_true', required=False, help="Enable Raspberry Pi Webcam")
    
    args = parser.parse_args()
    
    server = args.server
    client = args.client
    pi =   = args.pi
    
    fd_conf = open("address.config", "r")
    for line in fd_conf:
        if line[0] != "#":
            sp = line.split(";")
            client_ip, client_port = sp[0], int(sp[1])
    fd_conf.close()

    if client:
        client_addr  = (client_ip, client_port)    
        client_start(img_sender, client_addr);
        if pi:
            piWebcamCapture()
        else:
            webcamCapture()
    else:
        server_ip    = "0.0.0.0"
        server_addr  = (server_ip, client_port)

        load_Deep_Learning()
        server_start(server_OD, server_addr)
