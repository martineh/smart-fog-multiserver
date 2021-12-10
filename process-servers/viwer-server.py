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

#Deep learning
from deepLearning.objectDetection.detect import YoloV5OD
from deepLearning.objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop, face_crop

#------------ Global Variables ------------#
ROOT_WEIGHTS = "./deepLearning/objectDetection/yoloV5-weights/"
ROOT_FACES_DB = "./deepLearning/faceIdentify/faces_database/"

WEIGHTS = [ROOT_WEIGHTS+"yolov5s.pt",
           ROOT_WEIGHTS+"weapons-new-train.pt",
           ROOT_WEIGHTS+"face_detection_yolov5s.pt"]

bodyOD    = None
weaponOD  = None
faceOD    = None
faceIdent = None

PYTHON    = 1
C         = 0

task_queue   = queue.Queue(20)

PACK_SIZE    = 50000

FORMAT       = 'utf-8'
imgData_conn = {} #Dicctionary for each host and its image
totImg_conn  = {} #Dicctiorary for each host and the total number of image received

PRINT_LIM    = 10

#- Input variables

level        = 1
deepL        = True
mode         = C

#--- For a good Debug --#
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

#----- Function and Classes Definition -----#
def load_Deep_Learning():
    #Deep learning
    from deepLearning.objectDetection.detect import YoloV5OD
    from deepLearning.objectDetection.detect import pairing_object_to_bodies, save_pair_results, body_crop, face_crop

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
        
    bodiesKnifes  = bodyOD.do_inference(img, class_filter=['person'])
    weapons = weaponOD.do_inference(img, class_filter=['pistol'])

    bodies = []
    for obj in bodiesKnifes:
        bodies.append(obj) if obj.name == 'person' else weapons.append(obj) 
        
    #Pairing bodies with weapons and crop bodies
    pairs = pairing_object_to_bodies(weapons, bodies) 
    #bodies_crop = body_crop(pairs, weapons, bodies, img)
    
    #Save Image Results
    img_result = save_pair_results(pairs, weapons, bodies, img)

    return img_result


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

class imgDataConn():
    def __init__(self):
        self.rows       = 0
        self.cols       = 0
        self.channels   = 0
        self.depth      = 0
        self.total_size = 0
        self.data_remaining = 0
        self.array      = None
        self.data_tmp   = None
        
    def init(self, data):
        if self.data_tmp is not None:
            bits_to_parse = 6 - len(self.data_tmp)
            data_to_parse = self.data_tmp + data[:bits_to_parse]
            self.data_tmp = None
        else:
            bits_to_parse = 6
            data_to_parse = data
            
        self.rows, self.cols, self.channels, self.depth = self.get_size_from_data(data_to_parse)
        self.array = self.to_array(data[bits_to_parse:])
        self.set_total_size()
        self.data_remaining = self.total_size - len(data[bits_to_parse:])
        
    def reset(self):
        self.rows           = 0
        self.cols           = 0
        self.channels       = 0
        self.depth          = 0
        self.total_size     = 0
        self.data_remaining = 0
        self.array          = None
        self.data_tmp       = None
        
    def to_array(self, data):
        return np.frombuffer(data, dtype=np.uint8, count=-1)

    def parse_data(self, data):
        img = None
        if self.total_size == 0:
            self.init(data)
        else:
            img = self.conc_array(data)            
        return img
    
    def conc_array(self, data):
        if self.data_remaining < len(data):
            size_conc = self.data_remaining
            array_tmp = self.to_array(data[:size_conc])
            init      = True
        else:
            size_conc = len(data)
            array_tmp = self.to_array(data)
            init      = False
            
        self.array = np.concatenate((self.array, array_tmp), axis=None)
        self.data_remaining = self.data_remaining - size_conc
        imgFinal = None
        if self.is_completed():
            img = self.to_img()
            imgFinal = img.copy()
            self.reset()
            if init:
                if (len(data[size_conc:]) < 6):                    
                    self.data_tmp = data[size_conc:]
                else:
                    self.init(data[size_conc:])
                #self.init(data[size_conc:])
        return imgFinal
        
    def to_img(self):
        img = self.array.reshape(self.rows, self.cols, self.channels)
        return img

    def is_completed(self):
        return self.data_remaining == 0

    def set_total_size(self):
        self.total_size = self.rows * self.cols * self.channels

    def get_size_from_data(self, data):
        rows     = int.from_bytes(data[0:2], byteorder="little", signed=False )
        cols     = int.from_bytes(data[2:4], byteorder="little", signed=False )
        channels = int.from_bytes(data[4:5], byteorder="little", signed=False )
        depth    = int.from_bytes(data[5:6], byteorder="little", signed=False )
        
        return rows, cols, channels, depth
    
    def __str__(self):
        return "<Image (" + str(self.rows) + "x" + str(self.cols) + "x" + str(self.channels) \
            + ")> [Total Size: " + str(self.total_size) + ", Remaining Size: " + str(self.data_remaining) + "]"
    


def process_data(id_conn, data):
    final_img = None    
    #New connection if not found in the dictionary
    if id_conn not in imgData_conn:
        imgData_conn[id_conn] = imgDataConn()
        totImg_conn[id_conn]  = 0
        
    actualImg = imgData_conn[id_conn]
    img = actualImg.parse_data(data)
    
    if img is not None:
        totImg_conn[id_conn] += 1

    return img

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


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
def handle_client_Py(conn, addr):
    log = log_msg()
    print(log+"CLIENT connected from Host "+addr[0]+" ("+str(addr[1])+") [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")
    
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])
    
    while connected:
        length = recvall(conn, 16).decode()
        stringData = recvall(conn, int(length))
        data = np.frombuffer(stringData, dtype='uint8')
        img=cv2.imdecode(data,1)        
        #Process Image (Neuronal Network)
        if deepL:
            img = apply_Deep_Learning(img)
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    conn.close()
    
def handle_client_C(conn, addr):
    log = log_msg()
    print(log+"CLIENT connected from Host "+addr[0]+" ("+str(addr[1])+") [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")        
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])
    
    while connected:
        data = conn.recv(PACK_SIZE)
        if data:
            img = process_data(id_conn, data)
            if img is not None:
                if deepL:
                    #Process Image (Neuronal Network)
                    img = apply_Deep_Learning(img)
                cv2.imshow('webcam', img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    conn.close()


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


#-- MAIN MENU
#-- Server Python That Recives From Multiples C++-Clients
#-- And Sends to Other Python-Server
if __name__ == "__main__":
    #Input arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--onlySend',  '-s', action='store_true', required=False, help="Only for sending frames. Not deep learning.")
    parser.add_argument('--py', '-p', action='store_true', required=False, help="Enable python viwer")
    
    args = parser.parse_args() 
    deepL = not args.onlySend
    if args.py:
        mode  = PYTHON

    I_am_in = "[Jetson-" + str(level)+" (Py)]" if level < 3 else "[Cloud (Py)]"
    print(" ************************************************* ")
    print("|        I  N  P  U  T    P  A  R  A  M  S        |")
    print(" ************************************************* ")
    print("  >Deep Learning     : " + str(deepL)               )
    if mode == C:
        print("  >Mode              : C"                       )
    else:
        print("  >Mode              : Python"                  )
    print(" -------------------------------------------------" )
    print("                 [SCHEME LEVEL]                   " )
    print(" -------------------------------------------------" )
    print("          [WEAPON AND BODY DETECTION]             " )
    print(" *************************************************\n")
    
    addresses = []
    addresses.append(("None", 0)) #Ip and Port From level-0
    fd_conf = open("address.config", "r")
    for line in fd_conf:
        if line[0] != "#":
            sp = line.split(";")
            addresses.append((sp[1], int(sp[2])))
    fd_conf.close()

    print(addresses)    
    server_ip    = "0.0.0.0"
    server_port  = addresses[1][1]
    server_addr  = (server_ip, server_port)

    load_Deep_Learning()
    
    if mode == PYTHON:
        server_start(handle_client_Py, server_addr)
    else:
        server_start(handle_client_C, server_addr)
