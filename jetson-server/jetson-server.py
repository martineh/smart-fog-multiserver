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
from datetime import datetime

#Deep learning
import deepLearning.detection as dL

#------------ Global Variables ------------#
task_queue   = queue.Queue(20)

PACK_SIZE    = 50000

FORMAT       = 'utf-8'
imgData_conn = {} #Dicctionary for each host and its image
totImg_conn  = {} #Dicctiorary for each host and the total number of image received

PRINT_LIM    = 30
#- Input variables
debug        = False
write        = False
level        = 0
deepL        = True

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
def log_msg():
    now = datetime.now()
    t_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return "["+bcolors.WARNING+"LOG"+bcolors.ENDC+"]["+bcolors.BOLD+t_string+bcolors.ENDC +"] "

def timer_msg():
    now = datetime.now()
    t_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return "["+bcolors.OKBLUE+"TIMING"+bcolors.ENDC+"]["+bcolors.BOLD+t_string+bcolors.ENDC +"] "

def timing_handler(frameTot, tTot):
    log = timer_msg()
    print("%s%s%d%s frames processed in %s%0.3f(s)%s [%s%0.3f(fps)%s] " %
          (log,
           bcolors.BOLD, frameTot, bcolors.ENDC,
           bcolors.BOLD, tTot, bcolors.ENDC,
           bcolors.OKCYAN, frameTot / tTot, bcolors.ENDC))

        
def apply_deepLearning(image):
    if level == 1:
        #weapong and body detection
        info, result = dL.body_weapon_detector(image)
    elif level == 2:
        #face detection
        result = dL.face_detector(image)
    elif level == 3:
        #face identify
        names = dL.face_identify(image)
        result = [image, names]

    if debug:
        if level == 1:
            if len(result) > 0:
                log = log_msg()
                print(log + "DEEP LEARNING A person with a weapon detected: "+ bcolors.OKCYAN + 
                      "[Guns:" + str(info["gun"]) + ", " + "Knifes:" + str(info["knife"]) + "]" + bcolors.ENDC)
        elif level == 2:
            log = log_msg()
            if result is not None:
                print(log + "DEEP LEARNING Face detected.")
            else:
                print(log + "DEEP LEARNING Person received but face NOT detected.")
        elif level == 3:
            log = log_msg()
            if len(result) > 0:
                print(log + "DEEP LEARNING Face identified: " + bcolors.OKCYAN + str(result[1]) + bcolors.ENDC)
            else:
                print(log + "DEEP LEARNING Face received but NOT identified.")                
    return result

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
    

#-----  Other functions  -----#
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


#------------ Handler for each Thread ------------#
def handle_client_Py(conn, addr):
    log = log_msg()
    print(log+"CLIENT connected from Host "+addr[0]+" ("+str(addr[1])+") [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")
    
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])
    
    frameLim = 0
    frameTot = 0
    t0 = time.time()
    while True:
        length = recvall(conn, 16).decode()
        stringData = recvall(conn, int(length))
        data = np.frombuffer(stringData, dtype='uint8')
        decimg=cv2.imdecode(data,1)
        if deepL:
            #Process Image (Neuronal Network)
            output = apply_deepLearning(decimg)
            if output is not None:
                task_queue.put(output)
        else:
            task_queue.put(decimg)
        frameLim += 1
        frameTot += 1
        if frameLim == PRINT_LIM:
            t_tot = (time.time() - t0)
            t0  = time.time()
            timing_handler(frameTot, t_tot)
            frameLim = 0
            frameTot = 0
    conn.close()
    
def handle_client_C(conn, addr):
    log = log_msg()
    print(log+"CLIENT connected from Host "+addr[0]+" ("+str(addr[1])+") [" + bcolors.OKGREEN + "OK" + bcolors.ENDC+"]")        
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])
    
    frameLim = 0
    frameTot = 0
    t0 = time.time()
    while connected:
        data = conn.recv(PACK_SIZE)
        if data:
            img = process_data(id_conn, data)
            if img is not None:
                #Process Image (Neuronal Network)
                if deepL:
                    list_output = apply_deepLearning(img)
                    for image in list_output:
                        task_queue.put(image)
                else:
                    task_queue.put(img)
                frameLim  += 1
                frameTot += 1
                if frameLim == PRINT_LIM:
                    t_tot = (time.time() - t0)
                    t0    = time.time()
                    timing_handler(frameTot, t_tot)
                    frameLim = 0
                    frameTot = 0

    conn.close()


def handle_send_to_Py(client):
    connected    = True
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    
    while connected:
        frame = task_queue.get()
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        #stringData = data.tostring()
        client.send( str(data.size).ljust(16).encode());
        client.send( data );        
        #cv2.imshow('CLIENT', frame)
        #cv2.waitKey(30)
        del frame
        del data        
        task_queue.task_done()
        #del imgencode, frame, data, stringData
        #print ("[INFO] CLIENT: Remove Frame from queue(" + str(task_queue.qsize()) + ")")    

def handle_send_backend(client):
    connected    = True
    #Path to write images
    if write:
        output_path = "output"
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)        
        os.mkdir(output_path)
        frameId = 0
        
    while connected:
        output = task_queue.get()
        if write:
            if not deepL:
                cv2.imwrite(output_path+"/frame-"+str(frameId)+".jpg", output)
            else:
                tag = "Unknown"
                if len(output[1]) > 0:
                    tag = output[1][0]
                cv2.imwrite(output_path+"/face-"+str(frameId)+"-"+tag+".jpg", output[0])
            frameId += 1
        task_queue.task_done()

    
#------------ Thread Starters ------------#
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


#-- MAIN MENU
#-- Server Python That Recives From Multiples C++-Clients
#-- And Sends to Other Python-Server
if __name__ == "__main__":
    #Input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',  '-d', action='store_true', required=False, help="Enable debug messages")
    parser.add_argument('--onlySend',  '-s', action='store_true', required=False, help="Only for sending frames. Not deep learning.")
    parser.add_argument('--output', '-o', action='store_true', required=False, help="Enable output images")
    parser.add_argument('--level',  '-l', required=True, type=int, help="Indicates the level to run it. (Jetson 1, Jetson 2,..)")
    
    args = parser.parse_args()
    
    debug = args.debug
    write = args.output
    level = args.level
    deepL = not args.onlySend
    
    scheme_str = ""
    if level == 1:
        scheme_str = "[]-[RasPi (C++)]---->[Jetson-1 (Py)]" 
    elif level == 2:
        scheme_str = "[Jetson-1 (Py)]----->[Jetson-2 (Py)]" 
    elif level == 3:
        scheme_str = "[Jetson-2 (Py)]----->[Cloud (Py)]" 
    else:
        print("[ERROR] Level out of range [1-3]")
        sys.exit(-1)

    I_am_in = "[Jetson-" + str(level)+" (Py)]" if level < 3 else "[Cloud (Py)]"
    print(" ************************************************* ")
    print("|        I  N  P  U  T    P  A  R  A  M  S        |")
    print(" ************************************************* ")
    print("  >Debug             : " + str(debug))
    print("  >Output            : " + str(write))
    print("  >Deep Learning     : " + str(deepL))
    print("  >Level(you are at) : " + str(level) + " " + I_am_in+"")
    print(" -------------------------------------------------")
    print("                 [SCHEME LEVEL]                   ")
    print(" -------------------------------------------------")
    print("        " + scheme_str                             )
    print("                                                  ")
    if level == 1:
        print("          [WEAPON AND BODY DETECTION]            ")
    if level == 2:
        print("               [FACE DETECTION]               ")
    if level == 3:
        print("             [FACE IDENTIFCATION]             ")
    print(" *************************************************\n")
    
    #-------------------------------------------------------------------
    #                       RUN SCHEME TCP/IP                          |
    #-------------------------------------------------------------------
    # []-[RaspBerry Pi]----->[Jetson-1]----->[Jetson-2]----->[Cloud]   |
    #     Levl-0:C++       ->Levl-1:Py     ->Levl-2:Py     ->Levl-3:Py |
    #------------------------------------------------------------------
    #     (Moviment)           (Body)        (Weapon)         (Face)   |            
    #------------------------------------------------------------------

    #Reading IPs and Ports file configuration
    #Level;Ip;Port
    addresses = []
    addresses.append(("None", 0)) #Ip and Port From level-0
    fd_conf = open("address.config", "r")
    for line in fd_conf:
        if line[0] != "#":
            sp = line.split(";")
            addresses.append((sp[1], int(sp[2])))
    fd_conf.close()
    
    #SERVERS: IP and Port Configuration
    #IP FIXED FOR ALL SERVER!!. All of theme recive from any source
    server_ip    = "0.0.0.0"
    server_port  = addresses[level][1]    

    server_addr  = (server_ip, server_port)

    #CLIENTS: IP and Port Configuration
    if level == 3:
        client_ip   = "None"
        client_port = -1
    else:
        client_ip   = addresses[level + 1][0]
        client_port = addresses[level + 1][1]        
    client_addr  = (client_ip, client_port)


    #--Client-Python That Sends To Python Server or Finishes the Pipeline
    if level == 3:
        # START CLIENT: [Py](Back-end)
        client_start(handle_send_backend, client_addr);
    else:
        # START CLIENT: [Py] --->
        client_start(handle_send_to_Py, client_addr);

        
    #--Server-Python That Receives from C++/Py
    if level == 1:
        # START SERVER: [C++] -> [Py]
        server_start(handle_client_C, server_addr)
    else:
        # START SERVER: [Py]  -> [Py]
        server_start(handle_client_Py, server_addr)
