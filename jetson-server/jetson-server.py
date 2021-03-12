#! /usr/bin/python3
# -*- coding: utf-8 -*-

#------------------ Imports ----------------#
import socket 
import threading
import numpy as np
import cv2
import time as t
import sys
import os
import shutil
import deepLearning.weaponDetector.detector as wd

from datetime import datetime


#------------ Global Variables ------------#
PRINT_LIMIT  = 10
PACK_SIZE    = 50000
PORT         = 5050
SERVER       = "0.0.0.0" #Recive from any source (HOST)
ADDR         = (SERVER, PORT)
FORMAT       = 'utf-8'
imgData_conn = {} #Dicctionary for each host and its image
totImg_conn  = {} #Dicctiorary for each host and the total number of image received

#- Input variables
debug        = False
write        = False


#----- Function definition and classes ----#
def start_deepLearning(img):
    weapon_detected = wd.detect_weapons(img, debug=False)
    if weapon_detected:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        print("[" + date_time + "] Weapon Detected.")
    
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

    
def handle_client(conn, addr):
    if debug:
        print(f"[NEW CONNECTION] {addr} connected.")
    tot_img_recv = 0
    connected    = True
    id_conn = addr[0] + ":" + str(addr[1])

    #Path to write images
    if write: 
        if os.path.isdir(id_conn):
            shutil.rmtree(id_conn)        
        os.mkdir(id_conn)
  
    if debug:
        time_start   = t.time()
        
    while connected:
        data = conn.recv(PACK_SIZE)
        if data:
            img = process_data(id_conn, data)
            if img is not None:
                #Process Image (Neuronal Network)
                #start_deepLearning(img)
                if debug:
                    tot_img_recv += 1
                    if PRINT_LIMIT == tot_img_recv:
                        time_stop = t.time()
                        total_time = time_stop - time_start
                        print ("[INFO] (" + id_conn + "): " + str(round(tot_img_recv / total_time, 2))
                               + "(fps), Total Img Recived: " + str(totImg_conn[id_conn]))
                        tot_img_recv = 0
                        time_start = t.time()
                if write:
                    #cv2.imwrite(id_conn+"/"+str(totImg_conn[id_conn])+".jpg", img)
                    cv2.imshow("img", img)
                    cv2.waitKey(30)
                del img
                
    conn.close()
        

def start():
    server.listen()
    if debug:
        print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        if debug:
            print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


#-------- Initial code   --------#
if __name__ == "__main__":
    #Input arguments
    if len(sys.argv) > 1:
        for cmd in sys.argv:
            if cmd in ['-d', '--debug']:
                debug = True
            if cmd in ['-w', '--write']:
                write = True

    fLog = open("run.log", "w")
    #Socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    print("[START] Starting server...")
    start()

    fLog.close()
