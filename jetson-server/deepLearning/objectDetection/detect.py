import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from yoloV5Utils.datasets import LoadImages, LoadStreams
from yoloV5Utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from yoloV5Utils.plots import Annotator, colors
from yoloV5Utils.torch_utils import load_classifier, select_device, time_sync
from yoloV5Utils.augmentations import letterbox
from PIL import Image
from numpy import asarray
import time


class Detection:
    def __init__(self, box, conf, c, name):
        self.box    = box
        self.conf   = conf
        self.c      = c
        self.name   = name
        self.center = self.get_center()

    def get_centers_distance(self, obj):
        diff = (self.center[0] - obj.center[0], self.center[1] - obj.center[1])
        
        return torch.sqrt(diff[0]*diff[0] + diff[1]*diff[1])

    def get_center(self):
        c1, c2 = (self.box[0], self.box[1]), (self.box[2], self.box[3])
        x0, y0 = c1
        x1, y1 = c2
        width  = x1 - x0
        height = y1 - y0
        
        return (x0 + int(width/2), y0 + int(height/2))

class YoloV5OD:
    def __init__(self,
                 weights,
                 imgsz=[640, 640],
                 conf_thres=0.4,
                 iou_thres=0.45,
                 max_det=1000,
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 hide_labels=False,
                 half=True):
        self.device='' 
        self.imgsz=imgsz
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.half=half

        set_logging()
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA  

        # Load model
        #w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)    
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = attempt_load(weights, map_location=self.device)
        #model = torch.jit.load(self.weights) if 'torchscript' in w else attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        if self.half:
            self.model.half()  # to FP16
            
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def do_inference(self, im0s, class_filter=[]):
        objects = []    
        # Padded resize
        img = letterbox(im0s, self.imgsz, self.stride, True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)    
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=self.augment, visualize=False)[0]
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
    
        # Process predictions
        for det in pred:  # per image
            im0 = im0s
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *box, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if len(class_filter) == 0:
                        objects.append(Detection(box, conf, c, self.names[c]))
                    elif self.names[c] in class_filter:
                        objects.append(Detection(box, conf, c, self.names[c]))
        return objects
    
    def save_results(self, imgName, img, objects):
        ratio = 1
        im0 = img.copy()
        annotator = Annotator(im0, line_width=2, example=str(self.names))
        for obj in objects:
            label = f'{obj.name} {obj.conf:.2f}'
            annotator.box_label(obj.box, obj.name, color=colors(obj.c, True))
            center = obj.center
            annotator.box_label([center[0], center[1] - ratio, center[0] + ratio, center[1] + ratio], color=colors(obj.c, True))
        
        im0 = annotator.result()
        cv2.imwrite(imgName, im0)


def pairing_bodies_to_objects(objects, bodies):
    pairs = []
    for o, obj in enumerate(objects):
        min_distance, body_pair = sys.maxsize, -1
        for b, body in enumerate(bodies):
            distance = obj.get_centers_distance(body)
            if distance < min_distance:
                body_pair, min_distance = b, distance 
        if body_pair != -1:
            pairs.append([o, body_pair])
            
    return pairs


def body_crop(pairs, objects, bodies, img):
    final_bodies = []
    bodies_cropped = []
    for pair_obj, pair_body in pairs:
        if pair_body not in bodies_cropped:
            crop = save_one_box(bodies[pair_body].box, img, save=False)
            final_bodies.append([crop, objects[pair_obj].name])
            bodies_cropped.append(pair_body)
    return final_bodies
            
def save_pair_results(pairs, objects, bodies, img, saveName):
    ratio = 1
    im0 = img.copy()
    annotator = Annotator(im0, line_width=2)
    for o, obj in enumerate(objects):
        color = (255, 0, 0)
        line_body = -1
        for pair_obj, pair_body in pairs:
            if o == pair_obj:
                line_body = pair_body
                color = (0, 255, 0)
                break
        label = f'{obj.name} {obj.conf:.2f}'
        annotator.box_label(obj.box, label, color=color)
        center = obj.center
        annotator.box_label([center[0], center[1] - ratio, center[0] + ratio, center[1] + ratio], color=color)
        if line_body != -1:
            annotator.line(obj.center, bodies[line_body].center)
        
    for b, body in enumerate(bodies):
        color = (255, 0, 0)
        for pair_obj, pair_body in pairs:
            if b == pair_body:
                color = (0, 255, 0)
                break
        label = f'{body.name} {body.conf:.2f}'
        annotator.box_label(body.box, label, color=color)
        center = body.center
        annotator.box_label([center[0], center[1] - ratio, center[0] + ratio, center[1] + ratio], color=color)


    im0 = annotator.result()
    cv2.imwrite(saveName, im0)
    
