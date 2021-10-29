# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

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
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from utils.augmentations import letterbox

from PIL import Image
from numpy import asarray
import time


@torch.no_grad()



def run(im0s):
    weights='assets/yolov5s.pt'
    imgsz=[640, 640]
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    classes=None
    agnostic_nms=False
    augment=False
    line_thickness=3
    hide_labels=False
    hide_conf=False
    half=False
    dnn=False
    device=''
    
    # Initialize    
    print("T0) Starting loading models and making comprovations...")
    
    t0 = time.time()

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)

    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0        

    print("T1) Preparing images...")





    
    t1 = time.time()
    
    # Padded resize
    img = letterbox(im0s, imgsz, stride, True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    print("T2) Running inference...")
    t2 = time.time()
    # Inference
    pred = model(img, augment=augment, visualize=False)[0]
    print("T3) NMS Prediction...")
    t3 = time.time()
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t4 = time.time()
    print("T0)Load models: %0.2f(s), T1)Move Image to GPU: %0.2f(s), T3)Inference: %0.2f(s), T4)NMS: %0.2f(s)" % ((t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3)))



    
    save_crop = False
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        im0 = im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Print time (inference-only)
        # Stream results
        im0 = annotator.result()
        save_path="results.jpg"
        cv2.imwrite(save_path, im0)

    # Print results

    #print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


if __name__ == "__main__":
    source=sys.argv[1]#'data/bus.jpg'
    im0s = cv2.imread(source)  # BGR
    run(im0s)
