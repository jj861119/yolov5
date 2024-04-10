# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path
import json
import torch

import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import procbridge
import base64
import numpy as np
import cv2
 
def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

class Detector():

    def __init__(self,
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            dnn=False,  # use OpenCV DNN for ONNX inference
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            data=ROOT / 'data/ALPR.yaml',  # dataset.yaml path
            half=False,  # use FP16 half-precision inference
    ):
        # Load model
        device = select_device(device)
        self.weights = weights
        self.model = Detector.load_model(weights, device, dnn, data, half)

    def update_model(self, weights=['./yolov5s.pt'], key=None, iv=None):
        self.weights = Path(weights[0])
        if key and iv:
            self.model = Detector.load_model(weights, key=key.encode('utf-8'), iv=str.encode(iv))
        else:
            self.model = Detector.load_model(weights)
        return str(self.weights)

    @staticmethod
    @smart_inference_mode()
    def load_model(weights, device='', dnn=False, data='data/ALPR.yaml', fp16=False, key=None, iv=None):
        device = select_device(device)
        if isinstance(data, str):
            data = Path(data)
        print(type(weights), weights)
        if key and iv:
            return DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data, fp16=fp16, key=key, iv=iv)
        else:
            return DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data, fp16=fp16)

    @smart_inference_mode()
    def run(self,
            source,  # base64 str
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.4,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            vid_stride=1,  # video frame-rate stride
            weights=None,
            model=None,
    ):
        weights = weights if weights else self.weights
        model = model if model else self.model

        # Load model
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Load image
        bs = 1  # batch_size
        im0 = base64_cv2(source)
        im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        res_dict = dict()

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            det = det.cpu().numpy()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)
            for id, item in enumerate(det):
                res = dict()
                pos = [item[0], item[1], item[2]-item[0], item[3]-item[1]]
                pos = list(map(float, pos))
                res['position'] = pos
                res['type'] = names[int(item[5])]
                res['confidence'] = float(item[4])
                LOGGER.info(str(res))
                res_dict[id] = res
            LOGGER.info(f'---------------------{im0.shape}')

        return json.dumps(res_dict)


def start_server(port, server_args):
    global detector
    detector = Detector(**server_args)
    s = procbridge.Server('0.0.0.0', port, delegate)
    s.start(daemon=False)
    print(f'Server is on {port}...')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888, help='ports corresponding to servers')
    parser.add_argument('--data', type=str, default=ROOT / 'data/ALPR.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def delegate(method, args):
    # define remote methods:
    if method == 'echo':
        return args
    elif method == 'load_model':
        if detector:
            return detector.update_model(**args)
        else:
            raise RuntimeError("an server error: detector not found")
    elif method == 'detect':
        if detector:
            return detector.run(**args)
        else:
            raise RuntimeError("an server error: detectors not found")
    else:
        raise RuntimeError("an server error")


def main():
    args = vars(parse_opt())
    port = args['port']
    del args['port']
    start_server(port, args)

if __name__ == "__main__":
    main()
