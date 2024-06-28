import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from detect_server_alpr import Detector
from utils.general import print_args


class ImageHandler(BaseHTTPRequestHandler):
    def __init__(self, detector_config, *args):
        if detector_config is None:
            return
        with open(detector_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.detector = Detector(**config)
        BaseHTTPRequestHandler.__init__(self, *args)
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        print(self.headers)
        post_data = self.rfile.read(content_length)

        if self.path == '/data/image/recognize':
            res = self.detector(source=post_data)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(res)
        elif self.path == '/model/update':
            try:
                config = json.load(post_data)
                Detector.update_model(**config)
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write('update success')
            except Exception as e:
                self.send_error(400, f'Bad Request: update model config failed({e})')
        else:
            self.send_error(400, 'Bad Request: POST data is not an image or json format model config')


def start_server(address='127.0.0.1',
                 port=8888,
                 handler_class=ImageHandler):
    server_address = (str(address), port)
    httpd = HTTPServer(server_address, handler_class)
    print(f"Starting Echo Server on port {port}...")
    httpd.serve_forever()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='127.0.0.1', help='server ip')
    parser.add_argument('--port', type=int, default=8888, help='ports to server')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main():
    args = vars(parse_opt())
    port = args['port']
    del args['port']
    start_server(port, args)

if __name__ == "__main__":
    main()
