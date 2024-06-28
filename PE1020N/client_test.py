import procbridge
import base64
import cv2
import numpy as np

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str) # byte
    base64_str = str(base64_str, 'utf-8')
    return base64_str

def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def main():
    client = procbridge.Client('127.0.0.1', 8888)
    img = cv2.imread("/mnt/images/img000410.jpg")
    img_64 = cv2_base64(img)
    client.request('detect', {'source': img_64})   
    

if __name__ == "__main__":
    main()
