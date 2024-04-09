import procbridge
import base64
import numpy as np
import cv2

 
def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str) # byte
    base64_str = str(base64_str, 'utf-8')
    return base64_str

def main():
    client = procbridge.Client('127.0.0.1', 8888)

    img = cv2.imread("C:/Users/Yuting_Yen/Downloads/img000498_2.jpg")
    print(client.request('detect', {'source': cv2_base64(img)}))

if __name__ == "__main__":
    main()
