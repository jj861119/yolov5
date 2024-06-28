import procbridge
import base64
import cv2
from time import time
import requests
import json

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str) # byte
    base64_str = str(base64_str, 'utf-8')
    return base64_str

def main():
    client = procbridge.Client('127.0.0.1', 8888)

    frame = cv2.imread('/mnt/images/img000410.jpg')
    buffer = cv2.imencode('.jpg', frame)[1].tostring()
    data_base64 = base64.b64encode(buffer).decode()

    # img = cv2.imread("C:/Users/Yuting_Yen/Downloads/img000410.jpg")
    # img_64 = cv2_base64(img)

    for _ in range(10):
        client.request('detect', {'source': data_base64})

    s_t = time()
    for _ in range(100):
        client.request('detect', {'source': data_base64})
    e_t = time()
    print(f'Yolo fps : {100 / (e_t-s_t)}')

    
    for _ in range(10):
        response = requests.post('http://localhost:8080/data/image/recognize', data=data_base64)

    s_t = time()
    for _ in range(100):
        response = requests.post('http://localhost:8080/data/image/recognize', data=data_base64)
    e_t = time()
    print(f'License plate fps : {100 / (e_t-s_t)}')

    if response.status_code == 200:
        response_content = response.content.decode('UTF-8')
        response_json = json.loads(response_content)

        for res in response_json:
            print(res) # 車牌資訊在這
    
    

if __name__ == "__main__":
    main()
