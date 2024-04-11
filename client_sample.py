import procbridge
import base64
import cv2

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str) # byte
    base64_str = str(base64_str, 'utf-8')
    return base64_str

def main():
    client = procbridge.Client('127.0.0.1', 8888)
    img = cv2.imread("/mnt/images/img000410.jpg")
    img_64 = cv2_base64(img)
    result = client.request('detect', {'source': img_64})  
    print(result)

    # 未來若有需要更換模型可使用
    # client.request('load_model', {'weights':'models/yolov5n6.pt'}) 
    

if __name__ == "__main__":
    main()
