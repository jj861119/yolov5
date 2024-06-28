import base64
import http.client
import json

def send_image(image_base64, host='localhost', port=8080):
    conn = http.client.HTTPConnection(host, port)
    headers = {'Content-type': 'application/img-stream',
               'Content-length': len(image_base64)}
    
    conn.request('POST', '/data/image/recognize', image_base64, headers)
    
    response = conn.getresponse()
    print('Response from server:')
    print(response.status, response.reason)
    res = response.read()
    conn.close()
    
    return res

def update_model(config, host='localhost', port=8888):
    conn = http.client.HTTPConnection(host, port)
    send_str = json.dumps(config)
    headers = {'Content-type': 'application/json',
               'Content-length': len(send_str)}

    conn.request('POST', '/model/update', send_str, headers)

    response = conn.getresponse()
    print('Response from server:')
    print(response.status, response.reason)
    res = response.read()
    conn.close()
    
    return res

if __name__ == '__main__':
    img_file = './data/images/bus.jpg'
    with open(img_file, 'rb') as f:
        image_base64 = base64.b64encode(f.read())  # .decode()
    try:
        result = send_image(image_base64)
        print(result)
    except Exception as e:
        print(e)

    try:
        config = {'device': '',
                'data': 'data/ALPR.yaml'}
        result = update_model(config)
        print(result)
    except Exception as e:
        print(e)
