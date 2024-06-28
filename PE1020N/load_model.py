import procbridge
def main():
    client = procbridge.Client('127.0.0.1', 8888)
    print(client.request('load_model', {'weights':'models/yolov5n6.pt'}))
    

if __name__ == "__main__":
    main()
