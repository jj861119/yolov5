# Server (docker setup)
## Build image
```bash
  sudo docker build --no-cache -t yolov5_aisdit_l4t -f dockerfile.yolov5 .
```

## Run container 
- Object detection server default port is 8888

```bash
  sudo docker run --name aisdit_l4t --runtime=nvidia  -v /usr/local/cuda-10.2/:/usr/local/cuda-10.2/:ro -v /usr/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu -p 8888:8888 yolov5_aisdit_l4t
```

# Client
## Sample code
```python= 
  import procbridge as pb
  client = pb.Client('127.0.0.1', 8888)
  result = client.request('detect', {'source': base64_img})
  # client.request('load_model', {'weights':'models/yolov5n6.pt'})
```

## Sample output
```json 
  {
    "0": {
        "position": [
            635.087890625,     # Left top X
            202.4039306640625, # Left top Y
            414.1806640625,    # Width
            280.60614013671875 # Height
        ],
        "type": "car",
        "confidence": 0.8545274138450623
    },
    "1": {
        "position": [
            1006.4160766601562,
            1.58306884765625,
            179.88641357421875,
            556.5445556640625
        ],
        "type": "person",
        "confidence": 0.8258681893348694
    },
    "2": {
        "position": [
            1.1123046875,
            139.1927490234375,
            188.55459594726562,
            570.056640625
        ],
        "type": "person",
        "confidence": 0.7243578433990479
    }
}
```


