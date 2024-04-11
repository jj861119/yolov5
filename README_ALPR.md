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
    "result": [
        {
            "position": [
                77.18392944335938,
                137.66302490234375,
                156.8311767578125,
                314.31243896484375
            ],
            "type": "person",
            "confidence": 0.9272387027740479
        },
        {
            "position": [
                0.0,
                222.92477416992188,
                363.38458251953125,
                294.9822082519531
            ],
            "type": "motorcycle",
            "confidence": 0.9087759256362915
        },
        {
            "position": [
                412.5821838378906,
                318.1091003417969,
                57.46453857421875,
                78.4747314453125
            ],
            "type": "motorcycle",
            "confidence": 0.7929121255874634
        }
    ]
}
```


