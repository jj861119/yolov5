import openvino as ov
import numpy as np
import cv2
import time


def format_yolov5(frame):
    # padding black for yolov5 input requirement
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def wrap_prediction(input_image,
                    output_data,
                    input_width=640,
                    input_height=640,
                    conf_thres=0.45,
                    iou_thres=0.45,
                    classes=None,
                    class_id_map=None
    ):
    class_ids = []
    confidences = []
    boxes = []
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / input_width
    y_factor = image_height / input_height

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes and class_id not in classes:
                continue
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                if class_id_map:
                    class_ids.append(class_id_map[class_id])                
                else:
                    class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = (x - 0.5 * w) * x_factor
                top = (y - 0.5 * h) * y_factor
                width = w * x_factor
                height = h * y_factor
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def detect(
    model,
    image_path
):
    output_blob = model.output(0)
    img = cv2.imread(image_path)
    preprocessed_img = format_yolov5(img)
    input_image = cv2.dnn.blobFromImage(preprocessed_img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    predictions = model(input_image)[output_blob]
    class_id, confidence, bbox_ltwh = wrap_prediction(preprocessed_img, \
                                                        predictions[0]
                                                        )
    return bbox_ltwh, img, preprocessed_img.shape


def UI_box(x, img, color=(255, 255, 0), label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        img = cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
    return img


core = ov.Core()
model = core.read_model("runs/train/coco2014_3class_v5s_0205_openvino_model/coco2014_3class_v5s_0205.xml")
compiled_model = core.compile_model(model, 'AUTO')

for i in range(10):
    boxes, image, input_shape = detect(compiled_model, 
                                    image_path='E:/datasets/MAAS_smart_pole/test_set/images/img000536.jpg')

s_time = time.time()
for i in range(100):
    boxes, image, input_shape = detect(compiled_model, 
                                    image_path='E:/datasets/MAAS_smart_pole/test_set/images/img000536.jpg')
print(f'FPS : {100.0 / (time.time() - s_time)}')

NAMES = ['person', 'car', 'motorbike']
COLORS = {name: [np.random.randint(0, 255) for _ in range(3)] for i, name in enumerate(NAMES)}

# visualize results
image_with_boxes = image.copy()
for index, box in enumerate(boxes):
    image_with_boxes = cv2.rectangle(image_with_boxes, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (255, 255, 0), 3)
    image_with_boxes = UI_box(x=box, img=image_with_boxes, color=(0, 0, 255), label='', line_thickness=2)
cv2.imshow('image_with_boxes', image_with_boxes)
cv2.waitKey(0)