import cv2
import numpy as np
import os
import urllib.request

# URLs to model files
prototxt_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/ssd_mobilenet_v1_coco.pbtxt'
model_url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz'

# Filenames to save/load
prototxt_file = 'ssd_mobilenet_v1_coco.pbtxt'
model_file = 'frozen_inference_graph.pb'

# Download prototxt if not exists
if not os.path.isfile(prototxt_file):
    print("Downloading prototxt...")
    urllib.request.urlretrieve(prototxt_url, prototxt_file)

# Download and extract model if not exists
if not os.path.isfile(model_file):
    import tarfile
    import shutil

    print("Downloading model... (~25 MB) This might take a moment.")
    tmp_tar = 'ssd_mobilenet.tar.gz'
    urllib.request.urlretrieve(model_url, tmp_tar)

    print("Extracting model...")
    with tarfile.open(tmp_tar) as tar:
        tar.extract('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')
    shutil.move('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', model_file)
    os.remove(tmp_tar)
    shutil.rmtree('ssd_mobilenet_v1_coco_2017_11_17')

# Load class labels for COCO dataset (80 classes)
classNames = { 0: 'background',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }

# Load model
net = cv2.dnn.readNetFromTensorflow(model_file, prototxt_file)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare input blob and run forward pass
    blob = cv2.dnn.blobFromImage(frame, size=(300,300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            class_id = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = classNames.get(class_id, 'Unknown')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (10, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

    cv2.imshow("MobileNet SSD Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
