#!/usr/bin/bash
echo "Downloading Training Dataset"
gsutil cp -p gs://mit-driverless-open-source/YOLO_Dataset.zip ./dataset/
unzip dataset/YOLO_Dataset.zip -d ./dataset/
rm YOLO_Dataset.zip

echo "Downloading YOLOv3 pretrained Weights"
gsutil cp -p  gs://mit-driverless-open-source/pretrained_yolo.weights ./yolo_weights/

echo "Downloading Training and Validation Label"
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/all.csv ./dataset/
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/train.csv ./dataset/
gsutil cp -p gs://mit-driverless-open-source/yolov3-training/validate.csv ./dataset/
