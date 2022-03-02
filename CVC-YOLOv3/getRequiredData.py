import os

print("Downloading Training Dataset")
os.system("gsutil cp -p gs://mit-driverless-open-source/YOLO_Dataset.zip ./dataset/")
os.system("unzip dataset/YOLO_Dataset.zip -d ./dataset/ ./")
os.system("rm YOLO_Dataset.zip")

print("Downloading YOLOv3 Sample Weights")
os.system("gsutil cp -p  gs://mit-driverless-open-source/pretrained_yolo.weights")

print("Downloading Training and Validation Label")
os.system("gsutil cp -p gs://mit-driverless-open-source/yolov3-training/all.csv ./dataset/")
os.system("gsutil cp -p gs://mit-driverless-open-source/yolov3-training/train.csv ./dataset/")
os.system("gsutil cp -p gs://mit-driverless-open-source/yolov3-training/validate.csv ./dataset/")