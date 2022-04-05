#!/usr/bin/bash
echo "Downloading Training Dataset"
gsutil cp -p gs://mit-driverless-open-source/RektNet_Dataset.zip ./dataset/
unzip dataset/RektNet_Dataset.zip -d ./dataset/
rm dataset/RektNet_Dataset.zip

echo "Downloading Training and Validation Label"
cd dataset/
gsutil cp -p gs://mit-driverless-open-source/rektnet-training/mini_rektnet_label.csv ./
mv mini_rektnet_label.csv rektnet_label.csv
cd ..

echo "Downloading RektNet pretrained weights"
gsutil cp -p gs://mit-driverless-open-source/pretrained_kpt.pt ./rektnet_weights/