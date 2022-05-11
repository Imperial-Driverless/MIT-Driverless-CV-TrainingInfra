### Description

The repo is originally forked from https://github.com/ultralytics/yolov3 and contains inference and training code for YOLOv3 in PyTorch.
Moreover a more detailed readme can be found as README_MIT.md however this is a summary that leverages the use of a bash script to make setting up easier.

## Requirements:

* CUDA>=10.1
* python==3.6
* numpy==1.16.4
* matplotlib==3.1.0
* torchvision==0.3.0
* opencv_python==4.1.0.25
* torch==1.1.0
* requests==2.20.0
* pandas==0.24.2
* imgaug==0.3.0
* onnx==1.6.0
* optuna==0.19.0
* Pillow==6.2.1
* protobuf==3.11.0
* pymysql==0.9.3
* retrying==1.3.3
* tensorboardX==1.9
* tqdm==4.39.0

## Usage
### 1.Download our dataset

From the CVC-YOLOv3 folder run the bash script get-required-data.sh which will save the dataset along with the different partitions of the csv files all in the /dataset folder and save the pretrained yolo weights in the yolo_weights/ directory

#### 1.2 Environment Setup (Optional)

```
sudo python3 setup.py build develop
```

### 2.Inference

#### To download our pretrained YOLO weights for *Formula Student Standard*, click ***[here](https://storage.googleapis.com/mit-driverless-open-source/pretrained_yolo.weights)***

```
python3 detect.py --model_cfg=<path to cfg file> --target_path=<path to an image or video> --weights_path=<path to your trained weights file>
```

Once you've finished inference, you can access the result in `./outputs/visualization/`

#### Run Bayesian hyperparameter search

Before running the Bayesian hyperparameter search, make sure you know what specific hyperparameter that you wish to tuning on, and a reasonable operating range/options of that hyperparameter.

Go into the `objective()` function of `train_hyper.py` edit your custom search

Then launch your Bayesian hyperparameter search
```
python3 train_hyper.py --model_cfg=<path to cfg file> --study_name=<give it a proper name>
```

#### Convert .weights to .onnx manually

Though our training scrip will do automatical .weights->.onnx conversion, you can always do it manually
```
python3 yolo2onnx.py --cfg_name=<path to your cfg file> --weights_name=<path to your .weights file>
```

#### Splits your own csv file 

```
python3 generate_kmeans_dataset_csvs.py --input_csvs=<path to your csv file that contains all the label> --dataset_path=<path to your image dataset>
```

