from CVC_YOLOv3.detect import Detector
from CVC_YOLOv3.models import Darknet

img = "vid_107_frame_46.jpg"

yolo = Detector(
    target_path="CVC_YOLOv3/dataset/YOLO_Dataset/"+img,
    output_path="CVC_YOLOv3/outputs/visualization/",
    weights_path="CVC_YOLOv3/yolo_weights/pretrained_yolo.weights",
    model_cfg="CVC_YOLOv3/model_cfg/yolo_baseline.cfg",
    conf_thres=0.8,
    nms_thres=0.25,
    xy_loss=2,
    wh_loss=1.6,
    no_object_loss=25,
    object_loss=0.1,
    vanilla_anchor=False
)


