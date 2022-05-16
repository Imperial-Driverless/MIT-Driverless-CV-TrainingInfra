# from CVC_YOLOv3.detect import Detector
# from CVC_YOLOv3.models import Darknet
from final_YOLO import YOLO_N
from PIL import Image
import torchvision.transforms.functional
from cone_classifier import classify_cone

cone_classes = ['left', 'right', 'orange small', 'orange large']
# img_loc = "vid_107_frame_46.jpg"
img_loc = "cone_screenshot.png"
img = Image.open("dataset/YOLO_Dataset/" + img_loc)
# img = torchvision.transforms.functional.to_tensor(img)

# yolo = Detector(
#     target_path="CVC_YOLOv3/dataset/YOLO_Dataset/"+img,
#     output_path="CVC_YOLOv3/outputs/visualization/",
#     weights_path="CVC_YOLOv3/yolo_weights/pretrained_yolo.weights",
#     model_cfg="CVC_YOLOv3/model_cfg/yolo_baseline.cfg",
#     conf_thres=0.8,
#     nms_thres=0.25,
#     xy_loss=2,
#     wh_loss=1.6,
#     no_object_loss=25,
#     object_loss=0.1,
#     vanilla_anchor=False
# )


print(type(img))
print(img)
# print(img[0])


yolo = YOLO_N()
ret = yolo.detectFrame(img)
print(ret)
cropped_cones = yolo.crop_cones(img, ret)
for cone in cropped_cones:
    # cone.show()
    cone_class = classify_cone(cone)
    print(cone_classes[cone_class])
    input("press enter")
    # print(cone_class)


