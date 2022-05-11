#!/usr/bin/python3

import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import shutil
from typing import List, Tuple
import cv2

from pathlib import Path

import torch
import torch.cuda
import torch.backends

from PIL import Image, ImageDraw

import torchvision.transforms.functional
from models import Darknet
from utils.nms import nms
from utils.utils import calculate_padding

import warnings
from tqdm import tqdm

import numpy

import tempfile

warnings.filterwarnings("default")

detection_tmp_path = Path(tempfile.mkdtemp())

DetectionResults = torch.Tensor # for now, but we should make it better

class Detector:
    def __init__(self, 
        target_path,
        output_path,
        weights_path,
        model_cfg,
        conf_thres,
        nms_thres,
        xy_loss,
        wh_loss,
        no_object_loss,
        object_loss,
        vanilla_anchor
    ) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.setup_cuda()
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')

        self.device = torch.device('cpu')
        print('Using CPU')
        
        random.seed(0)
        torch.manual_seed(0)
        
        self.model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)

        self.model.load_weights(weights_path, self.model.get_start_weight_dim())

        self.model.to(self.device, non_blocking=True)
        self.model.eval()
        
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres

        self.detect(Path(target_path), Path(output_path))

    def setup_cuda(self):
        torch.manual_seed(0)
        torch.cuda.empty_cache()


    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        img = img.convert('RGB')
        w, h = img.size
        new_width, new_height = self.model.img_size()
        pad_h, pad_w, _ = calculate_padding(h, w, new_height, new_width)
        img_tensor = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=127, padding_mode="constant")
        img_tensor = torchvision.transforms.functional.resize(img_tensor, [new_height, new_width])
        img_tensor = torchvision.transforms.functional.to_tensor(img_tensor)

        if self.model.get_bw():
            raise NotImplementedError("Grayscale mode is not implemented")
            img_tensor = torchvision.transforms.functional.to_grayscale(img_tensor, num_output_channels=1)

        return img_tensor.unsqueeze(0)

    def detect_and_draw_bounding_boxes(self, img: Image.Image) -> Image.Image:
        detection_results = self.single_img_detect(img)
        img_with_bounding_boxes = self.draw_bounding_boxes(img, detection_results)
        return img_with_bounding_boxes

    def single_img_detect(self,
        img: Image.Image,
    ) -> DetectionResults:
        img_tensor = self.preprocess_image(img)
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device, non_blocking=True)
            output = self.model(img_tensor)

            detections: torch.Tensor = output[0]
            detections = detections[detections[:, 4] > self.conf_thres]
            box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
            xy: torch.Tensor = detections[:, 0:2]
            wh = detections[:, 2:4] / 2
            box_corner[:, 0:2] = xy - wh
            box_corner[:, 2:4] = xy + wh
            probabilities = detections[:, 4]
            nms_indices = nms(box_corner, probabilities, self.nms_thres)
            main_box_corner = box_corner[nms_indices]
            if nms_indices.shape[0] == 0:  
                raise Exception("I don't even know what happened")
            
            return main_box_corner

    def draw_bounding_boxes(self, original_image: Image.Image, main_box_corner: DetectionResults):
        w, h = original_image.size
        new_width, new_height = self.model.img_size()
        pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
        draw = ImageDraw.Draw(original_image)

        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
            draw.rectangle((x0, y0, x1, y1), outline="red")
        
        return original_image

    def detect(self,
        target_filepath: Path,
        output_path: Path
    ):
        mode_by_extension = {
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.tif': 'image',
            '.mov': 'video',
            '.mp4': 'video',
            '.avi': 'video',
        }
        
        extension = target_filepath.suffix.lower()
        mode = mode_by_extension[extension]
        print("Detection Mode is: " + mode)

        if mode == 'image':
            img_with_bounding_boxes = self.detect_and_draw_bounding_boxes(Image.open(target_filepath))
            img_with_bounding_boxes.show()
        elif mode == 'video':
            # raise NotImplementedError("Video detection is not implemented")
            self.video_detect(target_filepath, output_path)

    def video_detect(self, target_filepath: Path, output_path: Path):
        
        files, fps = self.split_video(target_filepath)
        #for sorting the file names properly
        files.sort(key = lambda x: int(x[5:-4]))
        
        frame_array = []
        size = (0, 0)
        for i in tqdm(files,desc='Doing Single Image Detection for every frame'):
            assert isinstance(i, str)
            frame_file=detection_tmp_path / i
            
            bounding_boxes_img_pil = self.detect_and_draw_bounding_boxes(Image.open(frame_file))
            bounding_boxes_img_opencv = numpy.array(bounding_boxes_img_pil)[:, :, ::-1].copy()
            #reading each files
            height, width, layers = bounding_boxes_img_opencv.shape
            size = (width,height)
            frame_array.append(bounding_boxes_img_opencv)

        local_output_uri = output_path / target_filepath.with_suffix(".mp4").name
        
        video_output = cv2.VideoWriter(str(local_output_uri),cv2.VideoWriter_fourcc(*'MPEG'), fps, size)

        for frame in tqdm(frame_array,desc='Creating Video'):
            video_output.write(frame)
        video_output.release()
        print(f'please check output video at {local_output_uri}')
        shutil.rmtree(detection_tmp_path)

    def split_video(self, target_filepath: Path) -> Tuple[List[str], float]:
        vidcap = cv2.VideoCapture(str(target_filepath))
        success,image = vidcap.read()
        count = 0

        while success:
            cv2.imwrite(str(detection_tmp_path) + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        vidcap.release(); 

        files = [f for f in os.listdir(detection_tmp_path) if isfile(join(detection_tmp_path, f))]
        return files, fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model-cfg', type=str, default='model_cfg/yolo_baseline.cfg')
    parser.add_argument('--target-path', type=str, help='path to target image/video')
    parser.add_argument('--output-path', type=str, default="outputs/visualization/")
    parser.add_argument('--weights-path', type=str, help='path to weights file')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.25, help='IoU threshold for non-maximum suppression')

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    ##### Loss Constants #####
    parser.add_argument('--xy-loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh-loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--no-object-loss', type=float, default=25, help='confidence loss for background')
    parser.add_argument('--object-loss', type=float, default=0.1, help='confidence loss for foreground')

    opt = parser.parse_args()

    d = Detector(target_path=opt.target_path,
         output_path=opt.output_path,
         weights_path=opt.weights_path,
         model_cfg=opt.model_cfg,
         conf_thres=opt.conf_thres,
         nms_thres=opt.nms_thres,
         xy_loss=opt.xy_loss,
         wh_loss=opt.wh_loss,
         no_object_loss=opt.no_object_loss,
         object_loss=opt.object_loss,
         vanilla_anchor=opt.vanilla_anchor)
