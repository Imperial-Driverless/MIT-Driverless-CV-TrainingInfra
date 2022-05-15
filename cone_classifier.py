import numpy as np
import torch
import torchvision.transforms.functional as ttf
from typing import Union
import math
from PIL import Image

def check_stripes(roi: torch.Tensor) -> int:
    return 2

def classify_cone(img: Image.Image) -> int:
    #define color labels
    nana = ttf.to_tensor(img)
    img = img.convert('HSV')
    hue = ttf.to_tensor(img)[0] * 359
    hue[hue>349] = 3
    # print(hue)
    left = 207 #blue white stripe --> 0
    right = 60 #yellow black stripe --> 1
    orange = 15 #orange --> 2, 3 depending on small (1 stripe) or large (2 stripes)
    colors = torch.Tensor((left, right, orange))

    #get a centered vertical region of interest
    height, width = hue.shape
    roi_w_margin = math.floor(width * 0.2)
    roi_h_margin = math.ceil(height * 0.15)
    # roi_w_margin = math.floor(width * 0.1)
    # roi_h_margin = math.ceil(height * 0.25)
    roi = hue[roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]
    roi2 = nana[:, roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]

    ttf.to_pil_image(roi2).show()
    # average_color = np.mean(roi)

    #check highest similarity
    # similarity = np.square(colors - average_color)
    
    similarity = np.zeros(3)
    for i, color in enumerate(colors):
        diff = roi - color
        similarity[i] = torch.abs(torch.mean(diff))

    cone_class = np.argmin(similarity)
    if cone_class == 2:
        cone_class = check_stripes(roi)

    return cone_class
