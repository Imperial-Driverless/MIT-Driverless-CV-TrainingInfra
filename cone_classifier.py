import numpy as np
import torch
from typing import Union
import math

def classify_cone(img: Union[np.ndarray, torch.Tensor]) -> int:
    #define color labels
    left = np.array([0]) #blue black stripe --> 0
    right = np.array([0]) #yellow white stripe --> 1
    orange = np.array([0]) #orange --> 2
    colors = np.vstack((left, right, orange))

    #get a centered vertical region of interest
    height, width = img.shape
    roi_w_margin = math.floor(width * 0.1)
    roi_h_margin = math.ceil(height * 0.5)
    roi = img[roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]
    average_color = np.mean(roi)

    #check highest similarity
    similarity = np.square(colors - average_color)
    return np.argmin(similarity)
