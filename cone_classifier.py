import numpy as np
import torch
import torchvision.transforms.functional as ttf
from typing import Union, Tuple
import math
from PIL import Image


class Classifier():

    def __init__(self) -> None:
        left = 207 #blue white stripe --> 0
        right = 60 #yellow black stripe --> 1
        orange = 25 #orange --> 2, 3 depending on small (1 stripe) or large (2 stripes)
        self.colors = torch.Tensor((left, right, orange))

    def getHSV(self, img: Image.Image, mode='bins') -> Tuple[torch.Tensor, ...]:
        img = img.convert('HSV')
        img = ttf.to_tensor(img)
        hue = img[0] * 359
        sat = img[1] * 100
        hue[hue>349] = 0

        if mode == 'bins':
            hue[sat<35] = 365 #meaningless value to discard them
        elif mode == 'mean':
            hue[sat<35] = torch.mean(hue)

        return (hue, sat)

    def check_stripes(self, sat: torch.Tensor) -> int:
        def check_neighbours(row, col, weight):
            i = -1
            delta = 2
            print("checking stripe")
            while weight < 0.1:
                neighbour = sat[row+i, col]
                # print("The neighbour value is", neighbour.item())
                if neighbour > 0:
                    weight += neighbour.item()
                    i += delta
                    # print(i)
                    delta = -(delta + 1)
                else:
                    return False

            return weight >= 0.03

        sat = sat < 30
        ttf.to_pil_image(sat*255).show()
        px_weight = 1/sat.shape[1]
        sat = sat.type(torch.DoubleTensor) * px_weight
        # print("px w", px_weight)
        stripe = False
        mid_col = math.ceil(sat.shape[1]/2)
        n_stripes = 0
        buffer = 0
        for row_i, px in enumerate(sat[:, mid_col]):
            # print(px, "\t", px>0, "\t", stripe)
            if px > 0 and stripe is False:
                # print("schumaker")
                stripe = check_neighbours(row_i, mid_col, px_weight)
                buffer = 0
            elif px == 0 and stripe is True:
                buffer += px_weight
                if buffer > 0.1:
                    # print(px)
                    print("stripe found")
                    n_stripes += 1
                    stripe = False

        print(n_stripes)

        return 1 + np.minimum(np.maximum(1, n_stripes), 2) #cap number of stripes between 1 and 2

    def check_color(self, roi: torch.Tensor) -> int:
        bin_edges = torch.Tensor([-1, 35, 73, 150, 260, 335, 359])
        bins = []
        for edge in range(bin_edges[:-1].shape[0]):
            bins.append(torch.histc(roi, 1, min=bin_edges[edge]+1, max=bin_edges[edge+1]))
        bins = np.array([bins[3], bins[1], bins[0]+bins[5]])
        print(bins)
        return np.argmax(bins)

    def classify_cone_mean(self, img: Image.Image) -> int:
        #define color labels
        nana = ttf.to_tensor(img)
        hue, sat = self.getHSV(img, 'mean')
        # print(hue)

        #get a centered vertical region of interest
        height, width = hue.shape
        roi_w_margin = math.floor(width * 0.2)
        roi_h_margin = math.ceil(height * 0.15)
        # roi_w_margin = math.floor(width * 0.1)
        # roi_h_margin = math.ceil(height * 0.25)
        roi = hue[roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]
        roi2 = nana[:, roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]
        # sat = sat[roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]

        # ttf.to_pil_image(roi2).show()
        # average_color = np.mean(roi)

        #check highest similarity
        # similarity = np.square(colors - average_color)
        
        similarity = np.zeros(3)
        for i, color in enumerate(self.colors):
            diff = roi - color
            similarity[i] = torch.abs(torch.mean(diff))

        cone_class = np.argmin(similarity)
        if cone_class == 2:
            cone_class = self.check_stripes(sat)

        return cone_class

    def classify_cone_bins(self, img: Image.Image) -> int:
        nana = ttf.to_tensor(img)
        hue, sat = self.getHSV(img)

        #get a centered vertical region of interest
        height, width = hue.shape
        roi_w_margin = math.floor(width * 0.2)
        roi_h_margin = math.ceil(height * 0.15)
        roi = hue[roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]
        roi2 = nana[:, roi_h_margin:height-roi_h_margin, roi_w_margin:width-roi_w_margin]

        # ttf.to_pil_image(roi2).show()

        cone_class = self.check_color(roi)
        if cone_class == 2:
            cone_class = self.check_stripes(sat)

        return cone_class
