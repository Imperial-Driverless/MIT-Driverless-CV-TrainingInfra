from keypoint_tutorial_util import print_kpt_L2_distance
from keypoint_net import KeypointNet
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ConeDataset
from utils import load_train_csv_dataset, prep_image, visualize_data, vis_tensor_and_save, calculate_distance, calculate_mean_distance

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def calc_img_coords(h, w, tensor_output):
    tensor_output = tensor_output[0].cpu().data
    coords = []
    i = 0
    for pt in np.array(tensor_output):
        coords.append((int(pt[0] * w), int(pt[1] * h)))
        i += 1    
    return coords

def calc_keypoints(image_filepath="test_kpt.png", img_size=int(80)):
    """
    This function calculates the 7 keypoints of the cone.

    The RektNet model outputs two tensors: (tensor_1, tensor_2)
    tensor_1 (shape [1, 7, 80, 80]) -> A heatmap over the image for each keypoint
    tensor_2 (shape [1, 7, 2]) -> The x and y coordinate of each keypoint
    """
    image_filepath = "dataset/RektNet_Dataset/vid_37_frame_400_3.jpg"
    model_filepath = "rektnet_weights/pretrained_kpt.pt"
    
    img_name = '_'.join(image_filepath.split('/')[-1].split('.')[0].split('_')[-5:])

    image_size = (img_size, img_size)

    image = cv2.imread(image_filepath)
    h, w, _ = image.shape

    # Turn the image so it can be processed by the KeypointNet
    image = prep_image(image=image, target_image_size=image_size)
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')

    model = KeypointNet()
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')).get('model'))
    model.eval()
    output = model(image)

    x_batch = image.to(device)
    y_hm_batch = output[0].to(device)
    y_point_batch = output[1].to(device)
    coords = calc_img_coords(h, w, y_point_batch)
    return x_batch, y_hm_batch, y_point_batch, coords

def calc_cone_width_pixels(coords):
    """
    coords: ["top", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"]
    """
    x1=coords[5][0]
    x2=coords[6][0]
    y1=coords[5][1]
    y2=coords[6][1]

    width = ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
    return width

def calc_cone_length_pixels(coords):
    """
    coords: ["top", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"]
    """
    x1=coords[5][0]
    x2=coords[6][0]
    y1=coords[5][1]
    y2=coords[6][1]

    midpoint_base = ((x1 + x2)/2, (y1 + y2)/2)
    
    midpoint_x = midpoint_base[0]
    midpoint_y = midpoint_base[1]
    top_x = coords[0][0]
    top_y = coords[0][1]

    length = ((((midpoint_x - top_x)**2) + ((midpoint_y - top_y)**2) )**0.5)
    return length

def given_real_and_pixels_get_distance(Object_length_meters,Object_width_meters,Object_length_pixels,Object_width_pixels, camera_specifications):
    Sensor_width_mms = camera_specifications["Sensor_width_mms"]
    Sensor_width_pixels = camera_specifications["Sensor_width_pixels"]
    Sensor_length_mms = camera_specifications["Sensor_length_mms"]
    Sensor_length_pixels = camera_specifications["Sensor_length_pixels"]
    Focal_length = camera_specifications["Focal_length"]

    Object_width_on_sensor_mms=(Sensor_width_mms*Object_width_pixels)/Sensor_width_pixels
    distance_width=(Focal_length*Object_width_meters)/Object_width_on_sensor_mms

    Object_length_on_sensor_mms=(Sensor_length_mms*Object_length_pixels)/Sensor_length_pixels
    distance_length=(Focal_length*Object_length_meters)/Object_length_on_sensor_mms
        
    print(f'For obtaining an Object_pixel_size={Object_length_pixels} x {Object_width_pixels}(pixels) of an object that has Object_actual_size ={Object_length_meters} x {Object_width_meters}(meters), distance must be distance_length = {distance_length}(meters) and distance_width= {distance_width}(meters) and mean_distance(suggested)={(distance_length+distance_width)/2}(meters) ')
    return distance_length, distance_width

def calc_dist_cone(image_filepath="test_kpt.png", label="N/A"):

    KPT_KEYS = ["top", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"] # set up geometry loss keys
    INPUT_SIZE = (80, 80) # dataset size
    img_size = int(80)
    
    model_filepath = "rektnet_weights/pretrained_kpt.pt"
    model = KeypointNet()
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')).get('model'))
    
    kpt_keys = KPT_KEYS
    study_name = "kpt_dist"
    evaluate_mode = True
    input_size = int(80)
    
    # Dimensions for small cones
    Object_length_meters = 0.325
    Object_width_meters = 0.228

    # Camera specifications
    camera_specifications = {
        "Sensor_length_mms": 7.410, #mm
        "Sensor_width_mms": 4.980, #mm
        "Sensor_length_pixels": 3088, #pixels
        "Sensor_width_pixels": 2076, #pixels
        "Focal_length": 9.6 #mm
    }

    x_batch, y_hm_batch, y_point_batch, coords = calc_keypoints()
    
    Object_length_pixels = calc_cone_length_pixels(coords)
    Object_width_pixels = calc_cone_width_pixels(coords)
    
    distance_length, distance_width = given_real_and_pixels_get_distance(Object_length_meters, Object_width_meters, Object_length_pixels, Object_width_pixels, camera_specifications)
    return distance_length, distance_width

if __name__ == "__main__":
    """
    # TODO: The problem is that all Perspective-n-point algorithms require the object points 
    # meaning the keypoints of the cone in real life, something which we do not have.
    # What we have done in this algorithm is use the cones actual dimensions, the size of the object in the image 
    # and the intrinsic camera parmeters. From there calculate the distance to the cone.
    """    
    # TODO: Get the true camera_specifications

    # calc_keypoints()
    calc_dist_cone()
    

