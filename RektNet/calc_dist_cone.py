from keypoint_tutorial_util import print_kpt_L2_distance
from keypoint_net import KeypointNet
import cv2

def calc_dist_cone(image_cone):
    KPT_KEYS = ["top", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"] # set up geometry loss keys
    
    model_filepath = "rektnet_weights/pretrained_kpt.pt"
    model = KeypointNet()
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')).get('model'))
    
    kpt_keys = KPT_KEYS
    study_name = "kpt_dist"
    evaluate_mode = True
    input_size = int(80)

    print_kpt_L2_distance(model, dataloader, kpt_keys, study_name, evaluate_mode, input_size)


if __name__ == "__main__":
    image_filepath = "test_kpt.png"
    image = cv2.imread(image_filepath)
    calc_dist_cone(image_cone)

