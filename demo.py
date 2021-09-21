import argparse
import os
from PIL import Image
from estimator import AnomalyDetector
import numpy as np

# function for segmentations
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

parser = argparse.ArgumentParser()
parser.add_argument('--demo_folder', type=str, default='./sample_images', help='Path to folder with images to be run.')
parser.add_argument('--save_folder', type=str, default='./results', help='Folder to where to save the results')
opts = parser.parse_args()

demo_folder = opts.demo_folder
save_folder = opts.save_folder

save_folder = 'Mbilly3_results_rarecase1'

images = [os.path.join(demo_folder, image) for image in os.listdir(demo_folder)]
detector = AnomalyDetector(True)

# Save folders
semantic_path = os.path.join(save_folder, 'semantic')
anomaly_path = os.path.join(save_folder, 'anomaly')
synthesis_path = os.path.join(save_folder, 'synthesis')
entropy_path = os.path.join(save_folder, 'entropy')
distance_path = os.path.join(save_folder, 'distance')
perceptual_diff_path = os.path.join(save_folder, 'perceptual_diff')

os.makedirs(semantic_path, exist_ok=True)
os.makedirs(anomaly_path, exist_ok=True)
os.makedirs(synthesis_path, exist_ok=True)
os.makedirs(entropy_path, exist_ok=True)
os.makedirs(distance_path, exist_ok=True)
os.makedirs(perceptual_diff_path, exist_ok=True)

from mtce_data_apis.interfaces.database_interface import db

# selection = db.get_user_selections(user='yifan')['yifan'][0]
# sequence = selection.sequences[0]
recording_id = "20210728_111927_Mbilly3.000720-001021"
sequence = db.get_sequence(recording_id)
print("sequence:", sequence)
frame = sequence[0]
print("image size:", frame.camera_SVM_front.image.shape)

# read from mbilly3 json file
import json
from calibration import Calib
json_file = "/m/das_data/00_config/20210618_AZEMB3/calib_result_MBilly3_SVM_front_AR_front_fisheye.json"
with open(json_file, 'r') as f:
    calibration = json.load(f)
K = np.array(calibration['INTRINSIC_CALIBRATION']['ORIGINAL_IMAGE']['INTRINSICS'])
D = np.array(calibration['INTRINSIC_CALIBRATION']['ORIGINAL_IMAGE']['DIST_COEFFS'])
Knew = np.array(calibration['INTRINSIC_CALIBRATION']['UNDISTORTED_IMAGE']['INTRINSICS'])

calib = Calib(calibration)
f_map_x, f_map_y = calib.img_rectification('SVM_front')

import tqdm
import cv2
for idx, img in enumerate(tqdm.tqdm(sequence)):

    image = cv2.cvtColor(img.camera_SVM_front.image, cv2.COLOR_BGR2RGB)
    # img_crop = cv2.fisheye.undistortImage(distorted = image, K = K, D = D, Knew = K)
    image_undistorted = cv2.remap(image, f_map_x, f_map_y, interpolation=cv2.INTER_LINEAR)
    y_margin = 400
    x_margin = 5
    img_crop = image_undistorted[0:-y_margin, x_margin:-x_margin].copy()
    cv2.imwrite('temp_distort.png', image)
    cv2.imwrite('temp_undistort.png', img_crop)

    basename = str(idx) + '.png'
    # print('Evaluating image %i out of %i'%(idx+1, len(images)))
    # image = Image.open(image)
    img_crop = Image.fromarray(img_crop)
    results = detector.estimator_image(img_crop)

    anomaly_map = results['anomaly_map'].convert('RGB')
    anomaly_map.save(os.path.join(anomaly_path,basename))

    semantic_map = colorize_mask(np.array(results['segmentation']))
    semantic_map.save(os.path.join(semantic_path,basename))

    synthesis = results['synthesis']
    synthesis.save(os.path.join(synthesis_path,basename))

    softmax_entropy = results['softmax_entropy'].convert('RGB')
    softmax_entropy.save(os.path.join(entropy_path,basename))

    softmax_distance = results['softmax_distance'].convert('RGB')
    softmax_distance.save(os.path.join(distance_path,basename))

    perceptual_diff = results['perceptual_diff'].convert('RGB')
    perceptual_diff.save(os.path.join(perceptual_diff_path,basename))

