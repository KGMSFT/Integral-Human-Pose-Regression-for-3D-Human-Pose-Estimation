import sys
from config import cfg
from Human36M import Human36M
from dataset import *
import numpy as np
import cv2
from tqdm import tqdm
import scipy.io as sio
from utils.pose_utils import pixel2cam, rigid_align, process_world_coordinate, warp_coord_to_original
import random
from utils.vis import vis_keypoints, vis_3d_skeleton

joint_id = 3
root_id = 0
idx = 0
skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
def vis_3D():
    skeleton_train = np.fromfile("skeleton_train_geo_img_all_ratiox.np.bin",dtype=np.float32)
    skeleton_train = skeleton_train.reshape(-1, 18, 3)
    db = Human36M("train")
    db_hm36 = db.load_data()
    joint_num = db.joint_num
    sample_num = len(db_hm36)
    bbox = db_hm36[idx]['bbox']
    joint_img = db_hm36[idx]['joint_img']
    joint_cam = db_hm36[idx]['joint_cam']
    joint_vis = db_hm36[idx]['joint_vis']
    f = db_hm36[idx]['f']
    c = db_hm36[idx]['c']
    vis_3d_skeleton(skeleton_train[idx, :, :], joint_vis, skeleton, "test1")

if __name__ == "__main__":
    vis_3D()
