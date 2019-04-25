import sys
from config import cfg
from Human36M import Human36M
from dataset import *
import numpy as np
import cv2
from tqdm import tqdm
from utils.vis import vis_keypoints, vis_3d_skeleton
joint_id = 3
root_id = 0
def test_my_gt():
    skeleton_train = np.fromfile("skeleton_test_geo_img_all_ratiox.np.bin",\
                                 dtype=np.float32)
    skeleton_train = skeleton_train.reshape(-1, 18, 3)
    skeleton_struct = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

    db = Human36M("test")
    db_hm36 = db.load_data()
    sample_num = len(db_hm36)
    idx = 100
    for idx in tqdm(range(sample_num)):
        #if idx == 100:
        #    break
        data = db_hm36[idx]
        bbox = data['bbox']
        #joint_img = data['joint_img']
        joint_img = skeleton_train[idx]
        joint_vis = data['joint_vis']

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape

        # 2. get augmentation params
        do_augment = False
        if do_augment:
            scale, rot, do_flip, color_scale = get_aug_config()
        else:
            scale, rot, do_flip, color_scale = 1.0, 0, False, [1.0, 1.0, 1.0]

        # 3. crop patch from img and perform data augmentation (flip, scale, rot, color scale)
        img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

        # 4. generate patch joint ground truth
        # flip joints and apply Affine Transform on joints
        #if do_flip:
        if False:
            joint_img[:, 0] = img_width - joint_img[:, 0] - 1
            for pair in flip_pairs:
                joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :], joint_img[pair[0], :].copy()
                joint_vis[pair[0], :], joint_vis[pair[1], :] = joint_vis[pair[1], :], joint_vis[pair[0], :].copy()

        for i in range(len(joint_img)):
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            joint_img[i, 2] = joint_img[i, 2] * trans[0,0]
            # joint_img[i, 2] /= (cfg.bbox_3d_shape[0]/2. * scale) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            # joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
            joint_vis[i] *= (
                            (joint_img[i,0] >= 0) & \
                            (joint_img[i,0] < cfg.input_shape[1]) & \
                            (joint_img[i,1] >= 0) & \
                            (joint_img[i,1] < cfg.input_shape[0]) 
                            )

        # change coordinates to output space
        joint_img[:, 0] = joint_img[:, 0] / cfg.input_shape[1] * cfg.output_shape[1]
        joint_img[:, 1] = joint_img[:, 1] / cfg.input_shape[0] * cfg.output_shape[0]
        joint_img[:, 2] = joint_img[:, 2] / cfg.input_shape[0] * cfg.output_shape[0]

        if idx == 0:
            joint_img_geo = joint_img
        else:
            joint_img_geo = np.concatenate((joint_img_geo, joint_img), axis=0)
    joint_img_geo.tofile("joint_img_geo_test.bin.np")
    print(joint_img_geo.shape)
def test_proj_ratio():
    db = Human36M("test")
    db_hm36 = db.load_data()
    joint_num = db.joint_num
    sample_num = len(db_hm36)
    # skeleton_train = np.zeros([sample_num, joint_num, 3], np.float32)
    for idx in tqdm(range(sample_num)):
        # if idx == 3:
        #     break
        bbox = db_hm36[idx]['bbox']
        joint_img = db_hm36[idx]['joint_img']
        joint_cam = db_hm36[idx]['joint_cam']
        joint_vis = db_hm36[idx]['joint_vis']
        f = db_hm36[idx]['f']
        c = db_hm36[idx]['c']
        ratio_x = (joint_img[root_id, 0]- c[0]) / joint_cam[root_id, 0]
        ratio_y = (joint_img[root_id, 1]- c[1]) / joint_cam[root_id, 1]
        joint_img[:, 0] = joint_cam[:, 0] * ratio_x + c[0]
        joint_img[:, 1] = joint_cam[:, 1] * ratio_y + c[1]
        joint_img[:, 2] = joint_img[:, 2] * ratio_x
        joint_img = joint_img.astype(np.float32)
        if idx == 0:
            skeleton_train = joint_img
        else:
            skeleton_train = np.concatenate((skeleton_train, joint_img), axis=0)
    print(skeleton_train.shape)
    skeleton_train.tofile("skeleton_test_geo_img_all_ratiox.np.bin")


def gt_joint_img():
    skeleton_struct = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    db = Human36M("train")
    db_hm36 = db.load_data()
    joint_num = db.joint_num
    sample_num = len(db_hm36)
    # skeleton_train = np.zeros([sample_num, joint_num, 3], np.float32)
    for idx in tqdm(range(sample_num)):
        if idx == 100:
            break
        bbox = db_hm36[idx]['bbox']
        joint_img = db_hm36[idx]['joint_img']
        joint_cam = db_hm36[idx]['joint_cam']
        joint_vis = db_hm36[idx]['joint_vis']
        c = db_hm36[idx]['c']
        # joint_img[:, 2] = (joint_img[:, 0] - c[0])/ joint_cam[:, 0] * joint_img[:, 2 
        cvimg = cv2.imread(db_hm36[idx]['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        scale, rot, do_flip, color_scale = 1.0, 0, False, [1.0, 1.0, 1.0]
        _, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot)

        for i in range(len(joint_img)):
            x = joint_img[i, 0]
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            #ratio_x = (joint_img[i, 0] - trans[0][2]) / x
            #joint_img[i, 2] = joint_img[i, 2] * ratio_x
            joint_img[i, 2] /= (cfg.bbox_3d_shape[0]/2. * scale) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
            joint_vis[i] *= (
                            (joint_img[i,0] >= 0) & \
                            (joint_img[i,0] < cfg.input_shape[1]) & \
                            (joint_img[i,1] >= 0) & \
                            (joint_img[i,1] < cfg.input_shape[0]) & \
                            (joint_img[i,2] >= 0) & \
                            (joint_img[i,2] < 1)
                            )
        joint_img[:, 0] = joint_img[:, 0] / cfg.input_shape[1] * cfg.output_shape[1]
        joint_img[:, 1] = joint_img[:, 1] / cfg.input_shape[0] * cfg.output_shape[0]
        joint_img[:, 2] = joint_img[:, 2] * cfg.depth_dim
        # joint_img[:, 2] = joint_img[:, 2] / cfg.input_shape[1] * cfg.output_shape[1]
        joint_img = joint_img.astype(np.float32)
        vis_3d_skeleton(joint_img, joint_vis, skeleton_struct, "test_gt")
        if idx == 0:
            skeleton_train = joint_img
        else:
            skeleton_train = np.concatenate((skeleton_train, joint_img), axis=0)
    print(skeleton_train.shape)
    skeleton_train.tofile("skeleton_train_geo_100.np.bin")

def evaluate():
    idx = 0
    db = Human36M("train")
    db_hm36 = db.load_data()
    joint_num = db.joint_num
    joint_cam = db_hm36[idx]['joint_cam']
    skeleton = np.fromfile("skeleton_train.np.bin", dtype=np.float32)
    skeleton = skeleton.reshape(-1, 18, 3)
    joint_img = skeleton[idx]
    skeleton_struct = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    bone_group = (((8,11), (8, 14)), ((0,1),(0,4)), ((11,12),(12, 13),(14,15),(15,16)), ((4,5),(5,6),(1,2),(2,3)))
    bone_len = []
    for (first, second) in skeleton_struct:
        lens = {}
        len_cam = ((joint_cam[first] - joint_cam[second])**2).sum()**0.5
        len_img = ((joint_img[first] - joint_img[second])**2).sum()**0.5
        lens["len_cam"] = len_cam
        lens['len_img'] = len_img
        lens['len_ratio'] = len_cam / (len_img + 1e-10)
        bone_len.append(lens)
    print(joint_cam)
    print(joint_img)
    for i in range(len(bone_len)):
        print(bone_len[i]['len_ratio'])
    for i in range(len(bone_len)):
        print(bone_len[i]['len_img'])
    for i in range(len(bone_len)):
        print(bone_len[i]['len_cam']) 

def main():
    np.set_printoptions(precision=16)
    idx = 30000
    skeleton = np.fromfile("skeleton_train.np.bin", dtype=np.float32)
    skeleton = skeleton.reshape(-1, 18, 3)
    joint_img = skeleton[idx]
    skeleton_struct = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    bone_group = (((8,11), (8, 14)), ((0,1),(0,4)), ((11,12),(12, 13),(14,15),(15,16)), ((4,5),(5,6),(1,2),(2,3)))
    bone_len_avg_group = []
    delt = skeleton[:, 11, :] - skeleton[:, 12, :]
    # print(delt)
    # print(delt**2)
    print((delt**2).sum(axis=1).mean())

    delt2 = skeleton[:, 14, :] - skeleton[:, 15, :]
    # print(delt2)
    # print(delt2**2)
    print((delt2**2).sum(axis=1).mean())
    print()
    print("=============")
    for g in bone_group:
        group_len_avg = []
        for i in range(len(g)):

            bone_len = (((skeleton[:,g[i][0],:] - skeleton[:,g[i][1],:])**2).sum(axis=1)**0.5).mean()
            group_len_avg.append(bone_len)
            print("{:.16f}".format(bone_len))
            print()
        # print(group_len_avg)
        print()
        group_len_avg.append(group_len_avg)

if __name__ == "__main__":
    # gt_joint_img()
    test_my_gt()
    # test_proj_ratio()
