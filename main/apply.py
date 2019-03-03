import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from torch.nn.parallel.scatter_gather import gather
from nets.loss import soft_argmax
from utils.vis import vis_keypoints
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn

from logger import colorlogger
from nets.balanced_parallel import DataParallelModel, DataParallelCriterion

from model import get_pose_net

from imageset import ImageSet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    logger = colorlogger(cfg.log_dir, log_name='log_apply.txt')
    #pose information
    joint_num = 18
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

    #load imgs
    imgeset = ImageSet('/home/song/toyset', cfg, transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]
                                                                ))
    img_loader = DataLoader(imgeset, batch_size=1, shuffle=False, num_workers=1)


    
    # prepare network
    model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % int(args.test_epoch))
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    logger.info('Load checkpoint from {}'.format(model_path))
    logger.info("Creating graph...")
    model = get_pose_net(cfg, False, joint_num)
    model = DataParallelModel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()

    preds = []
    with torch.no_grad():
        for itr, input_sample in enumerate(tqdm(img_loader)):

            input_img = input_sample[0].cuda()
            img_name = input_sample[1][0]
            # forward
            heatmap_out = model(input_img)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out,0)
            # print(heatmap_out.size())
            # coord_out = soft_argmax(heatmap_out, tester.joint_num)
            coord_out = soft_argmax(heatmap_out, joint_num)
        

            
            vis = True
            if vis:
                
                tmpimg = input_img[0].cpu().numpy()
                tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
                tmpimg = tmpimg.astype(np.uint8)
                tmpimg = tmpimg[::-1, :, :]
                tmpimg = np.transpose(tmpimg,(1,2,0)).copy()

                tmpkps = np.zeros((3,joint_num))
                tmpkps[:2,:] = coord_out[0,:,:2].transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, img_name ), tmpimg)

            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
            
    # evaluate
    preds = np.concatenate(preds, axis=0)
    

if __name__ == "__main__":
    main()
