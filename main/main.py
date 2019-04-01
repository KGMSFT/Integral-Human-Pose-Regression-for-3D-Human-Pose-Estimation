import argparse
from config import cfg
from base import Trainer, Tester
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
import cv2
from config import cfg
from torch.nn.parallel.scatter_gather import gather
from nets.loss import soft_argmax
from utils.vis import vis_keypoints
from utils.pose_utils import flip
from meter import AverageMeter
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    trainer = Trainer(cfg)

    trainer._make_batch_generator()
    trainer._make_model()

    train_loss = AverageMeter()
    test_loss = AverageMeter()

    train_loss_his = []
    test_loss_his = []
    p1_error_his = []
    p2_error_his = []
    writer = SummaryWriter(log_dir=cfg.output_dir+'/tensorboard')
    global_step = 0
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.scheduler.step()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (index, input_img, joint_img, joint_vis, joints_have_depth) in enumerate(trainer.batch_generator):
            global_step += 1
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            input_img = input_img.cuda()
            joint_img = joint_img.cuda()
            joint_vis = joint_vis.cuda()
            joints_have_depth = joints_have_depth.cuda()

            # if itr == 101:
            #     break
          
            # forward
            heatmap_out = trainer.model(input_img)

            # backward
            JointLocationLoss = trainer.JointLocationLoss(heatmap_out, joint_img, joint_vis, joints_have_depth)

            loss = JointLocationLoss
            train_loss.update(JointLocationLoss.detach())
            # print(JointLocationLoss)
            # print(JointLocationLoss.detach().cpu().numpy())
            if JointLocationLoss.detach().cpu().numpy() == np.nan:
                print(index)
            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()
            writer.add_scalar('scalar/train_loss', train_loss.avg, global_step)
            if itr % 100 == 0:
                screen = [
                    'Epoch [%d/%d] itr [%d/%d]:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.scheduler.get_lr()[0]),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time , trainer.gpu_timer.average_time , trainer.read_timer.average_time ),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch ),
                    '%s: %.4f' % ('train_loss', train_loss.avg),
                    ]
                trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        train_loss_his.append(train_loss.avg)
        writer.add_scalar('scalar/train_loss_epoch', train_loss.avg, epoch)
        train_loss.reset()


        tester = Tester(cfg, epoch)
        tester._make_batch_generator()
        tester._make_model(trainer.model.state_dict())

        preds = []
        
        with torch.no_grad():
            for itr_test, (index,input_img, joint_img, joint_vis, joints_have_depth) in enumerate(tqdm(tester.batch_generator)):

                input_img = input_img.cuda()
                joint_img = joint_img.cuda()
                joint_vis = joint_vis.cuda()
                joints_have_depth = joints_have_depth.cuda()
                # forward
                heatmap_out = tester.model(input_img)
                test_JointLocationLoss = tester.JointLocationLoss(heatmap_out, joint_img, joint_vis, joints_have_depth)

                if cfg.num_gpus > 1:
                    heatmap_out = gather(heatmap_out,0)
                # print(heatmap_out.size())
                # test_JointLocationLoss = tester.JointLocationLoss(heatmap_out, joint_img, joint_vis, joints_have_depth)
                coord_out = soft_argmax(heatmap_out, tester.joint_num)
                test_loss.update(test_JointLocationLoss.detach())
                
                if cfg.flip_test:
                    flipped_input_img = flip(input_img, dims=3)
                    flipped_heatmap_out = tester.model(flipped_input_img)
                    if cfg.num_gpus > 1:
                        flipped_heatmap_out = gather(flipped_heatmap_out,0)
                    flipped_coord_out = soft_argmax(flipped_heatmap_out, tester.joint_num)

                    flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                    for pair in tester.flip_pairs:
                        flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1], :].clone(), flipped_coord_out[:, pair[0], :].clone()
                    coord_out = (coord_out + flipped_coord_out)/2.

                vis = False
                if vis:
                    filename = str(itr_test)
                    tmpimg = input_img[0].cpu().numpy()
                    tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)

                    tmpimg = tmpimg.astype(np.uint8)
                    tmpimg = tmpimg[::-1, :, :]
                    tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
                    tmpkps = np.zeros((3,tester.joint_num))

                    tmpkps[:2,:] = coord_out[0,:,:2].transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                    tmpkps[2,:] = 1
                    # tmpimg = vis_keypoints(tmpimg, tmpkps, tester.skeleton)
                    cv2.imwrite(osp.join(cfg.vis_dir, filename + '_output.jpg'), tmpimg)

                coord_out = coord_out.cpu().numpy()

                preds.append(coord_out)
            screen = [
                'Epoch [%d/%d]:' % (epoch, cfg.end_epoch, ),
                'lr: %g' % (trainer.scheduler.get_lr()[0]),
                '%s: %.4f' % ('test_loss', test_loss.avg),
            ]
            test_loss_his.append(test_loss.avg)
            writer.add_scalar('scalar/test_loss_epoch', test_loss.avg, epoch)

            test_loss.reset()
            trainer.logger.info(' '.join(screen))

        # evaluate
        preds = np.concatenate(preds, axis=0)
        
        p1_error, p2_error= tester._evaluate(preds, cfg.result_dir, epoch)
        writer.add_scalar('scalar/p1_error', p1_error, epoch)
        writer.add_scalar("scalar/p2_error", p2_error, epoch)    
        p1_error_his.append(p1_error)
        p2_error_his.append(p2_error)
        trainer.save_model({
            'epoch': epoch,
            'test_loss_his': test_loss_his,
            'train_loss_his': train_loss_his,
            'p1_error_his': p1_error_his,
            'p2_error_his': p2_error_his,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
        }, epoch)
    writer.close()

if __name__ == "__main__":
    main()
