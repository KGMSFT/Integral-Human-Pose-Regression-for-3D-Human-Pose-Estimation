import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import DatasetLoader
from timer import Timer
from logger import colorlogger
from nets.balanced_parallel import DataParallelModel, DataParallelCriterion
from model import get_pose_net, get_se_pose_net
from nets import loss
from grammer_model import get_grammer_net
# dynamic dataset import
for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.log'):
        
        self.cfg = cfg
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, scheduler):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        state_dict_ckpt = ckpt['network']
        state_dict_model = model.state_dict()
        state_dict_model.update(state_dict_ckpt)

        # print(optimizer.state_dict())
        # print(ckpt['optimizer'])
        # model.load_state_dict(ckpt['network'])
        model.load_state_dict(state_dict_model)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
     
        return start_epoch, model, optimizer, scheduler


class Trainer(Base):
    
    def __init__(self, cfg):
        self.GrammerLoss = DataParallelCriterion(loss.GrammerLoss())
        self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
        super(Trainer, self).__init__(cfg, log_name = 'train.log')

    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd) 
        else:
            print("Error! Unknown optimizer name: ", optimizer_name)
            assert 0

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_dec_epoch, gamma=self.cfg.lr_dec_factor)
        return optimizer, scheduler
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("=== {} | sample_ratio:{} ====".format(cfg.exp_name, cfg.sample_ratio))
        self.logger.info("Creating dataset...")
        trainset_list = []
        for i in range(len(self.cfg.trainset)):
            trainset_list.append(eval(self.cfg.trainset[i])("train"))
        trainset_loader = DatasetLoader(trainset_list, True, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        ))
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.num_gpus*self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.joint_num = trainset_loader.joint_num[0]
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = batch_generator
    
    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_grammer_net(self.cfg, True, self.joint_num)
        model = DataParallelModel(model).cuda()
        optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer, scheduler = self.load_model(model, optimizer, scheduler)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

class Tester(Base):
    
    def __init__(self, cfg, test_epoch, log_name='test.log'):
        self.GrammerLoss = DataParallelCriterion(loss.GrammerLoss())
        self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
        self.coord_out = loss.soft_argmax
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(cfg, log_name = 'test.log')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(self.cfg.testset)("test")
        testset_loader = DatasetLoader(testset, False, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        ))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus*self.cfg.test_batch_size, shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.tot_sample_num = testset_loader.__len__()
        self.batch_generator = batch_generator
    
    def _make_model(self, ckpt=None):
            
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        
        if ckpt is None:
            model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
            assert os.path.exists(model_path), 'Cannot find model at ' + model_path
            self.logger.info('Load checkpoint from {}'.format(model_path))
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['network'])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        self.model = model
        # self.model = DataParallelModel(model).cuda()

    def _evaluate(self, preds, result_save_path, epoch):
        p1_error, p2_error, p1_eval_summary, p2_eval_summary, p1_action_eval_summary, p2_action_eval_summary,\
            p1_joint_eval_summary, p2_joint_eval_summary, p1_dim_eval_summary, p2_dim_eval_summary = self.testset.evaluate(preds, result_save_path)
        self.logger.info("=== {} | evaluate epoch: {}, sample_ratio: {}, geo_reg: {} ===\n".format(cfg.exp_name, epoch, cfg.sample_ratio, cfg.geo_reg))
        self.logger.info(p1_eval_summary)
        self.logger.info(p2_eval_summary)
        self.logger.info(p1_action_eval_summary)
        self.logger.info(p2_action_eval_summary)
        self.logger.info(p1_joint_eval_summary)
        self.logger.info(p2_joint_eval_summary)
        self.logger.info(p1_dim_eval_summary)
        self.logger.info(p2_dim_eval_summary)
        return p1_error, p2_error
