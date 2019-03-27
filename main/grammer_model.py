import torch.nn as nn
from nets.resnet import ResNetBackbone
from nets.se_resnet import se_resnet50
from config import cfg
from torch.nn import functional as F
import torch
from nets.grammer import Grammer
class HeadNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 256

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class GrammerPoseNet(nn.Module):
    def __init__(self, backbone, head, grammer):
        super(GrammerPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.grammer = grammer
    def forward(self, x):
        x = self.backbone(x)
        x1 = self.head(x)
        x2 = x1.detach()
        x2 = soft_argmax(x2, 18)
        x2 = x2.view(-1, 54)
        x2 = self.grammer(x2)
        return x1, x2 

def get_grammer_net(cfg, is_train, joint_num):
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()
    grammer = Grammer()
    model = GrammerPoseNet(backbone, head_net, grammer)
    return model

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def get_pose_net(cfg, is_train, joint_num):
    
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet(backbone, head_net)
    return model

def get_se_pose_net(cfg, is_train, joint_num):
    
    backbone = se_resnet50(is_train)
    head_net = HeadNet(joint_num)
    if is_train:
        head_net.init_weights()
    model = ResPoseNet(backbone, head_net)
    return model

def soft_argmax(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[1]+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[0]+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1,cfg.depth_dim+1).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out