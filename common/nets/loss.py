import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.pose_utils import pixel2cam, warp_coord_to_original
from config import cfg
from torch.autograd import Function
def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

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

class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()

    def forward(self, heatmap_out, gt_coord, gt_vis, gt_have_depth):
        
        joint_num = gt_coord.shape[1]
        coord_out = soft_argmax(heatmap_out, joint_num)

        _assert_no_grad(gt_coord)
        _assert_no_grad(gt_vis)
        _assert_no_grad(gt_have_depth)

        loss = torch.abs(coord_out - gt_coord) * gt_vis
        loss = (loss[:,:,0] + loss[:,:,1] + loss[:,:,2] * gt_have_depth)/3.

        loss = loss.mean()

        # print(gt_have_depth)
#        varloss = VarLoss(var_weight=0.1)(coord_out[:,:,2], gt_vis, gt_have_depth, gt_coord[:,:,:2])
#        loss += varloss[0]
        return loss

class VarLoss(Function):
  def __init__(self, var_weight):
    super(VarLoss, self).__init__()
    self.var_weight = var_weight
    self.device = torch.device('cuda:0')
    # self.skeleton_idx = [[[0,1],    [1,2],
    #                       [3,4],    [4,5]],
    #                      [[10,11],  [11,12],
    #                       [13,14],  [14,15]], 
    #                      [[2, 6], [3, 6]], 
    #                      [[12,8], [13,8]]]
    # self.skeleton_weight = [[1.0085885098415446, 1, 
    #                          1, 1.0085885098415446], 
    #                         [1.1375361376887123, 1, 
    #                          1, 1.1375361376887123], 
    #                         [1, 1], 
    #                         [1, 1]]

    self.skeleton_idx = (((8,11), (8, 14)), ((0,1),(0,4)), ((11,12),(12, 13),(14,15),(15,16)), ((4,5),(5,6),(1,2),(2,3)))
    self.skeleton_weight =[[4.8796018741361724,
                        4.8605417893065894],

                        [4.3234125917611230,
                        4.3159616923218680],

                        [9.2031303885011813,
                        8.0549408718332920],

                        [9.1672605379952188,
                        8.0938428483151714],

                        [14.3549282423755091,
                        13.9456724830988108,
                        14.3900068152373759,
                        13.9790465363496690]]

    
  def forward(self, input, visible, mask, gt_2d):
    """
    input: 预测深度
    visible: 真值深度
    mask: 关节深度是否标注
    gt_2d: 2D 坐标真值
    """
    xy = gt_2d.view(gt_2d.size(0), -1, 2)
    batch_size = input.size(0)
    output = torch.FloatTensor(1) * 0
    for t in range(batch_size):
      if mask[t].sum() == 0: # mask is the mask for supervised depth
      # if True:
        # xy[t] = 2.0 * xy[t] / ref.outputRes - 1
        for g in range(len(self.skeleton_idx)):
          E, num = 0, 0
          N = len(self.skeleton_idx[g])
          l = np.zeros(N)
          for j in range(N):
            id1, id2 = self.skeleton_idx[g][j]
            if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                      (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] / self.skeleton_weight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          loss = 0
          for j in range(N):
            if l[j] > 0:
              loss += (l[j]  - E) ** 2 / 2. / num
          output += loss 
    output = self.var_weight * output / batch_size
    self.save_for_backward(input, visible, mask, gt_2d)
    output = output.cuda(self.device, non_blocking=True)
    return output
    
  def backward(self, grad_output):
    input, visible, mask, gt_2d = self.saved_tensors
    xy = gt_2d.view(gt_2d.size(0), -1, 2)
    grad_input = torch.zeros(input.size())
    batch_size = input.size(0)
    for t in range(batch_size):
      if mask[t].sum() == 0: # mask is the mask for supervised depth
        for g in range(len(self.skeleton_idx)):
          E, num = 0, 0
          N = len(self.skeleton_idx[g])
          l = np.zeros(N, dtype=np.float32)
          for j in range(N):
            id1, id2 = self.skeleton_idx[g][j]
            if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                      (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] / self.skeleton_weight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          for j in range(N):            
            if l[j] > 0:
              id1, id2 = self.skeleton_idx[g][j]
              # E = E.astype(np.float32)
              res = self.var_weight / self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id1] - input[t, id2]) / batch_size
              #grad_input[t][id1] += self.var_weight / self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) / l[j] #* (input[t, id1] - input[t, id2]) #/ batch_size
              grad_input[t][id1] += res.float()
              res2 = self.var_weight / self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id2] - input[t, id1]) / batch_size
              grad_input[t][id2] += res.float()
    grad_input = grad_input.cuda(self.device, non_blocking=True)
    return grad_input, None, None, None
