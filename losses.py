from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()

    def forward(self, y_pred, y_true):
        sum_axis = [0, 1, 2, 3]
        cls_gt, offset_gt, wh_gt = torch.split(y_true, [4, 2, 2], dim=1)
        cls_pred, offset_pred, wh_pred = y_pred['hm'], y_pred['reg'], y_pred['wh']

        mask_cls = cls_gt.eq(1).float()
        num_pos = torch.clamp_min_(torch.sum(mask_cls, dim=sum_axis), 1)
        cls_pred = torch.sigmoid(cls_pred)
        cls_pos_loss = torch.log(cls_pred) * torch.pow(1 - cls_pred, 2) * mask_cls
        cls_neg_loss = torch.log(1 - cls_pred) * torch.pow(cls_pred, 2) * torch.pow(1 - cls_gt, 4) * (1. - mask_cls)
        cls_loss = -torch.sum(cls_pos_loss + cls_neg_loss, dim=sum_axis) / num_pos

        mask_loc = wh_gt.gt(0).float()
        offset_loss = torch.sum(torch.abs(offset_gt - offset_pred) * mask_loc, dim=sum_axis) / num_pos
        size_loss = torch.sum(torch.abs(wh_gt - wh_pred) * mask_loc, dim=sum_axis) / num_pos

        total_loss = torch.mean(cls_loss + 0.1 * size_loss + 1. * offset_loss)

        return total_loss
