from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def split_dataset(dir, ratio=0.8):
    train_txt = open(os.path.join(dir, 'train.txt'), 'w')
    val_txt = open(os.path.join(dir, 'val.txt'), 'w')
    for x in os.listdir(os.path.join(dir, 'image')):
        x = x.strip('.jpg') + '\n'
        if random.random() < ratio:
            train_txt.write(x)
        else:
            val_txt.write(x)
    train_txt.close()
    val_txt.close()


def draw_bboxes(image, bboxes, classnames, save_path=None):
    plt.figure()
    plt.imshow(image, aspect='auto')
    currentAxis = plt.gca()
    bboxes = np.array(bboxes, dtype=np.int)
    for box in bboxes:
        coords = (box[0], box[1]), box[2] - box[0], box[3] - box[1]
        color = '#FF0000'
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        currentAxis.text(box[0], box[1], '%s' % classnames[box[-1]], bbox={'facecolor': 'white', 'alpha': 0.})
    if save_path:
        plt.savefig(save_path)


def draw_detections(image, detections, classnames, save_path=None):
    plt.figure()
    plt.imshow(image, aspect='auto')
    currentAxis = plt.gca()
    for box in detections:
        confidence = box[-2]
        box = np.array(box, np.int32)
        coords = (box[0], box[1]), box[2] - box[0], box[3] - box[1]
        color = '#FF0000'
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        currentAxis.text(box[0], box[1], '%s, %0.2f' % (classnames[int(box[-1])], confidence),
                         bbox={'facecolor': 'white', 'alpha': 0.})
    if save_path:
        plt.savefig(save_path)
