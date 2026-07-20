'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os
import os.path as osp
import pickle
import random

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


def center_crop(img, size):
    """
    img: Tensor [C, H, W]
    size: int
    """
    _, H, W = img.shape
    if H < size or W < size:
        raise ValueError(f'Image too small for center crop: {H}x{W}, crop={size}')
    top = (H - size) // 2
    left = (W - size) // 2
    return img[:, top:top+size, left:left+size]


class TestDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.opt = opt

        self.txt_path = self.opt['txt_path']
        self.mask_path = self.opt['mask_path']
        self.temp_path = self.opt['temp_path']
        self.fps = self.opt['fps']

        random.seed(20)

        with open(self.txt_path, 'r') as f:
            self.dir_list = [a.strip('\n') for a in f.readlines()]

        with open(self.mask_path, 'r') as f:
            self.mask_list = [a.strip('\n') for a in f.readlines()]

        with open(self.temp_path, 'r') as f:
            self.temp_list = [a.strip('\n') for a in f.readlines()]

    def __getitem__(self, index):
        GT_size = 256

        cover_path = self.dir_list[index]
        mask_path = self.mask_list[index]
        secret_path = self.temp_list[index]

        # --- read images ---
        cover_img = util.read_img(None, cover_path)
        secret_img = util.read_img(None, secret_path)
        mask_img = util.read_img(None, mask_path)

        # --- BGR -> RGB ---
        cover_img = cover_img[:, :, [2, 1, 0]]
        secret_img = secret_img[:, :, [2, 1, 0]]

        # --- numpy -> torch [C, H, W] ---
        cover_img = torch.from_numpy(
            np.ascontiguousarray(np.transpose(cover_img, (2, 0, 1)))
        ).float()

        secret_img = torch.from_numpy(
            np.ascontiguousarray(np.transpose(secret_img, (2, 0, 1)))
        ).float()

        mask_img = torch.from_numpy(
            np.ascontiguousarray(np.transpose(mask_img, (2, 0, 1)))
        ).float()

        # --- center crop (same region) ---
        cover_img = center_crop(cover_img, GT_size)
        secret_img = center_crop(secret_img, GT_size)
        mask_img = center_crop(mask_img, GT_size)

        return {
            'Cover': cover_img,
            'Secret': cover_img,   # 원래 코드 유지
            'Change': secret_img,
            'Mask': mask_img,
            'Name': osp.splitext(osp.basename(cover_path))[0]
        }

    def __len__(self):
        assert len(self.dir_list)
        return len(self.dir_list)
