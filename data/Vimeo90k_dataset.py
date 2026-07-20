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

class Vimeo90KDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        # get train indexes
        self.txt_path = self.opt['txt_path']
        self.fps = self.opt['fps']

        with open(self.txt_path, 'r') as f:
            self.dir_list = f.readlines()
            self.dir_list = [a.strip('\n') for a in self.dir_list]

        
    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        cover_path = self.dir_list[index]
        random_num = random.randint(0, len(self.dir_list)-1)
        secret_path = self.dir_list[random_num]

        cover_img = util.read_img(None, cover_path)
        secret_img = util.read_img(None, secret_path)

        CH, CW, CC = cover_img.shape
        rnd_h = random.randint(0, max(0, CH - GT_size))
        rnd_w = random.randint(0, max(0, CW - GT_size))
        cover_img = cover_img[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
        cover_img = cover_img[:, :, [2, 1, 0]]
        cover_img = torch.from_numpy(np.ascontiguousarray(np.transpose(cover_img, (2, 0, 1)))).float() # T C H W

        SH, SW, SC = secret_img.shape
        rnd_h = random.randint(0, max(0, CH - GT_size))
        rnd_w = random.randint(0, max(0, CW - GT_size))
        secret_img = secret_img[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
        secret_img = secret_img[:, :, [2, 1, 0]]
        secret_img = torch.from_numpy(np.ascontiguousarray(np.transpose(secret_img, (2, 0, 1)))).float() # T C H W
        shift_h = SH // 4
        shift_w = SW // 4

        

        return {'Cover': cover_img, 'Secret': cover_img, 'Change':secret_img}

    def __len__(self):
        assert len(self.dir_list)
        return len(self.dir_list)