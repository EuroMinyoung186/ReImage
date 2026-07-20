import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_util import initialize_weights_xavier
from torch.nn import init
import cv2
from basicsr.archs.arch_util import flow_warp
from models.archs.catanet_arch import CATANet
from models.classifier.Mask_Generation import UNet
from models.modules.Subnet_constructor import subnet
from models.modules.shuffle import InvertGaussianEncoder, InvertGaussianDecoder, EncoderViT, DecoderViT
from models.modules.common import DWT, IWT 
from models.modules.extractor import build_extractor
from models.modules.vit import ImageEncoderViT
from models.modules.pixel_decoder import PixelDecoder
import numpy as np
from torch.utils.checkpoint import checkpoint
from models.classifier.DW_Encoder import Encoder2D


def thops_mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64, model='MIMO-VRN'):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
        # but this is how we trained the model in the first place and what we reported in the paper
        if model == 'LSTM-VRN':
            self.relu = nn.ReLU(inplace=True)
        elif model == 'MIMO-VRN':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out
    
class ShuffleInvBlock(nn.Module):
    def __init__(self, encoder_config, decoer_config, subnet_constructor, channel_num_ho, channel_num_hi, clamp=1.):
        super(ShuffleInvBlock, self).__init__()
        self.clamp = clamp

        self.F = build_extractor(encoder_config, pixel_decoder_config=decoer_config)
        self.G = subnet_constructor(channel_num_ho, channel_num_hi)
        self.H = build_extractor(encoder_config, pixel_decoder_config=decoer_config)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))
    
    def forward(self, x1, x2, rev=False):
        if not rev:
            t2 = self.F(x2)
            y1 = x1 + t2
            s1, t1 = self.H(y1), self.G(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.H(x1), self.G(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.F(y2)
            y1 = (x1 - t2)

        return y1, y2  # torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]



class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, subnet_constructor_v2, channel_num_ho, channel_num_hi, clamp=1.):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_num_ho  # channel_split_num
        self.split_len2 = channel_num_hi  # channel_num - channel_split_num
        self.clamp = clamp

        self.F = subnet_constructor_v2(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        if not rev:
            t2 = self.F(x2)
            y1 = x1 + t2
            s1, t1 = self.H(y1), self.G(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.H(x1), self.G(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.F(y2)
            y1 = (x1 - t2)

        return y1, y2  # torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]
    
class ShuffleInvNN(nn.Module):
    def __init__(self, opt, channel_in_ho=3, channel_in_hi=3, subnet_constructor=None):
        super(ShuffleInvNN, self).__init__()
        operations = []
#         current_channel = channel_in
        current_channel_ho = channel_in_ho
        current_channel_hi = channel_in_hi

        self.encoder_fig = opt['encoder']
        self.pixel_decoder_fig = opt['pixel_decoder']

        for i in range(opt['depth']):
            print(i)
            b = ShuffleInvBlock(encoder_config=self.encoder_fig, decoer_config=self.pixel_decoder_fig, subnet_constructor=subnet_constructor, channel_num_ho=current_channel_ho, channel_num_hi=current_channel_hi)
            operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, x_h, rev=False, cal_jacobian=False):
        # 		out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)
        else:
            for op in reversed(self.operations):
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)

        if cal_jacobian:
            return x, x_h, jacobian
        else:
            return x, x_h



class InvNN(nn.Module):
    def __init__(self, channel_in_ho=3, channel_in_hi=3, subnet_constructor=None, subnet_constructor_v2=None, block_num=[], down_num=2):
        super(InvNN, self).__init__()
        operations = []
#         current_channel = channel_in
        current_channel_ho = channel_in_ho
        current_channel_hi = channel_in_hi

        for i in range(down_num):
            for j in range(block_num[i]):
                b = InvBlock(subnet_constructor, subnet_constructor_v2, current_channel_ho, current_channel_hi)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, x_h, rev=False, cal_jacobian=False):
        # 		out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)
        else:
            for op in reversed(self.operations):
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)

        if cal_jacobian:
            return x, x_h, jacobian
        else:
            return x, x_h

class PredictiveModuleMIMO(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=8):
        super(PredictiveModuleMIMO, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        residual_block = []
        for i in range(block_num_rbm):
            residual_block.append(ResidualBlockNoBN(nf))
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        x = self.conv_in(x)
        res = self.residual_block(x)

        return res

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def gauss_noise_mul(shape):
    noise = torch.randn(shape).cuda()

    return noise



class VSN(nn.Module):
    def __init__(self, opt, subnet_constructor=None, subnet_constructor_v2=None, down_num=2):
        # down_num이 뭔지는 모름
        # in_nc : 
        super(VSN, self).__init__()
        self.model = opt['model']
        opt_net = opt['network_G']
        print(opt)
        gen_net = opt['sam_small']
        self.gop = opt['gop']
        self.channel_in = opt_net['in_hi_nc'] # 얘네 둘의 역할은 뭘까
        self.channel_in_hi = opt_net['in_hi_nc'] #in_hi_nc는 C
        self.channel_in_ho = opt_net['in_ho_nc'] #in_ho_nc는 2
        self.encoder_in1 = opt_net['encoder_in1'] # encoder_in1은 2
        self.encoder_in2 = opt_net['encoder_in2'] # encoder_in2는 3

        self.block_num = opt_net['block_num']
        self.block_num_rbm = opt_net['block_num_rbm']
        self.nf = self.channel_in_hi  
        # subnet_constructor_v2를 보자 / down_num은 잘 모르겠네 gropus라는 걸 없애버려도 될듯
        self.irn = InvNN(self.channel_in_ho, self.channel_in_hi, subnet_constructor, subnet_constructor_v2, self.block_num, down_num)
        self.pm = PredictiveModuleMIMO(self.channel_in_ho, self.nf, block_num_rbm=self.block_num_rbm)
        self.shuffleirn = ShuffleInvNN(gen_net, self.encoder_in1, self.encoder_in2, subnet_constructor)
        self.catanet = CATANet(upscale=1)
        self.mask_prediction = Encoder2D()
        self.permutation = None
        
        self.iwt = IWT()
        self.dwt = DWT()

        #self.make_no_require_grad()

    def print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}')

    

    def shuffle_patches_random(self, img, patch_size=4):
        """
        랜덤하게 패치를 셔플 (Unshuffle을 위해 permutation 반환)

        Args:
            img (torch.Tensor): (B, C, H, W)
            patch_size (int): 패치 크기

        Returns:
            shuffled_img (torch.Tensor): 셔플된 이미지
            permutation (torch.Tensor): 셔플 순서 (unshuffle에 사용)
        """
        B, C, H, W = img.shape
        assert H % patch_size == 0 and W % patch_size == 0

        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches, patch_size, patch_size)

        # 각 배치마다 다른 랜덤 permutation을 사용하려면 for loop 필요
        permutation = torch.stack([torch.randperm(num_patches) for _ in range(B)], dim=0).to(img.device)  # (B, num_patches)

        # Apply different permutations to each image in the batch
        shuffled_patches = torch.stack([patches[i, :, perm] for i, perm in enumerate(permutation)], dim=0)

        shuffled = shuffled_patches.view(B, C, h_patches, w_patches, patch_size, patch_size)
        shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
        shuffled_img = shuffled.view(B, C, H, W)

        return shuffled_img, permutation


    def unshuffle_patches_random(self, shuffled_img, permutation, patch_size=4):
        """
        랜덤 셔플을 원래대로 되돌림

        Args:
            shuffled_img (torch.Tensor): 셔플된 이미지 (B, C, H, W)
            permutation (torch.Tensor): 셔플 시 사용한 순서 (B, num_patches)
            patch_size (int): 패치 크기

        Returns:
            restored_img (torch.Tensor): 원래 이미지로 복원
        """
        B, C, H, W = shuffled_img.shape
        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        patches = shuffled_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches, patch_size, patch_size)

        # 복원 순서 = inverse permutation
        inv_perm = torch.argsort(permutation, dim=1)
        restored_patches = torch.stack([patches[i, :, inv_perm[i]] for i in range(B)], dim=0)

        restored = restored_patches.view(B, C, h_patches, w_patches, patch_size, patch_size)
        restored = restored.permute(0, 1, 2, 4, 3, 5).contiguous()
        restored_img = restored.view(B, C, H, W)

        return restored_img


    def get_fixed_permutation(self, num_patches, seed=42):
        """항상 동일한 패치 순서를 생성"""
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.randperm(num_patches, generator=g)

    def shuffle_patches_fixed(self, img, patch_size=4, seed=42):
        """
        고정된 순서로 패치를 셔플

        Args:
            img (torch.Tensor): (B, C, H, W)
            patch_size (int): 패치 크기
            seed (int): 고정 순서를 위한 시드

        Returns:
            shuffled_img (torch.Tensor): 셔플된 이미지
            permutation (torch.Tensor): 고정된 셔플 순서
        """
        B, C, H, W = img.shape
        assert H % patch_size == 0 and W % patch_size == 0

        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches, patch_size, patch_size)

        permutation = self.get_fixed_permutation(num_patches, seed)
        shuffled_patches = patches[:, :, permutation]

        shuffled = shuffled_patches.view(B, C, h_patches, w_patches, patch_size, patch_size)
        shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
        shuffled_img = shuffled.view(B, C, H, W)

        return shuffled_img

    def unshuffle_patches_fixed(self, shuffled_img, patch_size=4, seed=42):
        B, C, H, W = shuffled_img.shape
        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        patches = shuffled_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches, patch_size, patch_size)

        permutation = self.get_fixed_permutation(num_patches, seed)
        inv_perm = torch.argsort(permutation)
        restored_patches = patches[:, :, inv_perm]

        restored = restored_patches.view(B, C, h_patches, w_patches, patch_size, patch_size)
        restored = restored.permute(0, 1, 2, 4, 3, 5).contiguous()
        restored_img = restored.view(B, C, H, W)


        return restored_img

    def make_no_require_grad(self):
        for param in self.shuffleirn.parameters():
            param.requires_grad = False

    def forward(self, x, x_h=None, rev=False, stage=1.):
        if not rev:
            if stage == 1:
                x1, x2 = x_h.clone(), x_h
                gen_x1, gen_x2 = self.shuffleirn(x1, x2, rev)
                gen_x_h = gen_x2
                shuffle_x_h = self.shuffle_patches_fixed(gen_x_h, patch_size=1)
                out_y, out_y_h = self.irn(x, self.dwt(shuffle_x_h), rev)

                return out_y, x_h, gen_x_h, shuffle_x_h, out_y_h

            elif stage == 2:
                self.make_no_require_grad()
                x1, x2 = x_h.clone(), x_h
                gen_x1, gen_x2 = self.shuffleirn(x1, x2, rev)
                gen_x_h = gen_x1
                shuffle_x_h = self.shuffle_patches_fixed(gen_x_h, patch_size=1)
                out_y, out_y_h = self.irn(x, self.dwt(shuffle_x_h), rev)

                return out_y, x_h, gen_x_h, shuffle_x_h, out_y_h
        else:
            if stage == 1:
                out_z = self.pm(x).unsqueeze(1) #pm도 수정은 해야할 듯
                out_z_new = out_z.view(-1, self.channel_in, x.shape[-2], x.shape[-1]) #num_video 삭제
                out_x, out_x_h = self.irn(x, out_z_new, rev)
                out_x_h = self.iwt(out_x_h)
                predicted_mask = self.mask_prediction(out_x_h)
                unshuffle_x_h = self.unshuffle_patches_fixed(out_x_h, patch_size=1)
                x1, x2 = unshuffle_x_h.clone(), unshuffle_x_h
                gen_x1, gen_x2 = self.shuffleirn(x1, x2, rev)
                org_x_h = gen_x2
                rec_x_h = self.catanet(org_x_h)


                return out_x, out_x_h, unshuffle_x_h, org_x_h, rec_x_h, predicted_mask

            elif stage==2:
                self.make_no_require_grad()
                out_z = self.pm(x).unsqueeze(1) #pm도 수정은 해야할 듯
                
                out_z_new = out_z.view(-1, self.channel_in, x.shape[-2], x.shape[-1]) #num_video 삭제
                out_x, out_x_h = self.irn(x, out_z_new, rev)
                out_x_h = self.iwt(out_x_h)
                unshuffle_x_h = self.unshuffle_patches_fixed(out_x_h, patch_size=1)
                x1, x2 = unshuffle_x_h.clone(), unshuffle_x_h
                gen_x1, gen_x2 = self.shuffleirn(x1, x2, rev)

                
                org_x_h = gen_x1
                rec_x_h = self.catanet(org_x_h)

                return out_x, out_x_h, unshuffle_x_h, org_x_h, rec_x_h

