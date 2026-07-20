import torch
import torch.nn as nn
import numpy as np
from lpips import LPIPS


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.perceptual_loss = LPIPS(net='vgg').eval()

    def forward(self, imgs, imgs_w):
        return self.perceptual_loss(imgs, imgs_w)

class LowFrequencyFFTLoss(nn.Module):
    def __init__(self, scale='log', weight_map=None):
        super().__init__()
        self.scale = scale
        self.weight_map = weight_map  # optional: custom frequency weighting

    def forward(self, x):
        # x: (B, C, H, W)
        fft = torch.fft.fft2(x, norm='ortho')
        mag = torch.abs(fft)  # (B, C, H, W)

        # Optionally shift zero frequency to center
        mag = torch.fft.fftshift(mag, dim=(-2, -1))

        if self.weight_map is None:
            # Create frequency weight map: higher frequencies → higher weights
            B, C, H, W = x.shape
            fy = torch.fft.fftfreq(H).to(x.device).reshape(-1, 1)
            fx = torch.fft.fftfreq(W).to(x.device).reshape(1, -1)
            freq_radius = torch.sqrt(fx**2 + fy**2)  # (H, W)
            freq_weight = freq_radius / freq_radius.max()  # normalize to [0, 1]
            freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            freq_weight = self.weight_map.to(x.device)

        # Apply weighting: penalize high-frequency magnitude
        loss = (mag * freq_weight).mean()
        return loss

# video Stegangraphy 결과를 위해 사용되는 loss 함수
class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target, mask = 1.):
        if self.losstype == 'l2':
            return torch.mean(torch.sum(((x - mask * target) ** 2), (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = (x - target * mask)
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'center':
            return torch.mean(torch.sum(((x - mask * target) ** 2), (1, 2, 3)))

        else:
            print("reconstruction loss type error!")
            return 0
        
class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        """
        이미지의 pixel 간 변화량을 줄이도록 유도하는 loss (low-frequency 성질 강화).

        Args:
            image (Tensor): (B, C, H, W) 형식의 이미지

        Returns:
            loss (Tensor): 평균 smoothness loss
        """
        dh = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # height 방향 변화
        dw = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # width 방향 변화

        loss = dh.mean() + dw.mean()
        return loss

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, anchor_features, target_features):
        # anchor_features: (B*T, C, H, W)
        # target_features: (B,*T C, H, W)
        # change anchor_features to (B,T,H,W,C)
        BT, C, H, W = anchor_features.shape
        
        anchor_features = anchor_features.view(BT, 256, 16, 4).view(BT, 256, 8, 8)
        target_features = target_features.view(BT, 256, 16, 4).view(BT, 256, 8, 8)

        anchor_features = anchor_features.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        target_features = target_features.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        
        anchor_features = anchor_features / (torch.norm(anchor_features, dim=1, keepdim=True) + self.eps)
        target_features = target_features / (torch.norm(target_features, dim=1, keepdim=True) + self.eps)
                                             
        logits = torch.matmul(anchor_features, target_features.t())

        print(logits)

        labels = torch.arange(anchor_features.shape[0]).to(anchor_features.device)

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')

        return loss
    
    def print_current_values(self, values):
        if values.dim() == 1:
            for i in range(values.size(0)):
                print('{:.4f}'.format(values[i]), end=' ')

        elif values.dim() >= 2:
            for i in range(values.size(0)):
                for j in range(values.size(1)):
                    print('{:.4f}'.format(values[i, j]), end=' ')
                print('')

