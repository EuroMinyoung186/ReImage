import logging
import random
from collections import OrderedDict

import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
import utils.util as util
from .base_model import BaseModel
from models.classifier.Mask_Generator import MixedMaskEmbedder
from models.metric.metric import calculate_psnr, calculate_ssim, calculate_masked_psnr, compute_auc, compute_binary_iou
from models.modules.common import DWT, IWT
from models.modules.loss import (ReconstructionLoss, SmoothnessLoss,
                                 LowFrequencyFFTLoss, PerceptualLoss)
from noise.Valuemetric import GF, MF
from noise.video_noise.JPEG import DiffJPEG

logger = logging.getLogger('base')
dwt = DWT()
iwt = IWT()


class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = -1  # non dist training
            self.world_size = 1

        self.idx = 0
        self.opt = opt
        train_opt = opt['train']
        self.train_opt = train_opt
        self.opt_net = opt['network_G']
        self.remove = opt['remove']

        self.netG = networks.define_G_v2(opt).to(self.device)
        self.netG.print_param_count()

        self.middle_blur = MF().to(self.device)
        self.gf = GF().to(self.device)
        self.mask_embedder = MixedMaskEmbedder()

        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype='l2')
            self.Reconstruction_back = ReconstructionLoss(losstype='l2')
            self.Reconstruction_center = ReconstructionLoss(losstype='center')
            self.Soothness = SmoothnessLoss()
            self.LowFrequency = LowFrequencyFFTLoss()
            self.perceptual_loss = PerceptualLoss().to(self.device)
            self.bce_loss = nn.BCELoss()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params_G = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params_G.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params_G, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_H = data['Cover'].to(self.device)
        if 'Secret' in data:
            self.ref_L = data['Secret'].to(self.device)
        if 'Change' in data:
            self.exchange = data['Change'].to(self.device)
        if 'Mask' in data:
            self.mask = data['Mask'].to(self.device)

    def get_mask_bbox(self, mask):
        """Get the bounding box (ymin, ymax, xmin, xmax) of a binary mask (B=1, C=1)."""
        coords = torch.nonzero(mask[0, 0], as_tuple=False)
        if coords.numel() == 0:
            return None  # empty mask
        ymin, xmin = coords.min(dim=0).values
        ymax, xmax = coords.max(dim=0).values
        return ymin.item(), ymax.item(), xmin.item(), xmax.item()

    def shift_tensor_batch(self, tensor, dy, dx):
        """
        Shift a batch tensor of shape (B, C, H, W) by (dy, dx)
        Empty space is filled with 0.
        """
        B, C, H, W = tensor.shape
        shifted = torch.zeros_like(tensor)

        y_from = max(0, dy)
        y_to = H if dy >= 0 else H + dy
        x_from = max(0, dx)
        x_to = W if dx >= 0 else W + dx

        sy_from = max(0, -dy)
        sy_to = H if dy <= 0 else H - dy
        sx_from = max(0, -dx)
        sx_to = W if dx <= 0 else W - dx

        shifted[:, :, y_from:y_to, x_from:x_to] = tensor[:, :, sy_from:sy_to, sx_from:sx_to]
        return shifted

    def copy_and_move_masked_object_batch(self, image, mask, offsets=None):
        """
        image: (B, C, H, W)
        mask: (B, C, H, W) - binary mask
        offsets: list of (dy, dx) tuples or None (random)
        returns: (B, C, H, W) tensor with copy-moved objects
        """
        B, C, H, W = image.shape
        mask = (mask > 0).float()
        masked_object = image * mask
        result = image.clone()

        for b in range(B):
            bbox = self.get_mask_bbox(mask[b:b+1])
            if bbox is None:
                continue  # skip empty mask
            ymin, ymax, xmin, xmax = bbox

            max_dy_pos = H - 1 - ymax
            max_dy_neg = -ymin
            max_dx_pos = W - 1 - xmax
            max_dx_neg = -xmin

            if offsets is None:
                dy = random.randint(max_dy_neg, max_dy_pos)
                dx = random.randint(max_dx_neg, max_dx_pos)
            else:
                dy, dx = offsets[b]
                # clamp to safe range
                dy = max(max_dy_neg, min(dy, max_dy_pos))
                dx = max(max_dx_neg, min(dx, max_dx_pos))

            shifted_obj = self.shift_tensor_batch(masked_object[b:b+1], dy, dx)
            shifted_mask = self.shift_tensor_batch(mask[b:b+1], dy, dx)
            result[b] = torch.where(shifted_mask.bool(), shifted_obj, result[b])

        return result, shifted_mask
    
    def compute_true_positives(self, pred, gt):
        """
        pred: torch.Tensor of shape (1, H, W) - predicted binary mask
        gt: torch.Tensor of shape (1, H, W) - ground truth binary mask
        Returns: int - number of false positives
        """
        # Ensure binary masks: 0 or 1
        """
        pred: torch.Tensor of shape (1, H, W) - predicted mask (binary or [0,1])
        gt:   torch.Tensor of shape (1, H, W) - ground-truth mask (binary or [0,1])
        
        Returns:
            FPR: float - False Positive Rate
        """
        pred_bin = (pred > 0.5).int()
        gt_bin = (gt > 0.5).int()

        TP = ((pred_bin == 1) & (gt_bin == 1)).sum().item()
        FN = ((pred_bin == 0) & (gt_bin == 1)).sum().item()

        fpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return fpr

    def optimize_parameters(self, epoch, wandb, train_type='all'):
        self.netG.train()
        self.optimizer_G.zero_grad()
        self.idx += 1

        dummy_img = np.zeros(self.ref_L.shape)
        visual_mask = self.mask_embedder(dummy_img, None, verbose=True).to(self.device)

        # forward: embed the watermark
        container, _, _, hidden_watermark, _ = self.netG(x=dwt(self.real_H), x_h=self.ref_L, rev=False, stage=1)
        hidden_watermark = hidden_watermark.contiguous()
        l_low_frequency = self.Soothness(hidden_watermark)
        l_low_frequency2 = self.LowFrequency(hidden_watermark)

        container = iwt(container)
        self.containers = container.clone().detach()
        Gt_ref = self.real_H.detach()
        l_forw_fit = self.Reconstruction_forw(container, Gt_ref) + 10 * self.perceptual_loss(container, Gt_ref)

        # random degradation
        choice = random.randint(0, 3)
        if choice == 0:
            NL = float((np.random.randint(1, 16)) / 255)
            noise = np.random.normal(0, NL, container.shape)
            torchnoise = torch.from_numpy(noise).cuda().float()
            container = container + torchnoise
        elif choice == 1:
            NL = int(np.random.randint(70, 95))
            self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
            container = self.DiffJPEG(container)
        elif choice == 2:
            container = self.middle_blur(container)
        else:
            container = self.gf(container)

        # random tampering: splice or copy-move
        choice = random.randint(0, 3)
        if choice == 0:
            container = (1 - visual_mask) * container + visual_mask * self.exchange
        else:
            container, visual_mask = self.copy_and_move_masked_object_batch(container, visual_mask)

        # backward: localize and recover
        out_x, out_x_h, unshuffle_x_h, org_x_h, rec_x_h, predicted_mask = self.netG(x=dwt(container), rev=True, stage=1)

        l_center_x = self.Reconstruction_center(out_x_h, hidden_watermark.detach(), 1 - visual_mask)
        l_recon_x_2 = self.Reconstruction_center(org_x_h, self.ref_L.detach())
        l_recon_x = self.Reconstruction_center(rec_x_h, self.ref_L.detach()) + 10 * self.perceptual_loss(rec_x_h, self.ref_L.detach())
        l_bce = self.bce_loss(predicted_mask, visual_mask)

        loss = 20 * l_recon_x + l_forw_fit * 150 + 10 * l_center_x + l_recon_x_2 * 15 \
            + l_low_frequency2 * 10 + l_low_frequency * 10 + 10 * l_bce

        if self.world_size > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss /= self.world_size

        loss.backward()
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
        self.optimizer_G.step()

        # set log
        self.log_dict['l_low_frequency'] = l_low_frequency.item()
        self.log_dict['l_recon_x'] = l_recon_x.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_center_x'] = l_center_x.item()
        self.log_dict['l_recon_x_2'] = l_recon_x_2.item()
        self.log_dict['l_bce'] = l_bce.item()

        wandb.log(self.log_dict)
        if self.idx % 100 == 1:
            wandb.log({"SSEC": wandb.Image(hidden_watermark[0].detach().cpu()),
                       "SEC": wandb.Image(self.ref_L[0].detach().cpu()),
                       "EXT": wandb.Image(out_x_h[0].detach().cpu()),
                       "USEC": wandb.Image(org_x_h[0].detach().cpu()),
                       "CON": wandb.Image(self.containers[0].detach().cpu()),
                       "ATT": wandb.Image(container[0].detach().cpu()),
                       "COV": wandb.Image(self.real_H[0].detach().cpu()),
                       "REC": wandb.Image(rec_x_h[0].detach().cpu())})

    def test(self):
        self.netG.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            self.idx += 1

            b, c, h, w = self.real_H.shape

            visual_mask = self.mask

            container, _, gen_x_h, hidden_watermark, _  = self.netG(x=dwt(self.real_H), x_h=self.ref_L, rev=False, stage=1.)
            hidden_watermark = hidden_watermark.contiguous()
            hidden_watermark2 = hidden_watermark.clone()
            l_low_frequency = self.Soothness(hidden_watermark)

            container = iwt(container)
            self.container = container.clone()
            Gt_ref = self.real_H.detach()

            psnr_watermark = calculate_psnr(container.squeeze(0).detach().cpu().numpy(), self.real_H.squeeze(0).detach().cpu().numpy())
            ssim_watermark = calculate_ssim(container.squeeze(0).permute(1,2,0).detach().cpu().numpy(), self.real_H.squeeze(0).permute(1,2,0).detach().cpu().numpy())
            lpips_watermark = self.perceptual_loss(container * 2 -1, self.real_H.detach() * 2 -1).item()
            

            import random
            choice = random.randint(0, 2)

            
            if choice == 0:
                NL = float(np.random.randint(1, 9)/255)
                noise = np.random.normal(0, NL, container.shape)
                torchnoise = torch.from_numpy(noise).cuda().float()
                container = container + torchnoise

            elif choice == 1:
                NL = int(np.random.randint(90, 91))
                self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
                container = self.DiffJPEG(container)
                
            
            elif choice == 2:
                vals = 10**4

                container_non_neg = torch.clamp(container, min=0)
                if random.random() < 0.5:
                    noisy_img_tensor = torch.poisson(container_non_neg * vals) / vals
                else:
                    img_gray_tensor = torch.mean(container_non_neg, dim=0, keepdim=True)
                    noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                    noisy_img_tensor = container_non_neg + (noisy_gray_tensor - img_gray_tensor)

                container = torch.clamp(noisy_img_tensor, 0, 1)
            '''
            elif choice == 3:
                container = self.brightness(container)

            elif choice == 4:
                container = self.gf(container) 
            

            elif choice == 5:
                container = self.resize(container)

            elif choice==6:
                container = self.middle_blur(container)
                #self.ref_L = self.middle_blur(self.ref_L)

            elif choice==7:
                container = self.sp(container)

            elif choice==8:
                container = self.contrast(container)

            elif choice==9:
                container = self.hue(container)

            else:
                container, self.mask = self.horizontal_flip(container, self.mask)
   
            '''
            container = (1-visual_mask) * container + (visual_mask) * self.exchange

            out_x, out_x_h, unshuffle_x_h, org_x_h, rec_x_h, predicted_mask = self.netG(x=dwt(container), rev=True, stage=1.)
            auc = compute_auc(predicted_mask, self.mask.int())
            
            predicted_mask_thres = (predicted_mask > 0.5).float()
            iou = compute_binary_iou(predicted_mask_thres, self.mask)
            f1 =  self.compute_true_positives(predicted_mask_thres, self.mask)

            org_x_h = org_x_h.contiguous()

            self.containers = container.squeeze(0).detach().cpu()
            self.attacked = predicted_mask.repeat(1, 3, 1, 1).squeeze(0).detach().cpu()
            self.unshuffle = self.container.squeeze(0).detach().cpu()
            self.extract = self.mask.squeeze(0).detach().cpu()
            self.original = self.real_H.squeeze(0).detach().cpu()
            tmp = container * (1 - predicted_mask) + rec_x_h * (predicted_mask)
            self.restoration = tmp.squeeze(0).detach().cpu()


            psnr_recovery = calculate_psnr(self.original.detach().cpu().numpy(), self.restoration.detach().cpu().numpy())
            mask_pnsr_recovery = calculate_masked_psnr(self.original.detach().cpu(), self.restoration.detach().cpu(), self.mask.squeeze(0).detach().cpu())
            ssim_recovery = calculate_ssim(self.original.permute(1,2,0).detach().cpu().numpy(), self.restoration.permute(1,2,0).detach().cpu().numpy())
            lpips_recovery = self.perceptual_loss(self.real_H * 2 -1, tmp.detach() * 2 -1).item()
            
            print(psnr_recovery, ssim_recovery, lpips_recovery, mask_pnsr_recovery)

            

            #hidden_watermark = hidden_watermark * (1-visual_mask)

            

            


            return psnr_watermark, ssim_watermark, psnr_recovery, ssim_recovery, lpips_watermark, lpips_recovery, mask_pnsr_recovery, self.containers, self.attacked, self.restoration, self.original, self.unshuffle, self.extract , auc, iou, f1, _, _
            
    def img_recovery(self, image_edited, orig, mask):
        self.netG.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            image_edited = image_edited.to(self.device)
            out_x, out_x_h, unshuffle_x_h, org_x_h, rec_x_h, predicted_mask = self.netG(x=dwt(image_edited), rev=True, stage=1.)

            self.restoration = rec_x_h * mask + image_edited * (1 - mask)

            psnr_recovery = calculate_psnr(orig.detach().cpu().numpy(), self.restoration.detach().cpu().numpy())
            ssim_recovery = calculate_ssim(orig.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), self.restoration.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            lpips_recovery = self.perceptual_loss(orig.to(self.device).squeeze(0) * 2 - 1, self.restoration.to(self.device).squeeze(0).detach() * 2 - 1).item()
            mask_psnr_recovery = calculate_masked_psnr(orig.squeeze(0).detach().cpu(), self.restoration.squeeze(0).detach().cpu(), mask.squeeze(0).detach().cpu())

            print("PSNR Recovery:", psnr_recovery)
            print("SSIM Recovery:", ssim_recovery)
            print("LPIPS Recovery:", lpips_recovery)
            print("Masked PSNR Recovery:", mask_psnr_recovery)
            self.restoration = util.tensor2img(self.restoration)

            return self.restoration, psnr_recovery, ssim_recovery, lpips_recovery, mask_psnr_recovery

    def get_current_log(self):
        return self.log_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        print('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
