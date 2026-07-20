import os
import argparse
import random

import torch
import wandb

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.',
                        default='options/test/test.yml')
    parser.add_argument('--project_name', type=str, default='recovery')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save output images.')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['remove'] = args.remove
    opt['name'] = args.project_name
    opt['dist'] = False

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    wandb.init(project=args.project_name)

    # load checkpoint state
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # sets pretrain_model paths
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True

    # create val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)

    # create model
    model = create_model(opt)
    if resume_state:
        model.resume_training(resume_state)

    # evaluation
    img_dir = args.save_dir
    sub_dirs = ['GT', 'WATERMARK', 'RECOVER', 'ATTACKED', 'UNSHUFFLE', 'EXTRACT']
    util.mkdir(img_dir)
    for d in sub_dirs:
        util.mkdir(os.path.join(img_dir, d))

    total_idx = 0.0
    avg = {'PSNR_watermark': 0.0, 'SSIM_watermark': 0.0,
           'PSNR_recovery': 0.0, 'SSIM_recovery': 0.0,
           'LPIPS_watermark': 0.0, 'LPIPS_recovery': 0.0,
           'mask_psnr_recovery': 0.0, 'AUC': 0.0, 'IOU': 0.0, 'F1': 0.0,
           'watermarking_time': 0.0, 'recovery_time': 0.0}

    for val_data in val_loader:
        model.feed_data(val_data)
        (psnr_watermark, ssim_watermark, psnr_recovery, ssim_recovery,
         lpips_watermark, lpips_recovery, mask_psnr_recovery,
         watermarked_img, attacked_img, recovered_img, original_img,
         extract_img, unshuffled_img, auc, iou, f1,
         watermarking_time, recovery_time) = model.test()

        wandb.log({'psnr_watermark': psnr_watermark, 'ssim_watermark': ssim_watermark,
                   'psnr_recovery': psnr_recovery, 'ssim_recovery': ssim_recovery,
                   'lpips_watermark': lpips_watermark, 'lpips_recovery': lpips_recovery,
                   'mask_psnr_recovery': mask_psnr_recovery})

        total_idx += 1
        avg['PSNR_watermark'] += psnr_watermark
        avg['SSIM_watermark'] += ssim_watermark
        avg['PSNR_recovery'] += psnr_recovery
        avg['SSIM_recovery'] += ssim_recovery
        avg['LPIPS_watermark'] += lpips_watermark
        avg['LPIPS_recovery'] += lpips_recovery
        avg['mask_psnr_recovery'] += mask_psnr_recovery
        avg['AUC'] += auc
        avg['IOU'] += iou
        avg['F1'] += f1
        avg['watermarking_time'] += watermarking_time
        avg['recovery_time'] += recovery_time

        outputs = {'GT': original_img, 'WATERMARK': watermarked_img,
                   'RECOVER': recovered_img, 'ATTACKED': attacked_img,
                   'UNSHUFFLE': unshuffled_img, 'EXTRACT': extract_img}
        # Save each output under its own subdirectory, keeping the input filename
        img_name = val_data['Name'][0]
        for name, img in outputs.items():
            save_img_path = os.path.join(img_dir, name, '{:s}.png'.format(img_name))
            util.save_img(util.tensor2img(img), save_img_path)

    for k, v in avg.items():
        print('{}: {:.6f}'.format(k, v / total_idx))

    wandb.log({k: v / total_idx for k, v in avg.items()
               if k not in ('AUC', 'IOU', 'F1', 'watermarking_time', 'recovery_time')})


if __name__ == '__main__':
    main()
