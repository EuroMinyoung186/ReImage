import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from models import create_model


def init_dist(backend='nccl', **kwargs):
    '''initialization for distributed training'''
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.',
                        default='options/train/train_shuffle.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--project_name', type=str, default='recovery')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['remove'] = args.remove
    opt['name'] = args.project_name

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # distributed training settings
    if args.launcher == 'pytorch':
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    opt['rank'] = rank

    wandb.init(project=args.project_name)

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    # create train dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    dataset_opt = opt['datasets']['train']
    train_set = create_dataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    opt['total_epochs'] = total_epochs
    if opt['dist']:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
    else:
        train_sampler = None
    train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
    logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            model.feed_data(train_data)
            model.optimize_parameters(epoch=epoch, wandb=wandb, train_type='all')
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
