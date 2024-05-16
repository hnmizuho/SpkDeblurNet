import os
import math
import argparse
import random
import logging
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.data_sampler import DistIterSampler

import options.options as option
from utils import misc_utils, batch_utils
from dataset import create_dataloader, create_dataset
from models import create_model
# from torchvision.utils import flow_to_image

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, usage="train")

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print('Enabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt)  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        misc_utils.mkdirs((path for key, path in opt['path'].items() if key in [
            "experiments_root", "models", "training_state", "log", "val_images"
            ]))
            
        # config loggers. Before it, the log will not work
        misc_utils.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        misc_utils.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt["expr_rootname"])
    else:
        misc_utils.setup_logger('base', opt['path']['log'], 'train_', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    misc_utils.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True # faster
    # torch.backends.cudnn.deterministic = True # reproducible

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch. whatever.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt, phase)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, phase, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(val_set, dataset_opt, phase, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['d_name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    saved_gt = False
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0 and current_step>0:
                psnr_val_rgb = []
                ssim_val_rgb = []
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals() #仅仅展示一个batch的第一张图                    
                    pred = misc_utils.tensor2img(visuals['pred'][0])  # --> uint8 0-255
                    recon = misc_utils.tensor2img(visuals['recon'][0])  # --> uint8 0-255
                    soft_mask = misc_utils.tensor2img(visuals['soft_mask'][0])  # --> uint8 0-255
                    blur = misc_utils.tensor2img(visuals['blur'][0])  # --> uint8 0-255
                    gt = misc_utils.tensor2img(visuals['gt'][0])  # --> uint8 0-255
                    gt_gray = misc_utils.tensor2img(visuals['gt_gray'][0])  # --> uint8 0-255
                    # spike_tfp = misc_utils.tensor2img(torch.mean(visuals['spike'][0],dim=0))  # --> uint8 0-255
                    # flow = flow_to_image(visuals['flow'][0]).permute(1,2,0).numpy()

                    single_psnr = batch_utils.batch_PSNR(model.pred[0].unsqueeze(0).detach().cpu(), model.gt[0].unsqueeze(0).detach().cpu(), 1.)
                    single_psnr_recon = batch_utils.batch_PSNR(model.recon[0].unsqueeze(0).detach().cpu(), model.gt_gray[0].unsqueeze(0).detach().cpu(), 1.)
                    img_name = os.path.splitext(os.path.basename(val_data['img_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    misc_utils.mkdir(img_dir)

                    # Save recon images
                    save_img_path = os.path.join(img_dir, '{:s}_{:s}.png'.format(img_name, str(current_step) + '_psnr_{:.2f}'.format(single_psnr)))
                    misc_utils.save_img(pred, save_img_path)
                    save_img_path_recon = os.path.join(img_dir, '{:s}_{:s}_Recon.png'.format(img_name, str(current_step) + '_psnr_{:.2f}'.format(single_psnr_recon)))
                    misc_utils.save_img(recon, save_img_path_recon)
                    save_img_path_soft_mask = os.path.join(img_dir, '{:s}_{:s}_soft_mask.png'.format(img_name, str(current_step)))
                    misc_utils.save_img(soft_mask, save_img_path_soft_mask)
                    # save_img_path_flow = os.path.join(img_dir, '{:s}_{:s}_flow.png'.format(img_name, str(current_step)))
                    # misc_utils.save_img(flow, save_img_path_flow)

                    # Save ground truth
                    if not saved_gt:
                        save_img_path_gt = os.path.join(img_dir, '{:s}_GT.png'.format(img_name))
                        misc_utils.save_img(gt, save_img_path_gt)
                        save_img_path_gt_gray = os.path.join(img_dir, '{:s}_GT_Gray.png'.format(img_name))
                        misc_utils.save_img(gt_gray, save_img_path_gt_gray)
                        save_img_path_blur = os.path.join(img_dir, '{:s}_blur.png'.format(img_name))
                        misc_utils.save_img(blur, save_img_path_blur)
                        # save_img_path_spiketfp = os.path.join(img_dir, '{:s}_spiketfp.png'.format(img_name))
                        # misc_utils.save_img(spike_tfp, save_img_path_spiketfp)

                    # calculate metric 注意上面的pred是一个batch的第一张图，而model.pred是整个batch的图
                    psnr_val_rgb.append(batch_utils.batch_PSNR(model.pred.detach().cpu(), model.gt.detach().cpu(), 1.))
                    ssim_val_rgb.append(batch_utils.batch_SSIM(model.pred.detach().cpu(), model.gt.detach().cpu()))

                saved_gt = True
                psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
                ssim = sum(ssim_val_rgb) / len(ssim_val_rgb)

                # log psnr and ssim
                logger.info('# Validation # PSNR: {:.4e}. # SSIM: {:.4e}.'.format(psnr, ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}. ssim: {:.4e}.'.format(
                    epoch, current_step, psnr, ssim))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', psnr, current_step)
                    tb_logger.add_scalar('ssim', ssim, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0 and current_step>0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step, opt["model"])

    if rank <= 0:
        logger.info('Saving the final model and state.')
        model.save(current_step)
        model.save_training_state(epoch, current_step, opt["model"])
        logger.info('End of training.')


if __name__ == '__main__':
    main()
