import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .base_model import BaseModel, freeze
from codes.models.modules.arch_spkdeblur_transformer import TinySwinIR,SpkDeblurNet
import numpy as np

logger = logging.getLogger('base')

class SpikeDeblur_Model(BaseModel):
    def __init__(self, opt):
        super(SpikeDeblur_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt #TODO

        self.MoBanNet = SpkDeblurNet(S_in_chs=opt["datasets"]["train"]["length_spike"]).to(self.device)

        if opt['dist']:
            # self.MoBanNet = DistributedDataParallel(self.MoBanNet, device_ids=[torch.cuda.current_device()])

            # replace BN as SyncBN
            self.MoBanNet = nn.SyncBatchNorm.convert_sync_batchnorm(self.MoBanNet)
            self.MoBanNet = DistributedDataParallel(self.MoBanNet, device_ids=[self.rank],
                            output_device=self.rank,find_unused_parameters=True,broadcast_buffers=False)
        else:
            self.MoBanNet = DataParallel(self.MoBanNet)
            # pass

        # print network
        self.print_network()
        self.load()

        # freeze(self.MoBanNet.module.bsn)
        # logger.info("+++ FlowNet.fnet frozen. +++")

        if self.usage == "train":
            self.MoBanNet.train()

            # optimizers
            optim_params = []
            for k, v in self.MoBanNet.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                                # weight_decay=train_opt['weight_decay'],
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            # self.optimizer = torch.optim.AdamW(optim_params, lr=train_opt['lr'],
            #                                     weight_decay=train_opt['weight_decay'])
            #                                     # eps=train_opt['epsilon']) 
            #                                     # betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=train_opt['lr_milestones'],
                                                             gamma=train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max = train_opt['T_max'], 
                                                                   eta_min=train_opt['eta_min']))
            elif train_opt['lr_scheme'] == 'OneCycleLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.OneCycleLR(optimizer, train_opt['lr'], train_opt['niter']+100,
                                                            pct_start=0.05, cycle_momentum=False, 
                                                            anneal_strategy='linear'))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        else:
            self.MoBanNet.eval()

    def feed_data(self, data):
        self.t = data['t'].to(self.device) # useless
        self.gt = data['gt'].to(self.device)
        self.gt_gray = data['gt_gray'].to(self.device)
        self.blur_gray = data['blur_gray'].to(self.device)
        self.blur = data['blur'].to(self.device)
        self.spike = data['spike'].to(self.device)
        self.tfi = data['tfi'].to(self.device)
        # self.flow = data['flow'].to(self.device)

    def feed_test_data(self, data):
        self.t = data['t'].to(self.device)
        self.gt = data['gt'].to(self.device)  
        self.gt_gray = data['gt_gray'].to(self.device)
        self.blur_gray = data['blur_gray'].to(self.device)
        self.blur = data['blur'].to(self.device)
        self.spike = data['spike'].to(self.device)
        self.tfi = data['tfi'].to(self.device)
        # self.flow = data['flow'].to(self.device)

    def loss_forward(self, pred, gt):
        # loss = nn.MSELoss()(pred, gt)
        loss = nn.L1Loss()(pred, gt)

        return loss

    def get_gt_mask(self,blur_gray,sharp_gray):
        hard_mask = torch.abs(blur_gray-sharp_gray)
        hard_mask = (hard_mask-torch.min(hard_mask))/(torch.max(hard_mask)-torch.min(hard_mask))
        hard_mask[hard_mask<=0.1] = 1.
        hard_mask[hard_mask!=1] = 0.
        return hard_mask

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        self.pred, self.recon, self.quick_sharp, self.soft_mask = self.MoBanNet(self.blur, self.spike, self.t, self.tfi, self.blur_gray)
        # self.pred, self.recon = self.MoBanNet(self.blur, self.spike, self.t, self.tfi, self.blur_gray)
        # self.gt_mask = self.get_gt_mask(self.blur_gray,self.gt_gray)
        l_n2n = self.loss_forward(self.pred, self.gt) + self.loss_forward(self.recon, self.gt_gray) + self.loss_forward(self.quick_sharp, self.gt)
        # l_n2n += 0.1*self.loss_forward(self.gt_mask,self.soft_mask)
        # self.pred, self.recon = self.MoBanNet(self.blur, self.spike, self.t)
        # l_n2n = self.loss_forward(self.pred, self.gt) + self.loss_forward(self.recon, self.gt_gray)

        # total loss
        loss = 0.0
        loss += l_n2n

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.MoBanNet.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer.step()

        # set log
        self.log_dict['l_all'] = loss.item()

    def test(self):
        self.MoBanNet.eval()
        with torch.no_grad():
            # self.pred, self.recon = self.MoBanNet(self.blur, self.spike, self.t)
            self.pred, self.recon, self.quick_sharp, self.soft_mask = self.MoBanNet(self.blur, self.spike, self.t, self.tfi, self.blur_gray)
            # self.pred, self.recon = self.MoBanNet(self.blur, self.spike, self.t, self.tfi, self.blur_gray)

        self.MoBanNet.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict["pred"] = self.pred.detach().cpu()
        out_dict["blur"] = self.blur.detach().cpu()
        out_dict["gt"] = self.gt.detach().cpu()
        out_dict["spike"] = self.spike.detach().cpu()

        out_dict["recon"] = self.recon.detach().cpu()
        out_dict["gt_gray"] = self.gt_gray.detach().cpu()
        out_dict["soft_mask"] = self.soft_mask.detach().cpu()
        out_dict["quick_sharp"] = self.quick_sharp.detach().cpu()
        # out_dict["flow"] = self.flow.detach().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.MoBanNet)
        if isinstance(self.MoBanNet, nn.DataParallel) or isinstance(self.MoBanNet, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.MoBanNet.__class__.__name__,
                                             self.MoBanNet.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.MoBanNet.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        self.load_path = self.opt['path']['pretrain_model']
        if self.load_path is not None:
            logger.info('Loading model.PTH [{:s}] ...'.format(self.load_path))
            self.load_network(self.load_path, self.MoBanNet, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.MoBanNet, self.opt["model"], iter_label)
