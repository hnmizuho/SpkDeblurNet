import os, time
import sys, random
sys.path.append('..')

from pathlib import Path

import cv2
# import h5py
# import hdf5plugin
# import imageio
import numpy as np
from tqdm import tqdm
import torch
# from numba import jit
from PIL import Image
from torch.utils.data import Dataset
from utils.spike_utils import load_vidar_dat,middleTFI
from utils.flow_visualization import flow_visualization
import torchvision.transforms as transforms
from utils import misc_utils,shuffleMixer_transforms
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights 

# imageio.plugins.freeimage.download()

# os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


X4K10000FPS_RESOLUTION_DICT = {
    'train': {'H': 768, 'W': 768, 'Hs': 384, 'Ws': 384},
    'test': {'H': 4096, 'W': 2160},
    'val': {'H': 512, 'W': 512, 'Hs': 256, 'Ws': 256}
}

class Sequence(Dataset):
    def __init__(
        self, 
        root_path: Path, 
        seq_path: str, # 加载不同数据集。seq_path和phase不冲突，有时候自监督算法希望直接用测试集进行训练，需要两个参数配合。
        length_spike: int=65,
        crop_size: int=256,
        phase: str='train', # 指定用途，仅当test时不加载gt。
        transforms=None,
        reduce_scale: int=1, # not 1 for less validation data
        ):

        assert phase in ['train', 'val', 'test'], 'ERROR: \'phase\' should be train, val or test, but got %s' % phase
        self.phase = phase
        self.seq_path = seq_path
        self.reduce_scale = reduce_scale

        self.dataset_root = root_path
        self.length_spike = length_spike
        self.crop_size = crop_size
        self.blurry_data_list = self._make_datalist() # 002/occ008.320_f2881/0001_002.png, ...
        self.H = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['H']
        self.W = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['W']
        self.Hs = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['Hs']
        self.Ws = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['Ws']

        self.transforms = transforms

        # self.raft_img = raft_small(weights=Raft_Small_Weights.DEFAULT)
        # self.raft_img = self.raft_img.eval()

    def _make_datalist(self):
        data_list = []

        rgb_root = os.path.join(self.dataset_root, self.seq_path+"_blurry_{}".format(self.length_spike))
        rgb_subroots = os.listdir(rgb_root) # 002, 003, ...

        for rgb_subroot in rgb_subroots:
            fullpath = os.path.join(rgb_root, rgb_subroot)
            rgb_folders = os.listdir(fullpath) # occ008.320_f2881, ...
            rgb_folders = [os.path.join(fullpath, folder) for folder in rgb_folders]
            data_list.extend(rgb_folders)

        data_list2 = []
        for rgb_folder in data_list:
            rgbs = os.listdir(rgb_folder)
            rgb_paths = [os.path.join(rgb_folder, folder) for folder in rgbs]
            data_list2.extend(rgb_paths)

        return data_list2

    def _get_ori_index(self, filename): 
        # filename 002/occ008.320_f2881/0001_002.png
        filename = os.path.split(filename)[1] 
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts)==1:
            return int(parts[0])*4
        elif len(parts)==2:
            return int(parts[0])*4 + int(parts[1]) + 1
        else:
            raise NotImplementedError("Error dataset filename.")
    
    def _modify_filename_from_offest(self, blurry_rgb_path, offest):
        # 怪异的四进制
        # 例如，0000 + 3 = 0000_002, 0000 + 4 = 0001
        # 0001_002 - 3 = 0001
        # 0001_002 + 3 = 0002_001
        if offest == 0:
            return blurry_rgb_path
        filename = os.path.split(blurry_rgb_path)[1] 
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts)==1:
            parts1 = int(parts[0])
            parts2 = 0
        elif len(parts)==2:
            parts1 = int(parts[0])
            parts2 = int(parts[1]) + 1
        new_parts2 = (offest + parts2)%4
        new_parts1 = parts1 + (offest + parts2)//4
        if new_parts2 == 0:
            partsname = "%04d.png" % new_parts1
        else:
            partsname = "%04d" % new_parts1 + "_" + "%03d.png" % (new_parts2 - 1)
        return os.path.join(os.path.split(blurry_rgb_path)[0], partsname)

    # def __getitem__(self, index):
    #     index *= self.reduce_scale
    #     offest = 0
    #     # offest = random.randint(-self.length_spike//2, self.length_spike//2)

    #     blurry_rgb_path = self.blurry_data_list[index]
    #     sharp_rgb_path = blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_fullRGB")
    #     sharp_rgb_path = self._modify_filename_from_offest(sharp_rgb_path, offest)
    #     spike_path = os.path.join(os.path.split(blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_spike_2xds"))[0], 'spike.dat')

    #     # get sharp image and blur image
    #     blur_img = Image.open(blurry_rgb_path).convert('RGB')
    #     blur_img = transforms.ToTensor()(blur_img)
    #     sharp_img = Image.open(sharp_rgb_path).convert('RGB')
    #     sharp_img = transforms.ToTensor()(sharp_img)

    #     # get spike
    #     gt_center_ori_index = self._get_ori_index(blurry_rgb_path) + offest
    #     # left_idx = gt_center_ori_index - self.length_spike//2
    #     # right_idx = gt_center_ori_index + self.length_spike//2

    #     # spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)[left_idx:right_idx+1, ...]
    #     # spike = middleTFI(spike,middle=self.length_spike//2)
    #     spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)
    #     spike = middleTFI(spike,middle=gt_center_ori_index)
    #     spike = torch.from_numpy(spike).unsqueeze(0).float()
    #     # spike = torch.mean(spike,0).unsqueeze(0) #权宜之计
    #     spike = torch.nn.functional.interpolate(spike.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0)

    #     if self.transforms:
    #         blur_img, sharp_img, spike = self.transforms([blur_img, sharp_img, spike])        

    #     return {
    #         'blur': blur_img,
    #         'gt': sharp_img,
    #         'spike': spike,
    #         "img_path": "__".join(str(blurry_rgb_path).split('/')[-4:]) # rename to sole name for compatibility
    #     }

    def __getitem__(self, index):
        index *= self.reduce_scale
        offest = 0
        t=0.5
        # offest = random.randint(-self.length_spike//2, self.length_spike//2)
        # if self.seq_path=="train":
        #     # if self.length_spike==65:
        #     #     dt=4
        #     # elif self.length_spike==33:
        #     #     dt=2
        #     # offest = random.choice([dt*i for i in range(-5,6)])
        #     # t = (offest+dt*5)/(dt*10)
        #     ttmp = random.choice([(0,32/64),(4,36/64),(8,40/64),(12,44/64),(16,48/64),(-4,28/64),(-8,24/64),(-12,20/64),(-16,16/64)])
        #     offest = ttmp[0]
        #     t = ttmp[1]
        # elif self.seq_path=="val":
        #     offest=0
        #     t=0.5

        blurry_rgb_path = self.blurry_data_list[index]
        sharp_rgb_path = blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_fullRGB")
        sharp_rgb_path = self._modify_filename_from_offest(sharp_rgb_path, offest)
        spike_path = os.path.join(os.path.split(blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_spike_2xds"))[0], 'spike.dat')
        # spike_path = blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_spike_2xds_{}".format(self.length_spike))
        # spike_path = spike_path[:-3]+"dat"

        # get sharp image and blur image
        blur_img = Image.open(blurry_rgb_path).convert('RGB')
        blur_img = transforms.ToTensor()(blur_img)
        blur_img_gray = Image.open(blurry_rgb_path).convert('L')
        blur_img_gray = transforms.ToTensor()(blur_img_gray)
        sharp_img = Image.open(sharp_rgb_path).convert('RGB')
        sharp_img_gray = Image.open(sharp_rgb_path).convert('L')
        sharp_img = transforms.ToTensor()(sharp_img)
        sharp_img_gray = transforms.ToTensor()(sharp_img_gray)

        # get spike
        # gt_center_ori_index = self._get_ori_index(blurry_rgb_path)
        gt_center_ori_index = self._get_ori_index(blurry_rgb_path) + offest
        left_idx = gt_center_ori_index - self.length_spike//2
        right_idx = gt_center_ori_index + self.length_spike//2
        spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)
        ori_spike_len = spike.shape[0]
        if left_idx < 0:
            spike = spike[:right_idx+1, ...]
            spike = torch.from_numpy(spike).float()
            zeros = torch.ones(0-left_idx,self.Hs,self.Ws)
            spike = torch.cat([zeros,spike],0)
        if right_idx >= ori_spike_len:
            spike = spike[left_idx:, ...]
            spike = torch.from_numpy(spike).float()
            zeros = torch.ones(right_idx-ori_spike_len+1,self.Hs,self.Ws)
            spike = torch.cat([spike,zeros],0)
        else:
            spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)[left_idx:right_idx+1, ...]
            spike = torch.from_numpy(spike).float()

        # spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)
        # spike = torch.from_numpy(spike).float()

        # spike = middleTFI(spike, middle=self.length_spike//2)/0.5
        # spike = torch.from_numpy(spike).unsqueeze(0).float()
        # spike = torch.nn.functional.interpolate(spike.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0) 
        # # # spike = spike.repeat(3,1,1)
        
        if self.transforms:
            [blur_img, sharp_img, sharp_img_gray, blur_img_gray], spike = shuffleMixer_transforms.paired_random_crop([blur_img, sharp_img, sharp_img_gray, blur_img_gray],spike,self.crop_size,2)      
            # [blur_img, sharp_img, sharp_img_gray], spike = shuffleMixer_transforms.paired_random_crop([blur_img,sharp_img, sharp_img_gray],spike,self.crop_size,1)      
            [blur_img, sharp_img, sharp_img_gray, blur_img_gray, spike] = shuffleMixer_transforms.augment([blur_img, sharp_img, sharp_img_gray, blur_img_gray, spike],hflip=True,rotation=True)

        tfi = middleTFI(spike.numpy(), middle=self.length_spike//2)/0.5
        tfi = torch.from_numpy(tfi).unsqueeze(0).float()

        # tfi_L = middleTFI(spike.numpy(), middle=10)/0.5
        # tfi_L = torch.from_numpy(tfi_L).unsqueeze(0).unsqueeze(0).float()
        # tfi_R = middleTFI(spike.numpy(), middle=self.length_spike-10)/0.5
        # tfi_R = torch.from_numpy(tfi_R).unsqueeze(0).unsqueeze(0).float()
        # tfi_L = torch.cat([tfi_L, tfi_L, tfi_L], dim=1)
        # tfi_L = 2 * tfi_L - 1
        # tfi_R = torch.cat([tfi_R, tfi_R, tfi_R], dim=1)
        # tfi_R = 2 * tfi_R - 1

        # list_of_flows_forward = self.raft_img(tfi_L,tfi_R)
        # flow = list_of_flows_forward[-1].squeeze(0).detach() 

        return {
            # 't': torch.tensor(t).float(),
            'blur': blur_img,
            'gt': sharp_img,
            "gt_gray": sharp_img_gray,
            # "blur_gray": blur_img_gray,
            'spike': spike,
            'tfi': tfi,
            # "flow": flow,
            "img_path": "__".join(str(blurry_rgb_path).split('/')[-4:]) # rename to sole name for compatibility
        }

    def __len__(self):
        return len(self.blurry_data_list)//self.reduce_scale # 4408*257 for train
