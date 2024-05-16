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
    'train': {'H': 720, 'W': 1280, 'Hs': 360, 'Ws': 640},
    'val': {'H': 720, 'W': 1280, 'Hs': 360, 'Ws': 640}
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

    def _make_datalist(self):
        data_list = []

        tmp = "train" if self.seq_path=="train" else "test"
        rgb_root = os.path.join(self.dataset_root,"GOPRO_Large",tmp)
        rgb_subroots = os.listdir(rgb_root) # GOPR0372_07_00,...
        rgb_subroots.remove(".DS_Store")

        for rgb_subroot in rgb_subroots:
            fullpath = os.path.join(rgb_root, rgb_subroot, "blur_gamma")
            rgb_folders = os.listdir(fullpath) # occ008.320_f2881, ...
            rgb_folders = [os.path.join(fullpath, folder) for folder in rgb_folders]
            data_list.extend(rgb_folders) #.png

        return data_list

    def __getitem__(self, index):
        index *= self.reduce_scale
        t=0

        blurry_rgb_path = self.blurry_data_list[index]
        sharp_rgb_path = blurry_rgb_path.replace("blur_gamma", "sharp")
        spike_path = blurry_rgb_path.replace("GOPRO_Large", "GOPRO_Large_spike_seq").replace("blur_gamma", "spike")
        spike_path = spike_path[:-3]+"dat"

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
        spike = load_vidar_dat(spike_path, width=self.Ws, height=self.Hs)
        sslen = spike.shape[0]
        spike = spike[sslen//2-28:sslen//2+28,:,:]
        spike = torch.from_numpy(spike).float()
        
        if self.transforms:
            [blur_img, sharp_img, sharp_img_gray, blur_img_gray], spike = shuffleMixer_transforms.paired_random_crop([blur_img, sharp_img, sharp_img_gray, blur_img_gray],spike,self.crop_size,2)      
            # [blur_img, sharp_img, sharp_img_gray], spike = shuffleMixer_transforms.paired_random_crop([blur_img,sharp_img, sharp_img_gray],spike,self.crop_size,1)      
            [blur_img, sharp_img, sharp_img_gray, blur_img_gray, spike] = shuffleMixer_transforms.augment([blur_img, sharp_img, sharp_img_gray, blur_img_gray, spike],hflip=True,rotation=True)

        tfi = middleTFI(spike.numpy(), middle=self.length_spike//2)/0.5
        tfi = torch.from_numpy(tfi).unsqueeze(0).float()

        return {
            't': torch.tensor(t).float(),
            'blur': blur_img,
            'gt': sharp_img,
            "gt_gray": sharp_img_gray,
            "blur_gray": blur_img_gray,
            'spike': spike,
            'tfi': tfi,
            "img_path": "__".join(str(blurry_rgb_path).split('/')[-4:]) # rename to sole name for compatibility
        }

    def __len__(self):
        return len(self.blurry_data_list)//self.reduce_scale # 4408*257 for train
