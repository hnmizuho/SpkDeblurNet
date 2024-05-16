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
from utils.spike_utils import load_vidar_dat
from utils.flow_visualization import flow_visualization
import torchvision.transforms as transforms
from utils import misc_utils

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
        self.data_list = self._make_datalist() # 002/occ008.320_f2881, ...
        self.H = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['H']
        self.W = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['W']
        self.Hs = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['Hs']
        self.Ws = X4K10000FPS_RESOLUTION_DICT[self.seq_path]['Ws']

        self.transforms = transforms

        #TODO test phase

    def _make_datalist(self):
        data_list = []

        tmp = "train" if self.seq_path=="train" else "test"
        rgb_root = os.path.join(self.dataset_root,"GOPRO_Large",tmp)
        rgb_subroots = os.listdir(rgb_root) # GOPR0372_07_00,...
        rgb_subroots.remove(".DS_Store")

        for rgb_subroot in rgb_subroots:
            fullpath = os.path.join(rgb_root, rgb_subroot, "blur_gamma")
            data_list.append(fullpath) #.png

        return data_list

    def baocun_spike(self, in_path, H, W, num_frames, spike_seq_dir, len_to_avg, rgb_folders):
        sz = os.path.getsize(in_path)
        frame_sz = H * W // 8 #每一帧占多少字节
        data_len = sz // frame_sz #有多少帧
        frame_b = [] #按帧保存
        with open(in_path, 'rb') as fu:
            for i in range(data_len):
                a = fu.read(frame_sz)
                frame_b.append(a) 
        # for gt_center in range(self.length_spike//2, num_frames - self.length_spike//2, 15):
        #     left_idx = gt_center - self.length_spike//2 # 闭区间
        #     right_idx = gt_center + self.length_spike//2 # 闭区间
        #     assert left_idx >= 0 and right_idx < num_frames, 'ERROR: img index out of range.'
        #     data = frame_b[left_idx:right_idx+1]
        #     spike_save_dir = os.path.join(spike_seq_dir,os.path.split(self.rgblist[gt_center])[-1])
        #     spike_save_dir = spike_save_dir[:-3]+"dat"
        #     with open(spike_save_dir, "wb") as fr:
        #         for j in range(len(data)):
        #             fr.write(data[j])
        #     print("Saved to {}".format(spike_save_dir))
        for idx, name in enumerate(rgb_folders):
            left_idx = idx*len_to_avg*8
            right_idx = (idx+1)*len_to_avg*8
            data = frame_b[left_idx:right_idx]
            spike_save_dir = os.path.join(spike_seq_dir,name)
            spike_save_dir = spike_save_dir[:-3]+"dat"
            with open(spike_save_dir, "wb") as fr:
                for j in range(len(data)):
                    fr.write(data[j])
            print("Saved to {}".format(spike_save_dir))

    def __getitem__(self, index):        
        spike_seq_dir = self.data_list[index].replace("GOPRO_Large","GOPRO_Large_spike_seq").replace("blur_gamma","spike")
        len_to_avg = int(os.path.split(os.path.split(self.data_list[index])[0])[1].split("_")[1])

        misc_utils.mkdirs(spike_seq_dir)

        rgb_folders = os.listdir(self.data_list[index]) # 00047.png,...
        # rgb_folders = [os.path.join(self.data_list[index], folder) for folder in rgb_folders]
        rgb_folders = sorted(rgb_folders)
        num_frames = len(rgb_folders)

        spike_path = self.data_list[index].replace("GOPRO_Large","Spike_GOPRO_Large_all")
        spike_path = os.path.join(os.path.split(spike_path)[0],"spike.dat")
        self.baocun_spike(in_path=spike_path,
                          H=self.Hs,
                          W=self.Ws,
                          num_frames=num_frames,
                          spike_seq_dir=spike_seq_dir,
                          len_to_avg = len_to_avg,
                          rgb_folders = rgb_folders)

        return torch.ones(3,3,3)
    

    def __len__(self):
        return len(self.data_list)//self.reduce_scale # 4408 for train
    
