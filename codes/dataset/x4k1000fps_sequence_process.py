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

    def _make_datalist(self):
        data_list = []

        rgb_root = os.path.join(self.dataset_root, self.seq_path+'_fullRGB')
        rgb_subroots = os.listdir(rgb_root) # 002, 003, ...

        for rgb_subroot in rgb_subroots:
            fullpath = os.path.join(rgb_root, rgb_subroot)
            rgb_folders = os.listdir(fullpath) # occ008.320_f2881, ...
            rgb_folders = [os.path.join(rgb_subroot, folder) for folder in rgb_folders]
            data_list.extend(rgb_folders)

        return data_list

    def _getblurimage(self, datafolder):
        # datafolder: 002/occ008.320_f2881/0000_0000.png, ...
        # return: blur image
        blurimg = []
        for fname in datafolder:
            img = Image.open(fname).convert('RGB')
            img = transforms.ToTensor()(img)
            blurimg.append(img)
        blurimg = torch.stack(blurimg, dim=0)
        return torch.mean(blurimg, dim=0)

    def baocun_spike(self, in_path, H, W, num_frames, spike_seq_dir):
        sz = os.path.getsize(in_path)
        frame_sz = H * W // 8 #每一帧占多少字节
        data_len = sz // frame_sz #有多少帧
        frame_b = [] #按帧保存
        with open(in_path, 'rb') as fu:
            for i in range(data_len):
                a = fu.read(frame_sz)
                frame_b.append(a) 
        # print("len(frame_b)  ",len(frame_b))
        for gt_center in range(self.length_spike//2, num_frames - self.length_spike//2, 15):
            left_idx = gt_center - self.length_spike//2 # 闭区间
            right_idx = gt_center + self.length_spike//2 # 闭区间
            assert left_idx >= 0 and right_idx < num_frames, 'ERROR: img index out of range.'
            data = frame_b[left_idx:right_idx+1]
            spike_save_dir = os.path.join(spike_seq_dir,os.path.split(self.rgblist[gt_center])[-1])
            spike_save_dir = spike_save_dir[:-3]+"dat"
            with open(spike_save_dir, "wb") as fr:
                for j in range(len(data)):
                    fr.write(data[j])
            print("Saved to {}".format(spike_save_dir))

    def __getitem__(self, index):        
        blur_dir = os.path.join(self.dataset_root, self.seq_path+"_blurry_{}".format(self.length_spike), self.data_list[index])
        misc_utils.mkdirs(blur_dir)
        # if os.path.exists(blur_dir):
        #     shutil.rmtree(blur_dir)

        rgbfolder = os.path.join(self.dataset_root, self.seq_path+'_fullRGB', self.data_list[index])
        rgblist = sorted(os.listdir(rgbfolder))
        rgblist = [os.path.join(rgbfolder, fname) for fname in rgblist] # 002/occ008.320_f2881/0000_0000.png, ...
        num_frames = len(rgblist) # 257

        for gt_center in range(self.length_spike//2, num_frames - self.length_spike//2, 15):
            left_idx = gt_center - self.length_spike//2 # 闭区间
            right_idx = gt_center + self.length_spike//2 # 闭区间
            assert left_idx >= 0 and right_idx < num_frames, 'ERROR: img index out of range.'
            blurimg = self._getblurimage(rgblist[left_idx:right_idx+1])
            blurimg_dir = os.path.join(blur_dir,os.path.split(rgblist[gt_center])[-1])
            misc_utils.save_img(misc_utils.tensor2img(blurimg),blurimg_dir)
            print("Saved to {}".format(blurimg_dir))


        # spike_seq_dir = os.path.join(self.dataset_root, self.seq_path+"_spike_2xds_{}".format(self.length_spike), self.data_list[index])
        # misc_utils.mkdirs(spike_seq_dir)
        # # if os.path.exists(blur_dir):
        # #     shutil.rmtree(blur_dir)

        # rgbfolder = os.path.join(self.dataset_root, self.seq_path+'_fullRGB', self.data_list[index])
        # self.rgblist = sorted(os.listdir(rgbfolder))
        # self.rgblist = [os.path.join(rgbfolder, fname) for fname in self.rgblist] # 002/occ008.320_f2881/0000_0000.png, ...
        # num_frames = len(self.rgblist) # 257

        # spike_path = os.path.join(self.dataset_root, self.seq_path+'_spike_2xds', self.data_list[index], 'spike.dat')
        # self.baocun_spike(in_path=spike_path,
        #                   H=self.Hs,
        #                   W=self.Ws,
        #                   num_frames=num_frames,
        #                   spike_seq_dir=spike_seq_dir)

        return torch.ones(3,3,3)
    
    # def __getitem__(self, index):
    #     index *= self.reduce_scale

    #     # get rgb frames path
    #     rgbfolder = os.path.join(self.dataset_root, self.seq_path+'_fullRGB', self.data_list[index])
    #     rgblist = sorted(os.listdir(rgbfolder))
    #     rgblist = [os.path.join(rgbfolder, fname) for fname in rgblist] # 002/occ008.320_f2881/0000_0000.png, ...
    #     num_frames = len(rgblist) # 257

    #     # get spike path
    #     spikepath = os.path.join(self.dataset_root, self.seq_path+'_spike_2xds', self.data_list[index], 'spike.dat')
        
    #     # random select a center position #权宜之计。测试集需要固定才有意义。
    #     if self.phase=="train":
    #         gt_center = random.randint(self.length_spike//2, num_frames - self.length_spike//2 - 1) # randint 闭区间
    #     elif self.phase=="val":
    #         gt_center = num_frames//2
    #     left_idx = gt_center - self.length_spike//2 # 闭区间
    #     right_idx = gt_center + self.length_spike//2 # 闭区间
    #     assert left_idx >= 0 and right_idx < num_frames, 'ERROR: img index out of range.'

    #     # get sharp image and blur image
    #     blurimg = self._getblurimage(rgblist[left_idx:right_idx+1]) # 慢
    #     sharpimg = Image.open(rgblist[gt_center]).convert('RGB')
    #     sharpimg = transforms.ToTensor()(sharpimg)

    #     # get spike #257
    #     spike = load_vidar_dat(spikepath, width=self.Ws, height=self.Hs)[left_idx:right_idx+1, ...]
    #     spike = torch.from_numpy(spike).float()

    #     spike = torch.mean(spike,0).unsqueeze(0) #权宜之计
    #     spike = torch.nn.functional.interpolate(spike.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0)

    #     if self.transforms:
    #         sharpimg, blurimg, spike = self.transforms([sharpimg, blurimg, spike])        

    #     return {
    #         'gt': sharpimg,
    #         'blur': blurimg,
    #         'spike': spike,
    #         "img_path": "__".join(str(rgblist[gt_center]).split('/')[-4:]) # rename to sole name for compatibility
    #     }

    def __len__(self):
        return len(self.data_list)//self.reduce_scale # 4408 for train
    
if __name__ == "__main__":
    # save blur to local
    dataset = Sequence(root_path='/home/data/jyzhang/XVFI',
                       seq_path="train",
                       length_spike=33,
                       phase="train",
                       transforms=None,
                       reduce_scale=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8,
                                            pin_memory=True)
    idx=0
    for i in dataloader:
        idx+=1
