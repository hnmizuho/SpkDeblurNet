import os

import cv2
# import h5py
# import hdf5plugin
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

# os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

from pathlib import Path
from .gopro_sequence import Sequence

ALL_SEQUENCE_NAME = ['train', 'val', 'test'] # 文件夹前缀
TRAIN_SET_INDEX = [
    0
]
VAL_SET_INDEX = [
    1
]


class GoPro_DatasetProvider:
    def __init__(self, opt, train_transforms=None, val_transforms=None, phase="train"):
        dataset_root = Path(opt["dataset_root"])

        self.reduce_scale = opt["reduce_scale"]

        assert dataset_root.is_dir(), str(dataset_root)
        
        self.train_namelist = list()
        self.val_namelist = list()

        self.dataset_root = dataset_root
        self.length_spike = opt["length_spike"]
        self.crop_size = opt["crop_size"]

        self.phase = phase
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        for i in TRAIN_SET_INDEX:
            self.train_namelist.append(ALL_SEQUENCE_NAME[i])

        for i in VAL_SET_INDEX:
            self.val_namelist.append(ALL_SEQUENCE_NAME[i])

    def get_train_dataset(self):
        train_sequences = list()
        for child in self.train_namelist:
            train_sequences.append(Sequence(self.dataset_root, child, self.length_spike, self.crop_size, self.phase, self.train_transforms, reduce_scale=self.reduce_scale))

        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        return self.train_dataset

    def get_val_dataset(self):
        val_sequences = list()
        for child in self.val_namelist:
            val_sequences.append(Sequence(self.dataset_root, child, self.length_spike, self.crop_size, self.phase, self.val_transforms, reduce_scale=self.reduce_scale))

        self.val_dataset = torch.utils.data.ConcatDataset(val_sequences)
        return self.val_dataset