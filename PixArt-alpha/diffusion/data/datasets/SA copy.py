import os
import random
import time

import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
# from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor

from diffusion.data.builder import get_data_path, DATASETS
from pathlib import Path

@DATASETS.register_module()
class SAM(Dataset):
    def __init__(self,
                 root,
                 image_list_txt='part0.txt',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 mask_ratio=0.0,
                 mask_type='null',
                 **kwargs):
        
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = False #load_vae_feat
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.resolution = resolution
        self.img_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        image_list_txt = image_list_txt if isinstance(image_list_txt, list) else [image_list_txt]
        # if image_list_txt == 'all' or image_list_txt == 'part0.txt':
            # image_list_txts = os.listdir(os.path.join(self.root, 'partition'))
        # self.txt_feat_samples = os.listdir(os.path.join(self.root, 'caption_feature_wmask'))
        # for feat in self.txt_feat_samples:
        #     self.img_samples.append(feat.replace(".npz",".jpg").replace("caption_feature_wmask", 'images'))
        
        # self.txt_feat_samples = list(Path(os.path.join(self.root, 'caption_feature_wmask')).rglob("*.npz"))
        # self.img_samples = list(Path(os.path.join(self.root, 'images')).rglob("*.jpg"))
        self.txt_feat_samples = list(Path('/home/ray/train_data/caption_feature_wmask').rglob("*.npz"))
        # self.img_samples = list(Path( '/home/ray/train_data/images').rglob("*.jpg"))
        # self.vae_feat_samples = list(Path( '/home/ray/train_data/img_vae_feature/train_vae_256/noflip').rglob("*.npy"))
        self.txt_feat_samples = [str(file) for file in self.txt_feat_samples  ]
        self.img_samples = [f"/home/ray/train_data/images/{os.path.basename(file)}".replace(".npz", ".jpg") for file in self.txt_feat_samples  ]
        self.vae_feat_samples = [f"/home/ray/train_data/img_vae_feature/train_vae_256/noflip/{os.path.basename(file)}".replace(".npz", ".npy") for file in self.txt_feat_samples  ]


        print(f"found txt_feat_samples", self.txt_feat_samples)
        print(f"found img_samples",  self.img_samples)
        print(f"found vae_samples",  self.vae_feat_samples)

            # for txt in image_list_txts:
            #     image_list = os.path.join(self.root, 'partition', txt)
            #     with open(image_list, 'r') as f:
            #         lines = [line.strip() for line in f.readlines()]
            #         self.img_samples.extend([os.path.join(self.root, 'images', i+'.jpg') for i in lines])
            #         self.txt_feat_samples.extend([os.path.join(self.root, 'caption_feature_wmask', i+'.npz') for i in lines])
        # elif isinstance(image_list_txt, list):
        #     for txt in image_list_txt:
        #         image_list = os.path.join(self.root, 'partition', txt)
        #         with open(image_list, 'r') as f:
        #             lines = [line.strip() for line in f.readlines()]
        #             self.img_samples.extend([os.path.join(self.root, 'images', i + '.jpg') for i in lines])
        #             self.txt_feat_samples.extend([os.path.join(self.root, 'caption_feature_wmask', i + '.npz') for i in lines])
        #             self.vae_feat_samples.extend([os.path.join(self.root, 'img_vae_feature/train_vae_256/noflip', i + '.npy') for i in lines])

        self.ori_imgs_nums = len(self)
        # self.img_samples = self.img_samples[:10000]
        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

    def getdata(self, idx):
        img_path = self.img_samples[idx]
        npz_path = self.txt_feat_samples[idx]
        npy_path = self.vae_feat_samples[idx]
        data_info = {'img_hw': torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
                     'aspect_ratio': torch.tensor(1.)}

        if self.load_vae_feat:
            img = self.loader(npy_path)
        else:
            img = self.loader(img_path)
        npz_info = np.load(npz_path)
        txt_fea = torch.from_numpy(npz_info['caption_feature'])
        attention_mask = torch.ones(1, 1, txt_fea.shape[1])
        if 'attention_mask' in npz_info.keys():
            attention_mask = torch.from_numpy(npz_info['attention_mask'])[None]

        if self.transform:
            img = self.transform(img)

        data_info["mask_type"] = self.mask_type

        return img, txt_fea, attention_mask, data_info
    
    # def __getitem__(self, idx):
    #     try:
    #         data = self.getdata(idx)
    #         return data
    #     except Exception as e:
    #         print(self.img_samples[idx], ' info is not correct', e)
    #         return None
    def __getitem__(self, idx):
        for i in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(self.img_samples[idx], ' info is not correct', e)
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path, allow_pickle=True)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample
        # return mean

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]
        self.txt_feat_samples = [self.txt_feat_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)
