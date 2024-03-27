import os
import sys
import types
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import argparse
import datetime
import time
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from torch.utils.data import RandomSampler
from mmcv.runner import LogBuffer
from copy import deepcopy

from diffusion import IDDPM
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, CheckpointConfig, RunConfig
import ray
import os
from argparse import ArgumentParser, Namespace
from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union

import torch
import wandb
# from composer.devices import DeviceGPU
from composer.utils import dist
from diffusers import AutoencoderKL
from PIL import Image
from streaming import MDSWriter, Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusion.data.mosaic import build_streaming_laion_dataloader

# from diffusion.datasets.laion.transforms import LargestCenterSquare

import numpy as np
import pyarrow 
import tempfile
import streaming
import webdataset as wds
import torchvision
import sys

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'T2IDiTBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)



def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--n_gpu_workers', type=int, default=4)
    parser.add_argument('--use_dataloader', type=bool, default=True)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # config = read_config(args.config)
    # config["args"]= args
    ray.init("auto")
    os.umask(0) 

    @ray.remote(num_cpus=1)
    def download_dataset(captions, shard, args):
        import s3fs
        import pyarrow 
        imgs = []
        os.umask(0) 
        # os.umask(0o000)  # file permission: 666; dir permission: 777
        s3 = s3fs.S3FileSystem(#anon=True, 
        #                        use_listings_cache=False,
                            key='****************',
        #                        key='*****************',
                            secret='****************',
        #                        secret='*****************',
                            endpoint_url='ENTER_URL_HERE', version_aware=True)
        # captions = s3.find('akash-thumper-v1-t5-data')
        # captions= np.array(captions)
        # captions= [captions[i] for i in range(20)]s
        if not os.path.exists("/home/ray/ray_results"):
            os.makedirs('/home/ray/ray_results/', int('777', base=8) ,exist_ok=True)
        else:
            os.chmod('/home/ray/ray_results/', int('777', base=8))


        if not os.path.exists("/home/ray/train_data/caption_feature_wmask/"):
            os.makedirs('/home/ray/train_data/caption_feature_wmask/', int('777', base=8) ,exist_ok=True)
        if not os.path.exists("/home/ray/train_data/images/"):
            os.makedirs('/home/ray/train_data/images/', int('777', base=8) , exist_ok=True )
        if not os.path.exists("/home/ray/train_data/partition/"):
            os.makedirs('/home/ray/train_data/partition/', int('777', base=8)  ,exist_ok=True)
        if not os.path.exists("/home/ray/train_data/wds_data2/"):
            os.makedirs('/home/ray/train_data/wds_data2/', int('777', base=8)  ,exist_ok=True)
          
        if not os.path.exists("/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"):
            os.makedirs('/home/ray/train_data/img_vae_feature/train_vae_256/noflip/', int('777', base=8)  ,exist_ok=True)  

        remote_upload_base="s3://akash-thumper-v1-mds"
        # remote_upload = os.path.join(args.remote_upload, str((args.bucket - 1) * 8 + dist.get_local_rank()))
        # remote_upload = os.path.join(remote_upload_base, str((args.bucket - 1) * 8 + 1))
        imgfilepath = captions[0].replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
        imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
        shardpath = imgfilepath_save.replace(os.path.basename(imgfilepath_save), "").replace("/home/ray/train_data/images/","").strip("/")
        remote_upload = os.path.join(remote_upload_base, shardpath)

        # dataset = torchvision.datasets.MNIST(root="./temp", download=True)
        # sink = wds.ShardWriter( pattern="wds_0", maxcount=1000,)
        # for index, (input, output) in enumerate(dataset):
        #     if index%1000==0:
        #         print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        #     sink.write({
        #         "__key__": "sample%06d" % index,
        #         "input.pyd": input,
        #         "output.pyd": output,
        #     })
        # sink.close()

        def upload_shard(fname):
            os.system(f"b2 upload_file akash-thumper-v1-wds2 {fname} {os.path.basename(fname)}")  # replace with your preferred command
            os.unlink(fname)

        # with wds.ShardWriter("wds_at1-%d.tar", maxsize=1000, post=upload_shard) as writer:
        #     # write data to the ShardWriter
        #     for index, (input, output) in enumerate(dataset):
        #         if index%1000==0:
        #             print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        #         writer.write({
        #             "__key__": "sample%06d" % index,
        #             "input.pyd": input,
        #             "output.pyd": output,
        #         })
        if args.use_dataloader:
            streaming.base.util.clean_stale_shared_memory()
            train_dataloader = build_streaming_laion_dataloader( remote="s3://akash-thumper-v1-mds2",
                                                                # local =f'/tmp/cache171{str(os.environ["RANK"])}',
                                                                local =f'/tmp/cache17401',
                                                                # local="/home/ray/train_data/mds2",
                                                                batch_size=1,
                                                                resize_size=256, 
                                                                # num_samples=200,
                                                                # predownload=100000, 
                                                                download_retry= 3, download_timeout=300,
                                                                drop_last=True, shuffle=True, num_canonical_nodes=1,
                                                                persistent_workers=False, num_workers=0, pin_memory=False, #prefetch_factor=2,
                                                                )
            # with wds.TarWriter(f"/home/ray/train_data/wds_data/wds_at1-{os.path.basename(shard).strip('/')}-" + "-%d.tar", maxsize=50000, post=upload_shard) as writer:
            # with wds.ShardWriter(f"/home/ray/train_data/wds_data/wds_at1-{os.path.basename(shard).strip('/')}-" + "-%d.tar", maxsize=50000, post=upload_shard) as writer:
            with wds.ShardWriter(f"/home/ray/train_data/wds_data2/wds_at1" + "-%d.tar", post=upload_shard) as writer:

                for index, (batch) in enumerate(train_dataloader):
                    if index%1000==0:
                        print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
                    writer.write({
                        "__key__": "sample%06d" % index,
                        'vae_features.pyd':   batch["vae"],
                        'txt_features.pyd':   batch["txt_features"].squeeze().reshape(-1, 1, 120, 4096),
                        'attention_mask.pyd': batch["attention_mask"].squeeze().reshape(-1, 1, 1, 120),
                        'metadata.pyd' : {'imgfilepath': imgfilepath,  'width': 256, 'height': 256}
                    })
        else:
            with wds.ShardWriter(f"/home/ray/train_data/wds_data/wds_at1-{os.path.basename(shard)}-" + "-%d.tar", maxsize=50000, post=upload_shard) as writer:
                for i, cap in enumerate(captions):
                    imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
                    imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
                    vaefilepath = f"akash-thumper-v1-vae-data/{os.path.basename(cap)}".replace(".npz", ".npy")
                    vaefilepath_save = f"/home/ray/train_data/img_vae_feature/train_vae_256/noflip/{os.path.basename(vaefilepath)}"
                    
                    shardpath = imgfilepath_save.replace(os.path.basename(imgfilepath_save), "").replace("/home/ray/train_data/images/","").strip("/")
                    remote_upload = os.path.join(remote_upload_base, shardpath)
                    if s3.exists(cap) and s3.exists(vaefilepath):
                        try:
                            s3.get(cap, f"/home/ray/train_data/caption_feature_wmask/")
                            # s3.get(imgfilepath,  imgfilepath_save) #f"/home/ray/train_data/images/"
                            print(f"vae {vaefilepath} {vaefilepath_save}")
                            s3.get(vaefilepath,  vaefilepath_save)    
                            vae_features = np.load(vaefilepath_save)
                            caption_path= f"/home/ray/train_data/caption_feature_wmask/{os.path.basename(cap)}"
                            npz_info = np.load(caption_path)
                            txt_fea = npz_info['caption_feature']
                            attention_mask = npz_info['attention_mask']

                            mds_sample = {
                                "__key__": "sample%06d" % i,
                                'vae_features.pyd':  torch.from_numpy(vae_features),
                                'txt_features.pyd':  torch.from_numpy(txt_fea),
                                'attention_mask.pyd':  torch.from_numpy(attention_mask),
                                'metadata.pyd' : {'imgfilepath': imgfilepath,  'width': 256, 'height': 256}
                        
                            }
                            writer.write(mds_sample)

                            if i % 10:
                                print(f"{i} of {len(captions)}")
                
                        except Exception as e:
                            print(e)


    import s3fs
    import pyarrow , more_itertools
    imgs = []
    os.umask(0) 
    # os.umask(0o000)  # file permission: 666; dir permission: 777
    s3 = s3fs.S3FileSystem(#anon=True, 
    #                        use_listings_cache=False,
                        key='****************',
    #                        key='*****************',
                        secret='****************',
    #                        secret='*****************',
                        endpoint_url='ENTER_URL_HERE', version_aware=True)
    captions = s3.find('akash-thumper-v1-t5-data')
    shards={}
    for cap in captions:
        imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
        imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
        shardpath = imgfilepath.replace(os.path.basename(imgfilepath), "").replace("akash-thumper-v1-t5-data","").strip("/")
        if shardpath not in shards.keys():
            shards[shardpath] = [cap]
        else:
            shards[shardpath].append(cap)

    [print(k, len(v)) for k,v in shards.items()]
    
    # dir_chunks =list(more_itertools.divide(120,captions ))
    # dir_chunks = [list(achunk) for achunk in  dir_chunks]
    # print("dir chunk", len(dir_chunks))
    # print("dir chunk", len(dir_chunks))
    # workers = [download_dataset.remote(captions=v, shard=k, args=args) for k,v in enumerate(dir_chunks)]
    workers = [download_dataset.remote(captions=captions, shard="0", args=args)]

    # workers = [download_dataset.remote(captions=v, shard=k, args=args) for k,v in shards.items()]
    ray.get(workers)
    # [4] Configure scaling and resource requirements.
