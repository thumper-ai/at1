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

# from diffusion.datasets.laion.transforms import LargestCenterSquare

import numpy as np
import pyarrow 
import tempfile


import collections.abc
import functools
import json
import logging
import os
import random
import shutil
import tempfile
import urllib.parse
from collections import OrderedDict
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, List, Sequence, Tuple, Type, TypeVar, Union, cast, overload

import torch.distributed as dist

from streaming.base.constant import SHM_TO_CLEAN
from streaming.base.distributed import get_local_rank, maybe_init_dist
from streaming.base.format.index import get_index_basename
# from streaming.base.shared.prefix import _get_pathx

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
    def download_dataset(captions, shard):
        import s3fs
        import pyarrow 
        remote_upload_base="s3://akash-thumper-v1-mds2"
        # remote_upload = os.path.join(args.remote_upload, str((args.bucket - 1) * 8 + dist.get_local_rank()))
        # remote_upload = os.path.join(remote_upload_base, str((args.bucket - 1) * 8 + 1))
        imgfilepath = captions[0].replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
        imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
        shardpath = imgfilepath_save.replace(os.path.basename(imgfilepath_save), "").replace("/home/ray/train_data/images/","").strip("/")
        remote_upload = os.path.join(remote_upload_base, str(shard))
        columns = {
                'vae_features': "bytes",
                'txt_features': "bytes",
                'attention_mask': "bytes",
                'imgfilepath': 'str',

                # 'punsafe': 'float64',
                # 'pwatermark': 'float64',
                # 'similarity': 'float64',
                # 'caption': 'str',
                # 'url': 'str',
                # 'key': 'str',
                # 'status': 'str',
                # 'error_message': 'str',
                'width': 'int32',
                'height': 'int32',
                # 'original_width': 'int32',
                # 'original_height': 'int32',
                # 'exif': 'str',
                # # 'jpg': 'bytes',
                # 'hash': 'int64',
                # 'aesthetic_score': 'float64',
                # 'caption_latents': 'bytes',
                # 'latents_256': 'bytes',
                # 'latents_512': 'bytes',
            }       
        # print("found caps", captions)
       
        writer = MDSWriter(out=remote_upload,
                        columns=columns,
                        compression=None,
                        keep_local=True,
                        # hash=[],
                        size_limit=256 * (2**20),
                        max_workers=64)

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
          
        if not os.path.exists("/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"):
            os.makedirs('/home/ray/train_data/img_vae_feature/train_vae_256/noflip/', int('777', base=8)  ,exist_ok=True)  
        
        for i, cap in enumerate(captions):
            imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
            imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
            vaefilepath = f"akash-thumper-v1-vae-data/{os.path.basename(cap)}".replace(".npz", ".npy")
            vaefilepath_save = f"/home/ray/train_data/img_vae_feature/train_vae_256/noflip/{os.path.basename(vaefilepath)}"
            caption_path= f"/home/ray/train_data/caption_feature_wmask/{os.path.basename(cap)}"

            shardpath = imgfilepath_save.replace(os.path.basename(imgfilepath_save), "").replace("/home/ray/train_data/images/","").strip("/")
            remote_upload = os.path.join(remote_upload_base, shardpath)
            if (os.path.exists(caption_path) and os.path.exists(vaefilepath_save) )or ( s3.exists(cap) and s3.exists(vaefilepath)):
                try:
                    if not os.path.exists(caption_path):
                        s3.get(cap, f"/home/ray/train_data/caption_feature_wmask/")
                    # s3.get(imgfilepath,  imgfilepath_save) #f"/home/ray/train_data/images/"
                    print(f"vae {vaefilepath} {vaefilepath_save}")
                    if not os.path.exists(vaefilepath_save):
                        s3.get(vaefilepath,  vaefilepath_save)    
                    vae_features = np.load(vaefilepath_save).tobytes()

                    # vae_mean, vae_std = torch.from_numpy(np.load(vaefilepath_save).squeeze()).chunk(2)
                    # vae_mean = vae_mean.tobytes()
                    # vae_std = vae_std.tobytes()
                    npz_info = np.load(caption_path)
                    # print("txt_feat", npz_info['caption_feature'].shape)
                    # print("attention_mask", npz_info['attention_mask'].shape)

                    txt_fea = npz_info['caption_feature'].tobytes()
                    attention_mask = npz_info['attention_mask'].tobytes()
                    # attention_mask = torch.ones(1, 1, txt_fea.shape[1])
                    # if 'attention_mask' in npz_info.keys():
                    # #     attention_mask = torch.from_numpy(npz_info['attention_mask'])[None]

                    # latents_256_sample = latents_256[i].tobytes() if min(sample['width'][i],
                    #                                          sample['height'][i]) >= 256 else b''
                    # latents_512_sample = latents_512[i].tobytes() if min(sample['width'][i],
                    #                                                     sample['height'][i]) >= 512 else b''
                    mds_sample = {
                        'vae_features': vae_features ,
                        'txt_features': txt_fea,
                        'attention_mask': attention_mask,
                        'imgfilepath': imgfilepath,

                        # 'caption': sample['caption'][i],
                        # 'url': sample['url'][i],
                        # 'key': sample['key'][i],
                        # 'status': sample['status'][i],
                        # 'error_message': sample['error_message'][i],
                        'width': 256,
                        'height': 256,
                        # 'original_width': sample['original_width'][i],
                        # 'original_height': sample['original_height'][i],
                        # 'exif': sample['exif'][i],
                        # 'jpg': sample['jpg'][i],
                        # 'hash': sample['hash'][i],
                        # 'aesthetic_score': sample['aesthetic_score'][i],
                        # 'caption_latents': conditioning[i].tobytes(),
                        # 'latents_256': latents_256_sample,
                        # 'latents_512': latents_512_sample,
                    }
                    writer.write(mds_sample)

                    if i % 10:
                        print(f"{i} of {len(captions)}")
        
                except Exception as e:
                    print(e)
        writer.finish()


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
    # captions = s3.find('akash-thumper-v1-t5-data')
    shards={}
    # for cap in captions:
    #     imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
    #     imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
    #     shardpath = imgfilepath.replace(os.path.basename(imgfilepath), "").replace("akash-thumper-v1-t5-data","").strip("/")
    #     if shardpath not in shards.keys():
    #         shards[shardpath] = [cap]
    #     else:
    #         shards[shardpath].append(cap)

    # [print(k, len(v)) for k,v in shards.items()]
    
    # dir_chunks =list(more_itertools.divide(120,captions ))
    # dir_chunks = [list(achunk) for achunk in  dir_chunks]
    # print("dir chunk", len(dir_chunks))
    # print("dir chunk", len(dir_chunks))
    # captions = [v for k,v in shards.items()]
    # dir_chunks =list(more_itertools.divide(16,captions ))
    # dir_chunks = [list(achunk) for achunk in  dir_chunks]

    # workers = [download_dataset.remote(captions=captions, shard=a) for a, captions in enumerate(dir_chunks) ]
    # ray.get(workers)
    # [4] Configure scaling and resource requirements.

def _merge_index_from_list(index_file_urls: List[Union[str, Tuple[str, str]]],
                           out: Union[str, Tuple[str, str]],
                           keep_local: bool = True,
                           download_timeout: int = 60) -> None:
    """Merge index.json from a list of index files of MDS directories to create joined index.

    Args:
        index_file_urls (Union[str, Tuple[str,str]]): index.json from all the partitions
            each element can take the form of a single path string or a tuple string.

            The pattern of index_file_urls and corresponding reaction is one of:
            1. All URLS are str (local). All URLS are accessible locally -> no download
            2. All URLS are tuple (local, remote). All URLS are accessible locally -> no download
            3. All URLS are tuple (local, remote). Download URL that is not accessible locally
            4. All URLS are str (remote) -> download all

        out (Union[str, Tuple[str, str]]): path to put the merged index file
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``
        download_timeout (int): The allowed time for downloading each json file. Defaults to 60.
    """
    from streaming.base.storage.download import download_file
    from streaming.base.storage.upload import CloudUploader

    if not index_file_urls or not out:
        logger.warning('Either index_file_urls or out are None. ' +
                       'Need to specify both `index_file_urls` and `out`. ' + 'No index merged')
        return

    # This is the index json file name, e.g., it is index.json as of 0.6.0
    index_basename = get_index_basename()

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    # Remove duplicates, and strip '/' from right if any
    index_file_urls = list(OrderedDict.fromkeys(index_file_urls))
    urls = []
    for url in index_file_urls:
        if isinstance(url, str):
            urls.append(url.rstrip('/').strip())
        else:
            urls.append((url[0].rstrip('/').strip(), url[1].rstrip('/').strip()))

    # Prepare a temp folder to download index.json from remote if necessary. Removed in the end.
    with tempfile.TemporaryDirectory() as temp_root:
        logging.warning(f'A temporary folder {temp_root} is created to store index files')

        # Copy files to a temporary directory. Download if necessary
        partitions = []
        for url in urls:
            if isinstance(url, tuple):
                src = url[0] if os.path.exists(url[0]) else url[1]
            else:
                src = url

            obj = urllib.parse.urlparse(src)
            scheme, bucket, path = obj.scheme, obj.netloc, obj.path
            if scheme == '' and bucket == '' and path == '':
                raise FileNotFoundError(
                    f'Check data availability! local index {url[0]} is not accessible.' +
                    f'remote index {url[1]} does not have a valid url format')
            dest = os.path.join(temp_root, path.lstrip('/'))

            try:
                download_file(src, dest, download_timeout)
            except Exception as ex:
                raise RuntimeError(f'Failed to download index.json: {src} to {dest}') from ex

            if not os.path.exists(dest):
                raise FileNotFoundError(f'Index file {dest} does not exist or not accessible.')

            partitions.append(dest)

        # merge shards from all index files
        shards = []
        for partition_index in partitions:
            p = Path(partition_index)
            obj = json.load(open(partition_index))
            for i in range(len(obj['shards'])):
                shard = obj['shards'][i]
                for key in ('raw_data', 'zip_data', 'raw_meta', 'zip_meta'):
                    if shard.get(key):
                        basename = shard[key]['basename']
                        obj['shards'][i][key]['basename'] = os.path.join(
                            os.path.basename(p.parent), basename)
            shards += obj['shards']

        # Save merged index locally
        obj = {
            'version': 2,
            'shards': shards,
        }
        merged_index_path = os.path.join(temp_root, index_basename)
        with open(merged_index_path, 'w') as outfile:
            json.dump(obj, outfile)

        # Move merged index from temp path to local part in out
        # Upload merged index to remote if out has remote part
        shutil.move(merged_index_path, cu.local)
        if cu.remote is not None:
            cu.upload_file(index_basename)

        # Clean up
        if not keep_local:
            shutil.rmtree(cu.local, ignore_errors=True)

def _merge_index_from_root(out: Union[str, Tuple[str, str]],
                           keep_local: bool = True,
                           download_timeout: int = 60) -> None:
    """Merge index.json given the root of MDS dataset. Write merged index to the root folder.

    Args:
        out (Union[str, Tuple[str,str]]): folder that contain MDS partitions.
            :A local directory, merge index happens locally
            :A remote directory, download all the sub-directories index.json in a temporary
                sub-directories, merge locally, and then upload it to out location
            :A (local_dir, remote_dir), check if sub-directories index.json file present locally
                If yes, then merge locally and upload to remote_dir .
                If not, download all the sub-directories index.json from remote to local,
                merge locally, and upload to remote_dir .
        keep_local (bool): Keep local copy of the merged index file. Defaults to ``True``
        download_timeout (int): The allowed time for downloading each json file. Defaults to 60.
    """
    from streaming.base.storage.upload import CloudUploader

    def not_merged_index(index_file_path: str, out: str):
        """Check if index_file_path is the merged index at folder out.

        Args:
            index_file_path (str): the path to index.json file
            out (str): remote or local url of a folder
        Return:
            (bool): no if index.json sits in out instead of in the subfolders of out
        """
        prefix = str(urllib.parse.urlparse(out).path)
        return os.path.dirname(index_file_path).strip('/') != prefix.strip('/')

    if not out:
        logger.warning('No MDS dataset folder specified, no index merged')
        return

    cu = CloudUploader.get(out, exist_ok=True, keep_local=True)

    local_index_files = []
    cl = CloudUploader.get(cu.local, exist_ok=True, keep_local=True)
    for file in cl.list_objects():
        if file.endswith('.json') and not_merged_index(file, cu.local):
            local_index_files.append(file)

    if cu.remote:
        obj = urllib.parse.urlparse(cu.remote)
        remote_index_files = []
        for file in cu.list_objects():
            if file.endswith(get_index_basename()) and not_merged_index(file, cu.remote):
                join_char = '//'
                if obj.scheme == 'dbfs':
                    path = Path(cu.remote)
                    prefix = os.path.join(path.parts[0], path.parts[1])
                    if prefix == 'dbfs:/Volumes':
                        join_char = '/'
                remote_index_files.append(obj.scheme + join_char + os.path.join(obj.netloc, file))
        if len(local_index_files) == len(remote_index_files):
            _merge_index_from_list(list(zip(local_index_files, remote_index_files)),
                                   out,
                                   keep_local=keep_local,
                                   download_timeout=download_timeout)
        else:
            _merge_index_from_list(remote_index_files,
                                   out,
                                   keep_local=keep_local,
                                   download_timeout=download_timeout)
        return

    _merge_index_from_list(local_index_files,
                           out,
                           keep_local=keep_local,
                           download_timeout=download_timeout)


    os.umask(0) 
    mds = s3.find('s3://akash-thumper-v1-mds2/')
    indexfiles=[]
    for f in mds:
        if ".json" in f:
            # indexfiles.append((f"s3://{f}", f'/tmp/{f.replace("akash-thumper-v1-mds2/","")}'))
            indexfiles.append((f"s3://{f}", f'/home/ray/train_data/mds2/{f.replace("akash-thumper-v1-mds2/","")}'))
            # indexfiles.append(f's3://{f.replace("index.json","")}')
    print("indexfiles", indexfiles)
    # os.system("b2 sync b2://akash-thumper-v1-mds2/ /home/ray/train_data/mds2 --skipNewer")
    from streaming.base.util import merge_index
    from pathlib import Path
    # merge_index( out="s3://akash-thumper-v1-mds2/")
    # merge_index( index_file_urls="/home/ray/train_data/mds2/" keep_local=True)
    # merge_index(index_file_urls=indexfiles, out="s3://akash-thumper-v1-mds2/")
    partitions = list(Path("/home/ray/train/data/mds2").rglob("*.json"))
    merged_index_path="/home/ray/train/data/mds2/index.json"
    shards = []
    for partition_index in partitions:
        p = Path(partition_index)
        obj = json.load(open(partition_index))
        for i in range(len(obj['shards'])):
            shard = obj['shards'][i]
            for key in ('raw_data', 'zip_data', 'raw_meta', 'zip_meta'):
                if shard.get(key):
                    basename = shard[key]['basename']
                    obj['shards'][i][key]['basename'] = os.path.join(
                        os.path.basename(p.parent), basename)
        shards += obj['shards']

    # Save merged index locally
    obj = {
        'version': 2,
        'shards': shards,
    }
    # merged_index_path = os.path.join(temp_root, index_basename)
    with open(merged_index_path, 'w') as outfile:
        json.dump(obj, outfile)


    # merge_index( index_file_urls=indexfiles,  out="s3://akash-thumper-v1-mds2/")
    # merge_index( index_file_urls="/home/ray/train_data/mds2/",  keep_local=True)
    # _merge_index_from_root(index_file_urls="/home/ray/train_data/mds2/", out="/home/ray/train_data/mds2/")
    os.system("b2 cp /home/ray/train_data/mds2/index.json /home/ray/train_data/mds2/ --skipNewer")

    # _merge_index_from_root(out="s3://akash-thumper-v1-mds2/")