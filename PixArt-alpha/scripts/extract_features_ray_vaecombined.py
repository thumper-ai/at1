import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import time
import datetime
import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse

from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL
import ray 
import pandas as pd
import shutil
import more_itertools
import boto3

@ray.remote( num_cpus=3, num_gpus=1 , memory= 40* 1000 * 1024 * 1024) #10gb memory reserved 
class Actor:
    def __init__(self, args, captions):
        self.args = args
        self.captions = captions
        # self.df = df
        self.gpu_count = torch.cuda.device_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adirin = "" #sos.path.basename(adirin)        
        self.model_path_dir =args.model_path_dir
        self.save_path_root =args.save_path_root
        self.vae_save_root = f'{args.save_path_root}/img_vae_features_{args.image_resize}'
        os.umask(0o000)       # file permission: 666; dir permission: 777
        os.makedirs(self.vae_save_root, exist_ok=True)

        self.vae_save_dir = os.path.join(self.vae_save_root, 'noflip')
        os.makedirs(self.vae_save_dir, exist_ok=True)

        # os.mkdir("/home/ray/train_data/t5/", exists )s
        if not os.path.exists(self.model_path_dir ):
            print("downloading models")
            if not os.path.exists('/home/ray/models'):
                os.mkdir('/home/ray/models')
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")
          
        if not os.path.exists('/home/ray/models/PixArt-alpha/t5-v1_1-xxl/config.json'):
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            # os.system("cd /home/ray/models && git lfs pull https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")

        # self.t5 = T5Embedder(device=self.device, local_cache=True, cache_dir= self.model_path_dir , dir_or_name= self.model_path_dir  )
        # self.t5_save_root = f'{self.save_path_root}/t5_features'

        self.transform  = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(image_resize),  # Image.BICUBIC
            T.CenterCrop(image_resize),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])

        # self.vae = AutoencoderKL.from_pretrained(self.model_path_dir,  device=self.device, local_cache=True, cache_dir= self.model_path_dir  ).to(self.device)

        # self.t5 = T5Embedder(device=self.device, local_cache=True, cache_dir= self.model_path_dir , dir_or_name= self.model_path_dir  ).to(self.device)
        # self.t5_save_root = f'{self.save_path_root}t5_features'
        # t5_save_root = 'data/InternalData_2k/t5_features'

        # self.vae_save_dir = self.t5_save_root
        os.makedirs(self.vae_save_dir, exist_ok=True)
        # captions = set()
        # imgfiles = list(Path('/home/ray/train_data/').rglob("*.jpg"))
        # # [p.chmod(0o666) for p in imgfiles]
        # print(f"found {len(imgfiles)} imgs on init")

    def hasgpu(self):
        return self.gpu_count > 0
    
    def inference(self):
        print(self.args)

        if not os.path.exists("/home/ray/train_data/caption_feature_wmask/"):
            os.makedirs('/home/ray/train_data/caption_feature_wmask/', int('777', base=8) ,exist_ok=True)
        if not os.path.exists("/home/ray/train_data/images/"):
            os.makedirs('/home/ray/train_data/images/', int('777', base=8) , exist_ok=True )
        if not os.path.exists("/home/ray/train_data/partition/"):
            os.makedirs('/home/ray/train_data/partition/', int('777', base=8)  ,exist_ok=True)
        
        if not os.path.exists("/home/ray/train_data/combined_features/"):
            os.makedirs('/home/ray/train_data/combined_features/', int('777', base=8)  ,exist_ok=True)
        
        if not os.path.exists("/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"):
            os.makedirs('/home/ray/train_data/img_vae_feature/train_vae_256/noflip/', int('777', base=8)  ,exist_ok=True)   

        vae_dir = "/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"
        combined_dir = "/home/ray/train_data/combined_features/"


        client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
                                            aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
                                            aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
                                            region_name= os.environ.get("BUCKET_REGION", "auto"))
        # at1/LLaVA/llava/serve/cli_batch_ray_par_a.py
        # if not os.path.exists(adirin):
        #     os.mkdir(adirin)
        # s3 = boto3.resource('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL"))
        # bucket = s3.Bucket('akash-thumper-v1-vae')
        s3 = s3fs.S3FileSystem(#anon=True, 
        #                        use_listings_cache=False,
                            key='005d438bd2ef4c70000000002',
        #                        key='005d438bd2ef4c70000000003',
                            secret='K005fyvkZBOwFCUsl4z6af3xRgopy88',
        #                        secret='K005MjAaVuc3yWNDqy82XXHuVrFBP+8',
                            endpoint_url='https://s3.us-east-005.backblazeb2.com', version_aware=True)
        # captions = s3.find('akash-thumper-v1-t5-data')

        for a in range(len(self.captions)):
            cap = self.captions[a]
            imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
            imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
            image_name = os.path.basename(imgfilepath_save)

            vaefilepath = f"akash-thumper-v1-vae-data/{os.path.basename(cap)}".replace(".npz", ".npy")
            vaefilepath_save = f"/home/ray/train_data/img_vae_feature/train_vae_256/noflip/{os.path.basename(vaefilepath)}"

            if s3.exists(cap):
            # if s3.exists(cap) and os.path.exists( imgfilepath_save) :
                print(cap, imgfilepath, imgfilepath_save)
                # s3.get(imgfilepath,  imgfilepath_save)
                try:
                    s3.get(cap, f"/home/ray/train_data/caption_feature_wmask/")
                    npz_info = np.load(f"/home/ray/train_data/caption_feature_wmask/{os.path.basename(cap)}")
                    
                    if s3.exists(vaefilepath):
                        s3.get(vaefilepath,  vaefilepath_save)  
                        z = np.load(vaefilepath_save)
                        vae_found = True
                        save_path = os.path.join(combined_dir, image_name.replace('.jpg', '.npz'))
                        emb_dict = {
                            'caption_feature': npz_info["caption_feature"], # caption_emb.float().cpu().data.numpy(),
                            'attention_mask': npz_info["attention_mask"], # emb_mask.cpu().data.numpy(),
                            'z': z
                        }
                        np.savez_compressed(save_path, **emb_dict)
                        s3.put(save_path, "akash-thumper-v1-combined-data")
                    else:
                        vae_found = False
                        s3.get(imgfilepath,  imgfilepath_save)
                        img = Image.open(imgfilepath_save)
                        img = self.transform(img).to(self.device)[None]
                    # filepath= str(train_data['file_name'].values[a])
                    # image_name = os.path.basename(str(train_data['file_name'].values[a]))
                    save_path = os.path.join(combined_dir, image_name.replace('.jpg', '.npz'))
                    # print(f"trying {save_path}")
                    # if os.path.exists(save_path):
                    #     continue
                    # with torch.no_grad():
                        # # caption_emb, emb_mask = self.t5.get_text_embeddings(cap)
                        # if not vae_found:
                        #     posterior = self.vae.encode(img).latent_dist
                        #     z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
                        # emb_dict = {
                        #     'caption_feature': npz_info["caption_feature"], # caption_emb.float().cpu().data.numpy(),
                        #     'attention_mask': npz_info["attention_mask"], # emb_mask.cpu().data.numpy(),
                        #     'z': z
                        # }
                        # np.savez_compressed(save_path, **emb_dict)
                        # s3.put(save_path, "akash-thumper-v1-combined-data")

                except Exception as e:
                    print(e)
                    print(image_name)

                    
        os.system(f"b2 sync {combined_dir} b2://akash-thumper-v1-combined-data --skipNewer")
        return f"done {image_name}"
    


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_class', default='action', type=str)
    # parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--image_resize', default=256, type=int)
    parser.add_argument('--workers', default=2, type=int)

    # parser.add_argument('--dataset_name', default='at2s_test', type=str)
    # parser.add_argument('--base_dir', default='/mnt/sabrent', type=str)
    
    # parser.add_argument('--caption_filepath', default='/home/logan/thumperai/test5.csv', type=str)
    parser.add_argument('--vae_dir', default="/home/ray/models/PixArt-alpha/sd-vae-ft-ema", type=str)
    parser.add_argument('--model_path_dir', default="/home/ray/models/PixArt-alpha/sd-vae-ft-ema", type=str)
    parser.add_argument('--save_path_root', default="/home/ray/train_data/img_vae_feature", type=str)

    parser.add_argument('--t5_dir', default="/home/ray/models/PixArt-alpha/output/pretrained_models/t5-v1_1-xxl", type=str)
    # parser.add_argument('--img2img_dir', default="/mnt/sabrent/cc03", type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize =args.image_resize # 512
    print('Extracting Image Resolution %s' % image_resize)
    # dataset_name = args.dataset_name #"at1_test"
    # caption_filepath =  args.caption_filepath
    # base_dir = args.base_dir

    import s3fs
    import pyarrow 
    imgs = []
    os.umask(0) 
    # os.umask(0o000)  # file permission: 666; dir permission: 777
    s3 = s3fs.S3FileSystem(#anon=True, 
    #                        use_listings_cache=False,
                        key='005d438bd2ef4c70000000002',
    #                        key='005d438bd2ef4c70000000003',
                        secret='K005fyvkZBOwFCUsl4z6af3xRgopy88',
    #                        secret='K005MjAaVuc3yWNDqy82XXHuVrFBP+8',
                        endpoint_url='https://s3.us-east-005.backblazeb2.com', version_aware=True)
    captions = s3.find('akash-thumper-v1-t5-data')
    # captions= np.array(captions)
    # captions= [captions[i] for i in range(5000)]
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

    vae_dir = "/home/ray/train_data/img_vae_feature/train_vae_256/noflip/"
#     img_vae_feature/  (run tools/extract_img_vae_feature.py to generate image VAE features, same name as images except .npy extension)
# │  ├──train_vae_256/
# │  │  ├──noflip/

    for i, cap in enumerate(captions):
        imgfilepath = cap.replace("akash-thumper-v1-t5-data", "akash-thumper-v1-training-images").replace(".npz", ".jpg")
        imgfilepath_save = f"/home/ray/train_data/images/{os.path.basename(imgfilepath)}"
        if s3.exists(cap):
            print(cap, imgfilepath, imgfilepath_save)
            # s3.get(imgfilepath,  imgfilepath_save), #f"/home/ray/train_data/images/" )
            # if i % 10:
            #     print(f"{i} of {len(captions)}")
            imgs.append(imgfilepath_save)

    # indexlst= list(range(caption_df.shape[0]))
    dir_chunks =list(more_itertools.divide(args.workers, captions ))
    dir_chunks = [list(achunk) for achunk in  dir_chunks]
    # dir_chunks = [captions for a in range(args.workers)]

    print("dir chunk", dir_chunks, len(dir_chunks))

    ray.init(address="auto")

    actors = [Actor.remote(args=args, captions=captions) for captions in dir_chunks]
    jobs = []
    valid_actors =[]
    for actor in actors:
        if ray.get(actor.hasgpu.remote()) > 0:
            valid_actors.append(actor)


    print(f"{len(valid_actors)} of {len(actors)} validated for gpu detection")
    for i, actor in enumerate(valid_actors):
        jobs.append(actor.inference.remote())
    results = ray.get(jobs)

    # dir_chunks =list(chunked(dirlist, len(valid_actors)))

    # for adirin_list in dir_chunks:
    #     for i, actor in enumerate(valid_actors):
    #         if i < len(adirin_list):
    #             jobs.append(actor.inference.remote())
    # results = ray.get(jobs)

    print("result files:", results)


