import os
from pathlib import Path
# import sys
# current_file_path = Path(__file__).resolve()
# sys.path.insert(0, str(current_file_path.parent.parent))
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
import stat
from huggingface_hub import snapshot_download

# @ray.remote
def extract_caption_t5(fp, save_path_root, model_path_dir):
    # t5 = T5Embedder(device="cuda", local_cache=True, cache_dir='data/t5_ckpts')
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=model_path_dir, dir_or_name=model_path_dir )
    t5_save_root = f'{save_path_root}/t5_features'
    # t5_save_root = 'data/InternalData_2k/t5_features'

    t5_save_dir = t5_save_root
    try:
        os.makedirs(t5_save_dir, exist_ok=True)
    except Exception as e:
        print(e)
    captions = set()
    if ".csv" in fp:
        train_data = pd.read_csv(fp)
    else:
        train_data = pd.read_parquet(fp)
    # train_data_json = json.load(open('data/InternalData/InternalData_2k.json', 'r'))
    # train_data = train_data_json[args.start_index: args.end_index]
    with torch.no_grad():
        for a in tqdm(range(train_data.shape[0])):
            caption = train_data['text'][a].strip()
            if caption in captions:
                continue
            captions.add(caption)
            if isinstance(caption, str):
                caption = [caption]

            save_path = os.path.join(t5_save_dir, os.path.basename(train_data['file_name'][a]).replace('.png', '.npz').replace('.jpg', '.npz').replace('.webp', '.npz').replace('.jpeg', '.npz'))
            print(f"trying {save_path}")
            if os.path.exists(save_path):
                continue
            try:
                caption_emb, emb_mask = t5.get_text_embeddings(caption)
                emb_dict = {
                    'caption_feature': caption_emb.float().cpu().data.numpy(),
                    'attention_mask': emb_mask.cpu().data.numpy(),
                }
                np.savez_compressed(save_path, **emb_dict)

            except Exception as e:
                print(e)


@ray.remote( num_cpus=3, num_gpus=1 , memory= 40* 1000 * 1024 * 1024) #10gb memory reserved 
class Actor:
    def __init__(self, args, dirs, df):
        self.args = args
        self.dirs = dirs
        self.df = df
        self.gpu_count = torch.cuda.device_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adirin = "" #sos.path.basename(adirin)        
        self.model_path_dir =args.t5_dir
        self.save_path_root =args.save_path_root
        self.train_data = df
        # os.mkdir("/home/ray/train_data/t5/", exists )s
        # os.umask(0o000)
        # 
        os.umask(0) 
        # os.umask(int('777', base=8))  # file permission: 666; dir permission: 777

        if not os.path.exists(self.model_path_dir ):
            print("downloading models")
            if not os.path.exists('/home/ray/models'):
                os.mkdir('/home/ray/models')
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            snapshot_download("PixArt-alpha/PixArt-alpha", cache_dir='/home/ray/models', local_dir='/home/ray/models')
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")
          
        # if not os.path.exists('/home/ray/models/PixArt-alpha/t5-v1_1-xxl/config.json'):
        #     # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
        #     # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
        #     snapshot_download("PixArt-alpha/PixArt-alpha")
        #     # os.system("cd /home/ray/models && git lfs pull https://huggingface.co/liuhaotian/llava-v1.5-7b")
        # else:
        #     print("found model ")

        self.t5 = T5Embedder(device=self.device, local_cache=True, cache_dir= self.model_path_dir , dir_or_name= self.model_path_dir  )
        self.t5_save_root = f'{self.save_path_root}t5_features'
        # t5_save_root = 'data/InternalData_2k/t5_features'

        self.t5_save_dir = self.t5_save_root
        try:
            os.makedirs(self.t5_save_dir, 777, exist_ok=True)
        except Exception as e:
            print(e)
        # captions = set()
        # imgfiles = list(Path('/home/ray/train_data/').rglob("*.jpg"))
        # # [p.chmod(0o666) for p in imgfiles]
        # print(f"found {len(imgfiles)} imgs on init")

    def hasgpu(self):
        return self.gpu_count > 0
    
    def inference(self):
        print(self.args)
        client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
                                            aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
                                            aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
                                            region_name= os.environ.get("BUCKET_REGION", "auto"))
        # at1/LLaVA/llava/serve/cli_batch_ray_par_a.py
        # if not os.path.exists(adirin):
        #     os.mkdir(adirin)
        s3 = boto3.resource('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL"))
        bucket = s3.Bucket('akash-thumper-v1-t5-data')
        os.umask(0)  # file permission: 666; dir permission: 777

        for a in self.dirs:
            filepath= str(self.train_data['file_name'].values[a])
            filename = os.path.basename(filepath).replace('.png', '.npz').replace('.jpg', '.npz').replace('.webp', '.npz').replace('.jpeg', '.npz')
            shardpath = filepath.replace(os.path.basename(filepath), "").replace("/home/ray/train_data/","")
            if int(float(shardpath.strip("/"))) >638:
                filepath = os.path.basename(filepath).replace('.png', '.npz').replace('.jpg', '.npz').replace('.webp', '.npz').replace('.jpeg', '.npz').replace("/home/ray/train_data/", self.t5_save_dir)
                save_path = f"{self.t5_save_dir}/{shardpath}/{filename}"
                # save_path = f"{self.t5_save_dir}/{filename}"

                save_folder =  f"{self.t5_save_dir}/{shardpath}"
                # os.umask(int('777', base=8))  # file permission: 666; dir permission: 777
                if not os.path.exists(save_folder):
                    try:

                        os.makedirs(save_folder, int('777', base=8), True)
                        os.chmod(save_folder, int('777', base=8))

                    except Exception as e:
                        print(e)

                # os.chmod(save_path, 777)
                # os.chmod(save_folder, int('777', base=8))

                # filepath.replace("/home/ray/train_data/", self.t5_save_dir)

                caption = str(self.train_data['text'].values[a]).strip()
                caption = [caption]
            
                # save_path = os.path.join(self.t5_save_dir, os.path.basename(train_data['file_name'][a]).replace('.png', '.npz').replace('.jpg', '.npz').replace('.webp', '.npz').replace('.jpeg', '.npz'))
                
                print(f"trying {save_path}")
                if os.path.exists(save_path):
                    continue
                try:
                    with torch.no_grad():
                        caption_emb, emb_mask = self.t5.get_text_embeddings(caption)
                        emb_dict = {
                            'caption_feature': caption_emb.float().cpu().data.numpy(),
                            'attention_mask': emb_mask.cpu().data.numpy(),
                        }
                        np.savez_compressed(save_path, **emb_dict)
                        print("saved", caption, save_path)

                except Exception as e:
                    print(e)
                if a % 1000 == 0:
                    os.system("b2 sync /home/ray/train_data/t5_features/ b2://akash-thumper-v1-t5-data --skipNewer")
                
        os.system("b2 sync /home/ray/train_data/t5_features/ b2://akash-thumper-v1-t5-data --skipNewer")
        return
    

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_class', default='action', type=str)
    # parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--image_resize', default=256, type=int)
    parser.add_argument('--dataset_name', default='at2s_test', type=str)
    parser.add_argument('--base_dir', default='/home/ray/train_data', type=str)
    parser.add_argument('--caption_filepath', default='s3://akash-thumper-v1-training-data/captionfile_v1.parquet', type=str)
    parser.add_argument('--vae_dir', default="/home/ray/models/PixArt-alpha/sd-vae-ft-ema", type=str)
    parser.add_argument('--t5_dir', default="/home/ray/models/PixArt-alpha/t5-v1_1-xxl", type=str)
    parser.add_argument('--model_path_dir', default="/home/ray/models/PixArt-alpha/t5-v1_1-xxl", type=str)
    parser.add_argument('--save_path_root', default="/home/ray/train_data/", type=str)


    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--worker", type=int, default=4)

    # parser.add_argument("--dlworker", type=int, default=1)
    # parser.add_argument("--max-new-tokens", type=int, default=128)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--skip-download", action="store_true",default=False)
    # parser.add_argument("--all-local-images", action="store_true",default=False)


    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print("args:",args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize =args.image_resize # 512
    print('Extracting Image Resolution %s' % image_resize)
    dataset_name = args.dataset_name #"at1_test"
    caption_filepath =  args.caption_filepath
    base_dir = args.base_dir

    # s3://akash-thumper-v1-training-data/captionfile_v1.parquet
    print("reading caption df")
    caption_df = pd.read_parquet(#argsss.caption_filpath
                                 "s3://akash-thumper-v1-training-data/captionfile_v1.parquet", storage_options={"key":'****************',  
                                                                        "secret":'****************',
                                                                        "endpoint_url":'ENTER_URL_HERE'})
    caption_df= caption_df.iloc[2400000:,:]
    # fps = captssion_df["file_name"].tolist()      
                       
    indexlst= list(range(caption_df.shape[0]))
    dir_chunks =list(more_itertools.divide(args.worker,indexlst ))
    dir_chunks = [list(achunk) for achunk in  dir_chunks]
    print("dir chunk", len(dir_chunks))

    ray.init(address="auto")

    actors = [Actor.remote(args=args, dirs=dirs, df=caption_df) for dirs in dir_chunks]
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



    # s3://akash-thumper-v1-training-data/captionfile_v1.parquet

    # # prepare extracted caption t5 features for training
    # extract_caption_t5(fp =caption_filepath,
    #                     save_path_root=f"{base_dir}/{dataset_name}/caption_feature_wmask",
    #                      model_path_dir=args.t5_dir)
    # # prepare extracted image vae features for training
    # extract_img_vae(fp =caption_filepath, 
    #                 save_path_root=f"{base_dir}/{dataset_name}/img_vae_feature",
    #                 model_path_dir=args.vae_dir,
    #                   image_resize= image_resize)
    
    # if not os.path.exists(f"{base_dir}/{dataset_name}/partition"):
    #     os.mkdir(f"{base_dir}/{dataset_name}/partition")

    # fp = caption_filepath
    # if ".csv" in fp:
    #     train_data = pd.read_csv(fp)
    # else:
    #     train_data = pd.read_parquet(fp)

    # partionlist = train_data["file_name"].sort_values().drop_duplicates().tolist()
    # partionlist = [f"{base_dir}/{dataset_name}/images/" + os.path.basename(file) for file in partionlist]


    # with open(f'{base_dir}/{dataset_name}/partition/part0.txt', 'w') as fp:
    #     fp.write('\n'.join(partionlist))

    # if not os.path.exists(f"{base_dir}/{dataset_name}/images"):
    #     os.mkdir(f"{base_dir}/{dataset_name}/images")

    # p = Path(args.img2img_dir)
    # img_files = list(p.rglob("*.jpg"))
    # print(f"found {len(img_files)} imgs ")
    # for img in img_files:
    #     # print(f"dest {img.name}")
    #     shutil.copy(img, f"{base_dir}/{dataset_name}/images/" + str(img.name) )

    # parquet_files = list(p.rglob("*.parquet"))
    # # ['url', 'key', 'status', 'error_message', 'width', 'height',
    # #    'original_width', 'original_height', 'exif', 'sha256'],
    # dflist=[]
    # for pq in parquet_files:
    #     try:
    #         dfi = pd.read_parquet(pq)
            
    #         #  data_info={'img_hw': hw, 'aspect_ratio': ar}
    #         dflist.append(dfi)
    #     except Exception as e:
    #         print(e)
    # df= pd.concat(dflist, axis =0)
    # df["img_hw"]= df[['height','width']].min(axis=1)
    # df["aspect_ratio"]= 1
    # df["ratio"]= 1
    # df["img_folder"] =f'{base_dir}/{dataset_name}/images/'
    # df["suffix"] = 'jpg'
    # df["path"]=  df["img_folder"]+ df['key']  + df["suffix"]
    
    # df.to_csv(f'{base_dir}/{dataset_name}/partition/info.csv')
    # df.to_json(f'{base_dir}/{dataset_name}/partition/info.json', orient="records")
    # print(df.head())


    # s3://akash-thumper-v1-training-data/captionfile_v1.parquet

    # # prepare extracted caption t5 features for training
    # extract_caption_t5(fp =caption_filepath,
    #                     save_path_root=f"{base_dir}/{dataset_name}/caption_feature_wmask",
    #                      model_path_dir=args.t5_dir)
    # # prepare extracted image vae features for training
    # extract_img_vae(fp =caption_filepath, 
    #                 save_path_root=f"{base_dir}/{dataset_name}/img_vae_feature",
    #                 model_path_dir=args.vae_dir,
    #                   image_resize= image_resize)
    
    # if not os.path.exists(f"{base_dir}/{dataset_name}/partition"):
    #     os.mkdir(f"{base_dir}/{dataset_name}/partition")

    # fp = caption_filepath
    # if ".csv" in fp:
    #     train_data = pd.read_csv(fp)
    # else:
    #     train_data = pd.read_parquet(fp)

    # partionlist = train_data["file_name"].sort_values().drop_duplicates().tolist()
    # partionlist = [f"{base_dir}/{dataset_name}/images/" + os.path.basename(file) for file in partionlist]


    # with open(f'{base_dir}/{dataset_name}/partition/part0.txt', 'w') as fp:
    #     fp.write('\n'.join(partionlist))

    # if not os.path.exists(f"{base_dir}/{dataset_name}/images"):
    #     os.mkdir(f"{base_dir}/{dataset_name}/images")

    # p = Path(args.img2img_dir)
    # img_files = list(p.rglob("*.jpg"))
    # print(f"found {len(img_files)} imgs ")
    # for img in img_files:
    #     # print(f"dest {img.name}")
    #     shutil.copy(img, f"{base_dir}/{dataset_name}/images/" + str(img.name) )

    # parquet_files = list(p.rglob("*.parquet"))
    # # ['url', 'key', 'status', 'error_message', 'width', 'height',
    # #    'original_width', 'original_height', 'exif', 'sha256'],
    # dflist=[]
    # for pq in parquet_files:
    #     try:
    #         dfi = pd.read_parquet(pq)
            
    #         #  data_info={'img_hw': hw, 'aspect_ratio': ar}
    #         dflist.append(dfi)
    #     except Exception as e:
    #         print(e)
    # df= pd.concat(dflist, axis =0)
    # df["img_hw"]= df[['height','width']].min(axis=1)
    # df["aspect_ratio"]= 1
    # df["ratio"]= 1
    # df["img_folder"] =f'{base_dir}/{dataset_name}/images/'
    # df["suffix"] = 'jpg'
    # df["path"]=  df["img_folder"]+ df['key']  + df["suffix"]
    
    # df.to_csv(f'{base_dir}/{dataset_name}/partition/info.csv')
    # df.to_json(f'{base_dir}/{dataset_name}/partition/info.json', orient="records")
    # print(df.head())