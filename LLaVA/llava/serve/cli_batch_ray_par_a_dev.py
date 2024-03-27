import argparse
import os.path

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates

from PIL import Image
import pandas as pd
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
from pathlib import Path
import ray 
import boto3
import os
import io
from more_itertools import chunked

import pathlib


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):

        
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').squeeze(0)

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

class LlavaCaptionDataset(Dataset):
    """LLava for captioning inference dataset."""
    def __init__(self, prompt, paths, tokenizer, image_processor):
        """
        Arguments:
            npy_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = paths
        self.idx = 0
        self.prompt =prompt
        self.image_processor = image_processor
        self.tokenizer= tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[self.idx]
        self.idx+=1
        image = Image.open(path)
        # Check if the image has an alpha (transparency) channel
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Create a new white background image
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image onto the background using the alpha channel as a mask
            background.paste(image, (0, 0), image)

        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, args)#.squeeze(0)
        image_tensor = image_tensor.to(torch.float16)

        input_ids = tokenizer_image_token(self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') #.unsqueeze(0)

        sample = {'image_tensor': image_tensor, 'input_ids': input_ids, 'path':path}
        # if self.transform:
        #     sample = self.transform(sample)
        return sample


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __call__(self, sample):
#         image_tensor, input_ids = sample['image_tensor'], sample['input_ids']
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         # image = image.transpose((2, 0, 1))
#         return {'image_tensor': torch.from_numpy(image_tensor),
#                 'input_ids': input_idstorch.from_numpy(input_ids)}

def generate_texts(args, paths, model, tokenizer, image_processor, prompt, start_index, end_index):
    file_names=[]
    texts=[]
    image_tensors=[]
    for path in paths[start_index:end_index]:
        image = Image.open(path)

        # Check if the image has an alpha (transparency) channel
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Create a new white background image
            background = Image.new('RGB', image.size, (255, 255, 255))

            # Paste the image onto the background using the alpha channel as a mask
            background.paste(image, (0, 0), image)

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensors.append(image_tensor)

    image_tensor = torch.cat(image_tensors, 0)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = torch.repeat_interleave(input_ids, repeats=image_tensor.size()[0], dim=0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

    for a in range(output_ids.size()[0]):
        outputs = tokenizer.decode(output_ids[a, input_ids.shape[1]:],skip_special_tokens=True).strip()
        texts.append(outputs)
        file_names.append(os.path.basename(paths[start_index+a]))

    if args.debug:
        print("\n", {"outputs": outputs}, "\n")

    return file_names,texts


def upload_file(client, bucket,key, filepath= None):
    if filepath is None:
        fo = io.BytesIO(b'my data stored as file object in RAM')
        client.upload_fileobj(fo, bucket, key)
    else:
        client.upload_file(filepath, bucket, os.path.basename(filepath)
                           ) 
        
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL")) # assumes credentials & configuration are handled outside python in .aws directory or environment variables
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

def download_s3_folder2(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource('s3', endpoint_url= os.environ.get("R2_BUCKET_URL")) # assumes credentials & configuration are handled outside python in .aws directory or environment variables
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        print(f"downloading {obj.key}")
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            
            continue
        if not os.path.exists(f"{target}/{obj.key}"):
            bucket.download_file(obj.key, target) 

# client = boto3.client('s3',  endpoint_url= os.environ.get("BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
#                                        aws_access_key_id= os.environ.get("ACCESS_KEY_ID", ""),
#                                        aws_secret_access_key= os.environ.get("SECRET_ACCESS_KEY", ""),
#                                        region_name= os.environ.get("BUCKET_REGION", "auto"))

# @ray.remote(ngpu=1)
# def generate_caption_data(input_file):
#     client = s3_client()
#   client.download_file( args["checkpoint_filename"], f"{model_dir}/{args['checkpoint_filename']}")

@ray.remote(num_cpus=3, num_gpus=1 , memory= 40* 1000 * 1024 * 1024) #10gb memory reserved 
class Actor:
    def __init__(self, args):
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

        self.args = args
        disable_torch_init()

        self.gpu_count = torch.cuda.device_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adirin = "" #sos.path.basename(adirin)
        # print(f"starting {adirin}")
        # client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
        #                                     aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
        #                                     aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        #                                     region_name= os.environ.get("BUCKET_REGION", "auto"))
        
        if not os.path.exists('/home/ray/models/llava-v1.5-7b'):
            print("downloading models")
            if not os.path.exists('/home/ray/models'):
                os.mkdir('/home/ray/models')
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            os.system("cd /home/ray/models && git lfs clone https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")
          
        if not os.path.exists('/home/ray/models/llava-v1.5-7b/config.json'):
      
            # download_s3_folder2(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" )
            # os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
            os.system("cd /home/ray/models && git lfs pull https://huggingface.co/liuhaotian/llava-v1.5-7b")
        else:
            print("found model ")

        self.model_name = get_model_name_from_path(args.model_path)
        print(f"modelname {self.model_name}")
        self.model_name="liuhaotian/llava-v1.5-7b"
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path=args.model_path, model_base=args.model_base, model_name=self.model_name, load_8bit= args.load_8bit,load_4bit= args.load_4bit, device=self.device)


    def hasgpu(self):
        return self.gpu_count > 0
    
    def inference(self, adirin):
        client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
                                            aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
                                            aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
                                            region_name= os.environ.get("BUCKET_REGION", "auto"))
        # if not os.path.exists(adirin):
        #     os.mkdir(adirin)
        adirin =os.path.basename(adirin.replace("/",""))

        if not os.path.exists(f"/home/ray/train_data/{adirin}"):
               # if not os.path.exists(adirin):
        #     os.mkdir(adirin)
            print(f"downloading shard {adirin} ")
            download_s3_folder('akash-thumper-v1-training-images', adirin, local_dir="/home/ray/train_data")
    
        # adirin =os.path.basename(adirin)
        self.adirin = adirin

        self.model_name="liuhaotian/llava-v1.5-7b"

        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if self.args.conv_mode is not None and self.conv_mode != self.args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(self.conv_mode, self.args.conv_mode, self.args.conv_mode))
        else:
            args.conv_mode = self.conv_mode

        self.conv = conv_templates[args.conv_mode].copy()


        paths = glob.glob(os.path.join(f"{self.args.image_folder}/{self.adirin}","*.png"), recursive=True)
        paths += glob.glob(os.path.join(f"{self.args.image_folder}/{self.adirin}","*.jpg"), recursive=True)
        paths += glob.glob(os.path.join(f"{self.args.image_folder}/{self.adirin}" ,"*.jpeg"), recursive=True)
        paths += glob.glob(os.path.join(f"{self.args.image_folder}/{self.adirin}","*.webp"), recursive=True)
        # first message
        if self.model.config.mm_use_im_start_end:
            user_input = DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN+"\n"+self.args.user_prompt
        else:
            user_input = DEFAULT_IMAGE_TOKEN+"\n"+self.args.user_prompt

        self.conv.append_message(self.conv.roles[0], user_input)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        texts= []
        file_names=[]
        dataset = LlavaCaptionDataset(prompt= prompt, paths = paths, tokenizer=self.tokenizer, image_processor=self.image_processor )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        with torch.inference_mode():
            for i_batch, sample_batched in enumerate(dataloader):
                print(i_batch, sample_batched['image_tensor'].size(), sample_batched['input_ids'].size())
                output_ids = self.model.generate(
                    sample_batched['input_ids'].to("cuda"),
                    images=sample_batched['image_tensor'].to("cuda"),
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens
                )
                print("outputids",output_ids.size() , sample_batched['input_ids'].size(), output_ids[:,55:] )
                outputs = self.tokenizer.batch_decode(output_ids[:,sample_batched['input_ids'].size()[1]: ],skip_special_tokens=True)#.strip()
                outputs = [out.strip().replace("\n\n", "") for out in outputs]
                # outputs = [out[: -len(stop_str)] if out.endswith(stop_str) else out for out in outputs]
                texts.extend(outputs)
                file_names.extend(sample_batched['path'])
                # outputs = tokenizer.decode(output_ids[:, sample_batched['input_ids'].size()[1]:],skip_special_tokens=True).strip()
                print("outputs", outputs)
                # for a in range(output_ids.size()[0]):
                #     outputs = tokenizer.decode(output_ids[a, sample_batched['input_ids'].size()[1]:],skip_special_tokens=True).strip()
                #     texts.append(outputs)
                #     file_names.append(os.path.basename( sample_batched['path'][a]))
                # if i_batch ==2:
                #     results={"file_name":file_names,"text":texts}
                #     pd.DataFrame(results).to_parquet(args.output_csv,index=False)
                #     upload_file(client, bucket="akash-thumper-v1-training-data" , key= os.path.basename(args.output_csv.replace(".parquet", f"caption_{adirin}.parquet"), filepath= args.output_csv.replace(".parquet", f"_{worker}.parquet")))


                # if i_batch % 1000 ==0:
                #     results={"file_name":file_names,"text":texts}
                #     pd.DataFrame(results).to_parquet(args.output_csv,index=False)
                #     upload_file(client, bucket="akash-thumper-v1-training-data" , key= os.path.basename(self.args.output_csv.replace(".parquet", f"tempcaption_{i_batch}_{adirin}.parquet"), filepath= self.args.output_csv.replace(".parquet", f"tempcaption_{i_batch}_{adirin}.parquet")))
        results={"file_name":file_names,"text":texts}
        df = pd.DataFrame(results)
        # df.to_parquet(f"{self.args.image_folder}/{adirin}/final_{adirin}.parquet",index=False)
        # df.to_parquet(f"/tmp/final_{adirin}.parquet",index=False)

        # df.to_parquet(f"s3://akash-thumper-v1-captions/final_{adirin}.parquet",index=False)
        # upload_file(self.client, bucket="akash-thumper-v1-training-data" ,
        #              key= os.path.basename(f"{self.args.image_folder}/{adirin}/finalcaption_{adirin}.parquet"),
        #                filepath= f"{self.args.image_folder}/{adirin}/finalcaption_{adirin}.parquet")
     
        if df.shape[0] >10:
            status = "success"
            df.to_parquet(f"/tmp/final_{adirin}.parquet",index=False)
            upload_file(self.client, bucket="akash-thumper-v1-training-data" ,
                key= os.path.basename(f"/tmp/finalcaption_{adirin}.parquet"),
                filepath= f"/tmp/finalcaption_{adirin}.parquet")
        else:
            status = "failed"
            print("not enough captions generated")
        # self.value += 1
        return status, adirin, f"finalcaption_{adirin}.parquet"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)

    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--user-prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None) 

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--worker", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--temperature", type=float, default=0.2)
    
    args = parser.parse_args()

    client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
                                    aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
                                    aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
                                    region_name= os.environ.get("BUCKET_REGION", "auto"))
    
    # dirlist = list(pathlib.Path("/home/ray/train_data").iterdir())

    # dirlist = [x for x in pathlib.Path("/home/ray/train_data").iterdir() if x.is_dir()]
    dirlist = []
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket='akash-thumper-v1-files', Delimiter='/'):
        for prefix in result.get('CommonPrefixes'):
            dirlist.append(prefix.get('Prefix'))

    print("dirs", dirlist)
    ray.init(address="auto")

    actors = [Actor.remote(args) for a in range(args.worker)]
    jobs = []
    valid_actors =[]
    for actor in actors:
        if ray.get(actor.hasgpu.remote()) > 0:
            valid_actors.append(actor)
    print(f"{len(valid_actors)} of {len(actors)} validated for gpu detection")

    dir_chunks =list(chunked(dirlist, len(valid_actors)))

    for adirin_list in dir_chunks:
        for i, actor in enumerate(valid_actors):
            if i < len(adirin_list):
                jobs.append(actor.inference.remote(adirin= adirin_list[i]))
    results = ray.get(jobs)

    print("result files:", results)





    # if os.path.exists(args.image_folder):
    #   client.download_file( args["checkpoint_filename"], f"{model_dir}/{args['checkpoint_filename']}")
    # ray.init(address="auto")
    # jobs = [main.remote(args, a, pathlist[a] ) for a in range(args.worker)]
    # ray.get(jobs)

# python -m llava.serve.cli_batch --model-path liuhaotian/llava-v1.5-13b \ 
# --load-4bit \
# --user-prompt このイラストを日本語でできる限り詳細に説明してください。表情や髪の色、目の色、耳の種類、服装、服の色 など注意して説明してください。説明は反復を避けてください。\
#  --image-folder '/mnt/NVM/test'  \
# --output-csv '/mnt/NVM/test/metadata.csv'
# --batch-size 4




# python -m llava.serve.cli_batch --model-path '/mnt/sabrent/llava-v1.5-7b' \ 
# --load-4bit \
# --max-new-tokens 128 \
# --user-prompt 'describe this image and its style in a highly detailed manner' \
# --image-folder '/mnt/sabrent/cc03'  \
# --output-csv '/home/logan/thumperai/test5.csv'





# python -m llava.serve.cli_batch --model-path '/mnt/sabrent/llava-v1.5-7b' \ 
# --load-4bit \
# --max-new-tokens 128 \
# --user-prompt 'describe this image and its style in a highly detailed manner' \
# --image-folder '/mnt/sabrent'  \
# --output-csv '/home/logan/thumperai/test.csv'


# 15images


# python -m llava.serve.cli_batch --model-path '/mnt/sabrent/llava-v1.5-7b' \ 
# --load-4bit \
# --max-new-tokens 128 \
# --user-prompt 'describe this image in a highly detailed manner' \
# --image-folder '/home/ray/train_datat'  \
# --output-csv '/home/ray/train_datat/'
