####data util to get and preprocess data from a text and image pair to latents and text embeddings.
### all that is required is a csv file with an image url and text caption:
#!pip install datasets img2dataset accelerate diffusers
#!pip install git+https://github.com/openai/CLIP.git

import os
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import webdataset as wds
from img2dataset import download

#models:
import clip
from diffusers import AutoencoderKL
import s3fs

@torch.no_grad()
def encode_text(label, model):
    text_tokens = clip.tokenize(label, truncate=True).cuda()
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()

@torch.no_grad()
def encode_image(img, vae):
    x = img.to('cuda').to(torch.float16)

    x = x*2 - 1 #to make it between -1 and 1.
    encoded = vae.encode(x, return_dict=False)[0].sample()
    return encoded.cpu()

@torch.no_grad()
def decode_latents(out_latents, vae):
    #expected to be in the unscaled latent space 
    out = vae.decode(out_latents.cuda())[0].cpu()

    return ((out + 1)/2).clip(0,1)


def quantize_latents(lat, clip_val=20):
    """scale and quantize latents to unit8"""
    lat_norm = lat.clip(-clip_val, clip_val)/clip_val
    return (((lat_norm + 1)/2)*255).to(torch.uint8)

def dequantize_latents(lat, clip_val=20):
    lat_norm = (lat.to(torch.float16)/255)*2 - 1
    return lat_norm*clip_val

def get_text_and_latent_embedings(dataloader, vae, model, drive_save_path):
    """dataloader that outputs an img tensor and text_prompts"""
    text_encodings = []
    img_encodings = []

    i = 0
    def upload_shard(fname):
        s3 = s3fs.S3FileSystem(#anon=True, 
    #                        use_listings_cache=False,
                    key='****************',
    #                        key='*****************',
                    secret='****************',
    #                        secret='*****************',
                    endpoint_url='ENTER_URL_HERE', version_aware=True)
        s3.put(fname, "akash-thumper-v1-wds-256" )
        # os.system(f"b2 upload_file akash-thumper-v1-wds3 {fname} {os.path.basename(fname)}")  # replace with your preferred command
        # os.unlink(fname)
    if not os.path.exists("/home/ray/wds256_7/"):
        os.mkdir("/home/ray/wds256_7/")
    with wds.ShardWriter(f"/home/ray/wds256_7/wds_at1-" + "-%d.tar", maxsize=500000, post=upload_shard) as writer:

        for img, label in tqdm(dataloader):
            encode_images = encode_image(img, vae).cpu().half().numpy()
            encode_texts = encode_text(label, model).cpu().half().numpy()
            print(encode_images.shape,encode_texts.shape)
            for a in range(min( encode_texts.shape[0], encode_images.shape[0])):   
                try:
                #encode text:
                    # text_encodings.append(encode_text(label, model).cpu())
                    # ##encode images:

                    # # img_encodings.append(encode_image(img, vae).cpu())
                    # encode_images = encode_image(img, vae).cpu().half().numpy()
                    # encode_texts = encode_text(label, model).cpu().half().numpy()

                    mds_sample = {
                                    "__key__": "sample%06d" % i,
                                    "image_latents.npy": encode_images[a,],
                                    'text_encodings.npy': encode_texts[a,],
                                            }
                                            
                    writer.write(mds_sample)
                    if i%100 == 1:
                        print(f"Saving {i}")
                    i += 1

                except Exception as e:
                    print(e)

            #     np.save(os.path.join(drive_save_path, 'image_latents.npy'), torch.cat(img_encodings).numpy())
            #     np.save(os.path.join(drive_save_path, 'text_encodings.npy'), torch.cat(text_encodings).numpy())


        # img_encodings = torch.cat(img_encodings)
        # text_encodings = torch.cat(text_encodings)

        # return img_encodings, text_encodings

        
        
def download_and_process_data(latent_save_path='latents',
                              raw_imgs_save_path='raw_imgs',
                              csv_path = 'imgs.csv',
                              image_size = 256,
                              bs = 64,
                              caption_col = "captions",
                              url_col = "url",
                              download_data=True,
                              number_sample_per_shard=10000,
                             ):
    
    save_data = True

    if not os.path.exists(raw_imgs_save_path):
        os.mkdir(raw_imgs_save_path)
        os.mkdir(f"{raw_imgs_save_path}/_tmp/")

    if not os.path.exists(latent_save_path):
        os.mkdir(latent_save_path)

    # if download_data:

    #     download(
    #         processes_count=34,
    #         thread_count=32,
    #         url_list=csv_path,
    #         image_size=image_size,
    #         output_folder=raw_imgs_save_path,
    #         output_format="webdataset",
    #         input_format="csv",
    #         url_col=url_col,
    #         caption_col=caption_col,
    #         enable_wandb=False,
    #         number_sample_per_shard=number_sample_per_shard,
    #         distributor="multiprocessing",
    #         # distributor="ray",

    #         resize_mode="center_crop"
    #     )

    files = os.listdir(raw_imgs_save_path)
    tar_files = [os.path.join(raw_imgs_save_path, file) for file in files if file.endswith('.tar')]
    print(tar_files)

    dataset = wds.WebDataset(tar_files)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = dataset.decode("pil").to_tuple("jpg;png", "json").map_tuple(transform, lambda x: x["caption"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

    model, preprocess = clip.load("ViT-L/14")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    model.to('cuda')

    print("Starting to encode latents and text:")
    # image_latents, text_encodings = 
    get_text_and_latent_embedings(dataloader, vae, model, latent_save_path)

    # if save_data:
    #     np.save(os.path.join(latent_save_path, 'image_latents.npy'), image_latents.numpy())
    #     np.save(os.path.join(latent_save_path, 'text_encodings.npy'), text_encodings.numpy())

if __name__ == '__main__':
    data_link = 'https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true'
    df = pd.read_parquet(data_link)
    ###add additional data cleaning here...
    df = df.iloc[0:500000]
    df[["key", "url", "caption"]].to_csv("imgs.csv", index=None)


    caption_col = 'text' 
    url_col = 'url'
    latent_save_path = '/home/ray/train_data/latent256_7' #could also be a path to google drive for persisting the data
    raw_imgs_save_path = '/home/ray/train_data/raw_imgs256_7' #raw imgs are not needed for training so they can be deleted once done
    use_drive = False

    # if use_drive:
    #     from google.colab import drive
    #     drive.mount('/content/drive')


    download_and_process_data(latent_save_path=latent_save_path,
                                raw_imgs_save_path=raw_imgs_save_path,        ##encode images:

                                csv_path='imgs.csv',
                                image_size=256,
                                bs=256,
                                caption_col=caption_col,
                                url_col = url_col,
                                download_data=True,
                                number_sample_per_shard=1024
                            )


