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
import ray 
import more_itertools

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

def get_text_and_latent_embedings(dataloader, vae, model, drive_save_path=""):
    """dataloader that outputs an img tensor and text_prompts"""
    text_encodings = []
    img_encodings = []

    i = 0

    if drive_save_path != "" and not os.path.exists(drive_save_path):
        os.mkdir(drive_save_path)

    for img, label in tqdm(dataloader):
        try:
            encode_images = encode_image(img, vae)
            encode_texts = encode_text(label, model)
            text_encodings.append(encode_texts)
            img_encodings.append(encode_images)
            
        except Exception as e:
            print(e)
        if i % 1000 == 0:
            print(f"{i} done")
        #     np.save(os.path.join(drive_save_path, 'image_latents.npy'), torch.cat(img_encodings).numpy())
        #     np.save(os.path.join(drive_save_path, 'text_encodings.npy'), torch.cat(text_encodings).numpy())
    
    img_encodings = torch.cat(img_encodings)
    text_encodings = torch.cat(text_encodings)
    # if drive_save_path != "":
    #     np.save(os.path.join(drive_save_path, 'image_latents.npy'), torch.cat(img_encodings).numpy())
    #     np.save(os.path.join(drive_save_path, 'text_encodings.npy'), torch.cat(text_encodings).numpy())

    return img_encodings, text_encodings

        
@ray.remote(num_cpus=4, num_gpus=1)  
def download_and_process_data(dfind, shard,
                              latent_save_path='latents',
                              raw_imgs_save_path='raw_imgs',
                              csv_path = 'imgs.csv',
                              image_size = 256,
                              bs = 256,
                              caption_col = "prompt",
                              url_col = "url",
                              download_data=True,
                              number_sample_per_shard=256,
                              
                             ):
    from diffusers import AutoencoderKL
    if not os.path.exists("/home/ray/train_data"):
        os.makedirs("/home/ray/train_data", exist_ok=True)
    
    pqs = ["https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/0.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/1.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/2.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/3.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/4.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/5.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/6.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/7.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/8.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/9.parquet"]
    dflist=[]
    for f in pqs:
        df = pd.read_parquet(f)
        df = df.loc[df['is_upscaled'] == True, ["id", "prompt", "url"]]
        dflist.append(df)
        # df.to_parquet(f"/home/logan/thumperai/data/{os.path.basename(f)}")
    df = pd.concat(dflist, axis = 0)
    df.loc[:, "key"] = df["id"]
    df= df.reset_index(drop=True)

    # data_link = 'https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true'
    # df = pd.read_parquet(data_link)
    # df = df.loc[:,["key", "url", "caption"]] #.to_csv("imgs.csv", index=None)
    df = df.iloc[dfind,:]
    save_data = True
    df.to_csv(csv_path, index=None)

    if not os.path.exists(raw_imgs_save_path):
        os.mkdir(raw_imgs_save_path)
        os.mkdir(f"{raw_imgs_save_path}/_tmp/")

    if not os.path.exists(latent_save_path):
        os.mkdir(latent_save_path)

    if download_data:

        download(
            processes_count=16,
            thread_count=16,
            url_list=csv_path,
            image_size=image_size,
            output_folder=raw_imgs_save_path,
            output_format="webdataset",
            input_format="csv",
            url_col=url_col,
            caption_col=caption_col,
            enable_wandb=False,
            number_sample_per_shard=number_sample_per_shard,
            distributor="multiprocessing",
            disallowed_header_directives=[],
            # distributor="ray",

            resize_mode="center_crop"
        )

        # img = A.smallest_max_size(img, self.image_size, interpolation=interpolation)

    files = os.listdir(raw_imgs_save_path)
    tar_files = [os.path.join(raw_imgs_save_path, file) for file in files if file.endswith('.tar')]
    print("tar files", tar_files)

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
    image_latents, text_encodings = get_text_and_latent_embedings(dataloader, vae, model, latent_save_path)

    if save_data:
        np.save(os.path.join(latent_save_path, f'image_latents_{shard}.npy'), image_latents.numpy())
        np.save(os.path.join(latent_save_path, f'text_encodings_{shard}.npy'), text_encodings.numpy())
        s3 = s3fs.S3FileSystem(#anon=True, 
        #                        use_listings_cache=False,
                            key='****************',
        #                        key='*****************',
                            secret='****************',
        #                        secret='*****************',
                            endpoint_url='ENTER_URL_HERE', version_aware=True)
        s3.put(os.path.join(latent_save_path, f'image_latents_{shard}.npy'), "akash-thumper-v1-mj-256")
        s3.put(os.path.join(latent_save_path, f'text_encodings_{shard}.npy'), "akash-thumper-v1-mj-256" )


if __name__ == '__main__':
    # data_link = 'https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true'
    # df = pd.read_parquet(data_link)
    ###add additional data cleaning here...
    # df = df.iloc[0:20000]

    import pandas as pd
    import os
    # curl -X GET \
    #      "https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train"

    pqs = ["https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/0.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/1.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/2.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/3.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/4.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/5.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/6.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/7.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/8.parquet","https://huggingface.co/api/datasets/vivym/midjourney-prompts/parquet/default/train/9.parquet"]
    dflist=[]
    for f in pqs:
        df = pd.read_parquet(f)
        df = df.loc[df['is_upscaled'] == True, ["id", "prompt", "url"]]
        dflist.append(df)
        # df.to_parquet(f"/home/logan/thumperai/data/{os.path.basename(f)}")
    df = pd.concat(dflist, axis = 0)
    df = df.reset_index(drop=True)
    
    indexlst= list(range(df.shape[0]))
    dir_chunks = list(more_itertools.divide(15 ,indexlst ))
    indexes = [list(achunk) for achunk in  dir_chunks]

    # print("indexes",indexes)
    dflist = [df.iloc[ind,:] for ind in indexes]
    
    # df = df.iloc[0:10000]
    df = df.loc[:,["id", "url", "prompt"]] #.to_csv("imgs.csv", index=None)
    df.loc[:, "key"] = df["id"]


    caption_col = 'prompt' 
    url_col = 'url'
    # latent_save_path = f'/home/ray/train_data/latent256_8_{}' #could also be a path to google drive for persisting the data
    # raw_imgs_save_path = f'/home/ray/train_data/raw_imgs256_8_{}' #raw imgs are not needed for training so they can be deleted once done
    use_drive = False

    # if use_drive:
    #     from google.colab import drive
    #     drive.mount('/content/drive')
    ray.init("auto")
    jobs=[]
    for i, df in enumerate(dflist):  
        jobs.append( download_and_process_data.remote( dfind= indexes[i], shard=i,
                                latent_save_path= f"/home/ray/train_data/latent256_9_{i}",
                                raw_imgs_save_path = f"/home/ray/train_data/raw_imgs256_9_{i}",  
                                csv_path=f"/home/ray/train_data/imgs_{i}.csv",
                                image_size=256,
                                bs=128,
                                caption_col=caption_col,
                                url_col = url_col,
                                download_data=True,
                                number_sample_per_shard=2560))

    ray.get(jobs)
    from torch.utils.data import DataLoader, TensorDataset

    s3 = s3fs.S3FileSystem(#anon=True, 
            #                        use_listings_cache=False,
                                key='****************',
            #                        key='*****************',
                                secret='****************',
            #                        secret='*****************',
                                endpoint_url='ENTER_URL_HERE', version_aware=True)
    latent_train_data =[]
    train_label_embeddings =[]
    #26
    for a in range(52):
        image_latent_path = f"/home/ray/train_data/image_latents_{a}.npy"
        text_emb_path = f"/home/ray/train_data/text_encodings_{a}.npy"
        if not os.path.exists(image_latent_path):
            s3.get(f"akash-thumper-v1-mj-256/image_latents_{a}.npy", "/home/ray/train_data")
        if not os.path.exists(text_emb_path):
            s3.get(f"akash-thumper-v1-mj-256/text_encodings_{a}.npy", "/home/ray/train_data")
        try:
            lt_img = np.load(image_latent_path).squeeze()
            text_emb = np.load(text_emb_path).squeeze()
            print(a, "size ", lt_img.shape, text_emb.shape )
            
            latent_train_data.append(torch.tensor(lt_img, dtype=torch.float16))
            train_label_embeddings.append(torch.tensor(text_emb, dtype=torch.float16))
        except Exception as e:
            print(e)
            
    latent_train_data= torch.cat(latent_train_data).squeeze()
    train_label_embeddings= torch.cat(train_label_embeddings).squeeze()
    print("latent img size ",latent_train_data.size(),  train_label_embeddings.size())
    dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    def upload_shard(fname):
        s3 = s3fs.S3FileSystem(#anon=True, 
    #                        use_listings_cache=False,
                    key='****************',
    #                        key='*****************',
                    secret='****************',
    #                        secret='*****************',
                    endpoint_url='ENTER_URL_HERE', version_aware=True)
        s3.put(fname, "akash-thumper-v1-mj-256" )
        # os.system(f"b2 upload_file akash-thumper-v1-wds3 {fname} {os.path.basename(fname)}")  # replace with your preferred command
        os.unlink(fname)        
    if not os.path.exists("/home/ray/wds256_11/"):
        os.mkdir("/home/ray/wds256_11/")
    i=0
    with wds.ShardWriter(f"/home/ray/wds256_10/wds_at1-" + "-%d.tar",  maxcount=600000, maxsize=6000000000.0, post=upload_shard) as writer:
        for x,label in train_loader:
            mds_sample = {
                            "__key__": "sample%06d" % i,
                            "image_latents.pyd": x,
                            'text_encodings.pyd': label,
                                    }
                                    
            writer.write(mds_sample)
            if i%100 == 1:
                print(f"Saving {i}")
            i += 1
