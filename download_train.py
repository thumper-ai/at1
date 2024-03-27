
import boto3
import os , io
import ray 

# def upload_file(client, bucket,key, filepath= None):
#     if filepath is None:
#         fo = io.BytesIO(b'my data stored as file object in RAM')
#         client.upload_fileobj(fo, bucket, key)
#     else:
#         client.upload_file(filepath, bucket, os.path.basename(filepath)
#             
#                ) 

# RUN cd ./LLaVA && conda create -n llava python=3.10 -y 
# RUN conda create -n pixart python==3.9.0
# os.system("cd /home/ray/LLaVA && conda create -n llava python=3.10 -y ")
# os.system("cd /home/ray/PixArt-alpha && conda create -n pixart python==3.9.0 -y")

os.system('./home/ray/install_conda_deps.sh')
# os.system("/bin/bash/ -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate pixart && cd ./PixArt-alpha &&  pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 -y && pip install -r requirements.txt ")
# #   
# def generate_caption_data():
#     client = boto3.client('s3',  endpoint_url="ENTER_URL_HERE", 
#                                         aws_access_key_id= "****************",
#                                         aws_secret_access_key= '****************',
#                                         # region_name= "auto"
#                                         )
#     model_dir = "/tmp"
#     filename = "df_download_images.parquet"
#     print("starting download")
#     client.download_file(Bucket="akash-thumper-v1-files", Key= filename, Filename= f"{model_dir}/{filename}")
#     print("finished download")

client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL"), 
                                       aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
                                       aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
                                    #    region_name= os.environ.get("BUCKET_REGION", "auto")
                                       )

# @ray.remote(num_gpus=7, num_cpus=12)
def generate_caption_data():
    client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL"), 
                                        aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID"),
                                        aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY"),
                                        # region_name= os.environ.get("BUCKET_REGION", "auto")
                                        )
    model_dir = "/home/ray/train_data"
    filename = "df_download_images.parquet"
    print("starting download")
    client.download_file(Bucket="akash-thumper-v1-files", Key= filename, Filename= f"{model_dir}/{filename}")
    print("finished download")

    if not os.path.exists('/home/ray/models'):
        os.mkdir('/home/ray/models')
    os.system("cd /home/ray/models && git lfs clone https://huggingface.co/PixArt-alpha/PixArt-alpha")
    os.system("cd /home/ray/models && git lfs clone https://huggingface.co/liuhaotian/llava-v1.5-7b")

generate_caption_data()

# ray.init(address="auto")
# jobs = [generate_caption_data.remote() for i in range(4)]
# ray.get(jobs)

# print("done")