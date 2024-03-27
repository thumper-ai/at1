from pathlib import Path
import ray 
import numpy as np
import boto3, os
import ray 
# @ray.remote(num_cpus=40)
# def count_images():
#     path = '/home/ray'

#     p = Path(path)

#     images = list(p.rglob("*.jpg"))
#     print(f"n images {len(images)} ")

#     return len(images)

# n_images= ray.get([count_images.remote() for a in range(4)])
# total = np.array(n_images).sum()
# print(f"total: {total} n images {n_images} ")
# @ray.remote(num_cpus=55, num_gpus=8)
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
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
        bucket.download_file(obj.key, target)

# @ray.remote(num_gpus=4)
def upload_files(path):
    session = boto3.Session(
        # endpoint_url= os.environ.get("R2_BUCKET_URL"), 
        aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        region_name="auto"
        )
    #     aws_access_key_id='YOUR_AWS_ACCESS_KEY_ID',
    #     aws_secret_access_key='YOUR_AWS_SECRET_ACCESS_KEY_ID',
    #     region_name='YOUR_AWS_ACCOUNT_REGION'
    # )
    # client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL"), 
    #                                    aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
    #                                    aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    #                                 #    region_name= os.environ.get("BUCKET_REGION", "auto")
    #                                    )
    s3 = session.resource('s3', endpoint_url= os.environ.get("R2_BUCKET_URL"), region_name="auto")
    bucket = s3.Bucket('akash-thumper-v1-files')
 
    for subdir, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                bucket.put_object(Key=full_path[len(path)+1:], Body=data)
 
 
upload_files('/home/ray/models')

# ray.get([download_s3_folder.remote(bucket_name='akash-thumper-v1-files', s3_folder="models" , local_dir="/home/ray/models" ) for a in range(4)])