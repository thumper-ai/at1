import sys
import time
from collections import Counter
import argparse
import boto3
import ray
# from img2dataset import download
import img2dataset
import os
import pandas as pd

@ray.remote
def main(args):
    # if not os.path.exists(args.out_folder):
    #     os.mkdir(args.out_folder)
	# 
	df = pd.read_parquet(args.url_list)
	print("found ", df.shape)
	img2dataset.download(
	processes_count=1, 
	thread_count=32,
	retries=0,
	timeout=10,
	url_list=args.url_list,
	image_size=args.image_size,
	resize_only_if_bigger=True,
	resize_mode="border",
	skip_reencode=True,
	output_folder=args.out_folder,
	min_image_size=128,
	# output_format="webdataset",
	input_format="parquet",
	url_col="URL",
	# caption_col="alt",
	enable_wandb=False,
	subjob_size=48*120*2,
	number_sample_per_shard=10000,
	distributor="ray",
	oom_shard_count=8,
	compute_hash="sha256"
	# save_additional_columns=["uid"]
	)
    # os.system(f"b2 sync /home/ray/train_data b2://akash-thumper-v1-training-data --skipNewer --threads 32")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--url_list",  default='/home/ray/train_data/df_download_images.parquet', type=str)
	parser.add_argument("--out_folder", default='/home/ray/train_data/', type=str)
	parser.add_argument("--image_size", default=256, type=int)
	parser.add_argument("--n_processes", default=1, type=int)
	parser.add_argument("--local", action='store_true', default=False, help="local or cluster")
	parser.add_argument("--smoketest", action='store_true', default=False, help="local or cluster")

	args = parser.parse_args()
	if args.smoketest:
		import pandas as pd
		df = pd.read_parquet(args.url_list)
		df.iloc[0:10,:].to_parquet('/home/ray/smoketest.parquet')
		args.url_list = '/home/ray/smoketest.parquet'
	# args={"url_list":"/home/ray/train_data/final.parquet", "out_folder": "/home/ray/train_data/imgs"}
	# args={"url_list":"/mnt/sabrent/fdai/final.parquet", "out_folder": "/mnt/sabrent/cc02"}

	# client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
	# 									aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
	# 									aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
	# 									region_name= os.environ.get("BUCKET_REGION", "auto"))

	# model_dir = "/home/ray/train_data"
	# filename = "final.parquet"
	# import pandas as pd
	# df = pd.read_parquet(args.url_list)
	# df.iloc[0:60,:].to_parquet(args.url_list)
	# client.download_file( filename, f"{model_dir}/{filename}")
	# ray.init(address="auto")
	# ray.shutdown(

	if args.local:
		# os.system(f'cat <<< "{os.environ["AWS_ACCESS_KEY_ID"]}:{os.environ["AWS_SECRET_ACCESS_KEY"]}" > /etc/passwd-s3fs')
		# os.system(f'chmod 640 /etc/passwd-s3fs')
		# os.system("echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ${HOME}/.passwd-s3fs")
		# os.system("chmod 600 ${HOME}/.passwd-s3fs")

		# os.system(f"mkdir -p ~/s3fs-bucket2")
		# os.system(f's3fs akash-thumper-v1-training-images ~/s3fs-bucket2 -o url={os.environ["S3_ENDPOINT_URL"]}')
		os.environ["FSSPEC_S3_ENDPOINT"] = 	os.environ["S3_ENDPOINT_URL"]
		os.environ["FSSPEC_S3_KEY"] = 	os.environ["AWS_ACCESS_KEY_ID"]
		os.environ["FSSPEC_S3_SECRET"] = 	os.environ["AWS_SECRET_ACCESS_KEY"]

		img2dataset.download(
			processes_count=args.n_processes, 
			thread_count=32,
			retries=0,
			timeout=10,
			url_list=args.url_list,
			image_size=args.image_size,
			resize_only_if_bigger=True,
			resize_mode="border",
			skip_reencode=True,
			output_folder=args.out_folder,
			# output_format="webdataset",
			input_format="parquet",
			url_col="URL",
			# caption_col="alt",
			enable_wandb=False,
			subjob_size=48*120*2,
			number_sample_per_shard=10000,
			distributor="multiprocessing",
			oom_shard_count=8,
			compute_hash="sha256"
			# save_additional_columns=["uid"]
			)
	else:
		ray.init()
		job = main.remote(args)
		ray.get(job)
	# ray.shutdown()

# os.system("b2 sync /home/ray/train_data b2://akash-thumper-v1-training-data")
# b2 sync /home/ray/train_data b2:/akash-thumper-v1-training-images --skipNewer --threads 32

# from cloudpathlib import CloudPath

# cloud_path = CloudPath("s3://akash-thumper-v1-files")   # same for S3Path(...)
# cloud_path.upload_from("/home/ray/train_data")

# client = boto3.client('s3',  endpoint_url= os.environ.get("R2_BUCKET_URL", "https://.r2.cloudflarestorage.com"), 
# 										aws_access_key_id= os.environ.get("R2_ACCESS_KEY_ID", ""),
# 										aws_secret_access_key= os.environ.get("R2_SECRET_ACCESS_KEY", ""),
# 										region_name= os.environ.get("BUCKET_REGION", "auto"))

# client = S3Client(aws_access_key_id="myaccesskey", aws_secret_access_key="mysecretkey")

# these next two commands are equivalent
# use client's factory method
# cp1 = client.CloudPath("s3://akash-thumper-v1-files/")
# or pass client as keyword argument
# cp2 = CloudPath("s3://akash-thumper-v1-files", client=client)
	# "s3://akash-thumper-v1-training-images.s3.us-east-005.backblazeb2.com"
