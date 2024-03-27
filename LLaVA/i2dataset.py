import sys
import time
from collections import Counter

import ray
# from img2dataset import download
import img2dataset
import argparse
import os
import json 


output = {
  "s3": {
    "client_kwargs": {
            "endpoint_url": os.environ["S3_ENDPOINT_URL"],
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
           "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"]
    }
  }
}
# Serializing json
json_object = json.dumps(output, indent=4)
# Writing to sample.json
with open("/home/ray/.config/fsspec/s3.json", "w") as outfile:
    outfile.write(json_object)
# akash-thumper-v1-training-images
@ray.remote
def main(args):
    img2dataset.download(
	processes_count=1, 
	thread_count=32,
	retries=0,
	timeout=10,
	url_list=args.url_list,
	image_size=512,
	resize_only_if_bigger=True,
	resize_mode="border",
	skip_reencode=True,
	output_folder=args.out_folder,
	# output_format="webdataset",
	input_format="parquet",
	url_col="image_url",
	# caption_col="alt",
	enable_wandb=True,
	subjob_size=48*120*2,
	number_sample_per_shard=10000,
	distributor="ray",
	oom_shard_count=8,
    compute_hash="sha256"
	# save_additional_columns=["uid"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_list")
    parser.add_argument("--out_folder")
    args = parser.parse_args()

    ray.init()
    job = main.remote(args)
    ray.get(job)
    ray.shutdown()
    
	
