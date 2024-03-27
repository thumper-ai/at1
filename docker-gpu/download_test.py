import pandas as pd
from pathlib import Path
import boto3
import os,io
import boto3
import requests


S3_ENDPOINT_URL = os.environ['S3_ENDPOINT_URL']
# print(f"using init S3_ENDPOINT_URL: {S3_ENDPOINT_URL }")
# if ":" not in "S3_ENDPOINT_URL":
#     S3_ENDPOINT_URL = f"http://{os.environ['S3_ENDPOINT_URL']}:9000"
#     os.environ['S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
print(f"using S3_ENDPOINT_URL: {S3_ENDPOINT_URL }")


os.makedirs("/home/ray/cc0foundation/cc0_testdataset", exist_ok = True)
p = Path("/home/ray/cc0foundation/")
output_path = "/home/ray/cc0foundation/cc0_testdataset/"

urls = requests.get("https://huggingface.co/api/datasets/laion/laion2B-en/parquet/default/train")
urls = urls.json()
# files = list(p.rglob("*.parquet"))
print(urls)
file = urls[0]
print(f"reading {file}")
df = pd.read_parquet(file)
df.loc[df["URL"].str.contains("wikimedia"),:]
df.to_parquet(output_path+"cc0"+str(0)+".parquet")
op = Path(output_path)

files = list(op.rglob("*.parquet"))
dflist =[]
for i,file in enumerate(files):
    df = pd.read_parquet(file, engine='fastparquet')
    df.loc[df["URL"].str.contains("wikimedia"),:]
    dflist.append(df)
dfall=  pd.concat(dflist, axis =0)
dfall.to_parquet(output_path+"cc0_all"+".parquet")


def upload_file(client, bucket,key, filepath= None):
    if filepath is None:
        fo = io.BytesIO(b'my data stored as file object in RAM')
        client.upload_fileobj(fo, bucket, key)
    else:
        client.upload_file(filepath, bucket, os.path.basename(filepath)
                           ) 
        
client = boto3.client('s3',  endpoint_url= os.environ.get("S3_ENDPOINT_URL"), 
                                       aws_access_key_id= os.environ.get("ACCESS_KEY_ID"),
                                       aws_secret_access_key= os.environ.get("SECRET_ACCESS_KEY"),
                                       region_name= os.environ.get("BUCKET_REGION", "auto"))
# try:
#     client.create_bucket(Bucket="cc0dataset")
# except:
#     print("failed bucket creation")

upload_file(client, "cc0dataset",  key="", filepath= output_path+"cc0_all"+".parquet")