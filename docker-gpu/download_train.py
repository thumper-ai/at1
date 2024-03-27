from cloudpathlib import CloudPath,S3Client
import os 

print( "url", os.environ.get("S3_ENDPOINT_URL"))
print( "id", os.environ.get("AWS_ACCESS_KEY_ID"))
print( "key", os.environ.get("AWS_SECRET_ACCESS_KEY"))

client =  S3Client(endpoint_url= os.environ.get("S3_ENDPOINT_URL"), 
                                       aws_access_key_id= os.environ.get("AWS_ACCESS_KEY_ID"),
                                       aws_secret_access_key= os.environ.get("AWS_SECRET_ACCESS_KEY"), #extra_args={"endpoint_url": "http://provider.europlots.com:32270",
                  )

root_dir = client.CloudPath("s3://output")
root_dir.download_to("/home/ray/train_data")