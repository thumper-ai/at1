import sys
import time
from collections import Counter
import argparse
# import boto3
import ray
from ray.runtime_env import RuntimeEnv
# from img2dataset import download
# import img2dataset
import os
import pandas as pd
from ray.job_submission import JobSubmissionClient, JobStatus
import time
# from train import GetTrainConfigurations, GetTrainConfigurations3, GetInferenceConfiguration, GetControlNetConfiguration
# from inference import create_inference_endpoint
# import utils
import ray
# from logger import job_status_db
import datetime
import uuid
# import global_config
import os
import shutil

# from logger import Logger
# from config import settings
# from logger import eta_loss_db

class JobScheduler:
    def __init__(self, address=f'ray://{os.environ.get("RAY_HOST", "http://127.0.0.1")}:{os.environ.get("RAY_PORT", "8265")}',
                    dashboard_address = f'http://{os.environ.get("RAY_HOST", "http://127.0.0.1")}:{os.environ.get("RAY_DASHBOARD_PORT", "8265")}'
                 ) -> None:
        # if not ray.is_initialized():
        #     print(f'initializing sing ray address {address} dashboard address {dashboard_address}')
        #     ray.init(address=f'ray://{address}')
        print(f'Using ray address {address} dashboard address {dashboard_address}')
        self.client = JobSubmissionClient(dashboard_address)
        self.jobs = {}

    def schedule_job(self, config, args):
        import shutil
        # config["job_id"] = job_id
        job_name = config["job_name"]
        scriptcmd = config['cmd']
        scriptfile = scriptcmd.split(" ")[1]

        log_dir = f'output/logs/{job_name}/'
        if args.env=="pixart":
            print("copying tree ")
            shutil.copytree('PixArt-alpha/', log_dir+'/')
        else:
            try:
                os.mkdir(log_dir, exists_ok=True)
                os.mkdir(log_dir+'results/')
            except:
                print('dir already exists')
            # log_dir = log_dir = f'/opt/logs/foundation/'
            if "python -m" not in  scriptcmd:
                shutil.copy(scriptfile, log_dir)


        # script_command = f"python raycomposer_at.py"
        # script_command = f"python mosaic_cifar10.py --address 127.0.0.1:8265 -n 1 --use-gpu"

        # runtime_env = "training_scripts/khoya_minimal/requirements_kohya.txt"
        
        print(f'existing command {scriptcmd}')

        if args.env !="pixart":
            runtime_env= { "working_dir": log_dir,
                    #  "pip": ["git+https://github.com/rakataprime/at1.git@main#egg=llava&subdirectory=LLaVA", "setuptools", "git+https://github.com/fsspec/s3fs"
                            
                            #  "numpy","pandas", "fastparquet", "b2", "bs4", "fastapi", "seaborn", "matplotlib", #"torch==2.0.1", 
                            #  'git+https://github.com/fsspec/s3fs' ,
                            #    "shortuuid"
                                #    ]
                    "conda": {"dependencies": [ "pip", {"pip": [
                    "git+https://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@github.com/rakataprime/at1.git@main#egg=llava&subdirectory=LLaVA" , "setuptools", "git+https://github.com/fsspec/s3fs"
                    ,"pandas", "fastparquet", "numpy" ,
                    'b2',
                    # 'git+https://github.com/fsspec/s3fs', 
                    'fsspec', #'s3fs',
                    # "git+https://github.com/rom1504/img2dataset",
                    "s3transfer", #"fsspec[full]==2023.10.0",
                    # "fastapi", 
                    "boto3", 'bs4',
                    # "bs4",
                    "shortuuid",
                    # "uvicorn", 
                    # "img2dataset",
                    "more-itertools"
                    ]}]}
                    }# "git+https://github.co
        else:
            runtime_env= { "config":{"setup_timeout_seconds":3600}, "working_dir": log_dir,
                # "conda": {"channels":["conda-forge"], "dependencies": [ "s5cmd", "s3fs", 'pandas', 'fastparquet', 'b2'  # "pytorch" "torchvision" "torchaudio","pytorch-cuda=11.8",
                # "pip" , {"pip": [
                # # "git+https://github.com/rakataprime/at1.git@main#egg=pixart-alpha&subdirectory=PixArt-alpha" , 
                # "setuptools", #"git+https://github.com/fsspec/s3fs",
                
                # #   "numpy" ,
                # # 'fsspec', #'s3fs',
                # # "git+https://github.com/rom1504/img2dataset",
                # # "s3transfer", #"fsspec[full]==2023.10.0",
                # # "fastapi", 
                # "boto3", 'bs4',
                # # "bs4",
                # "shortuuid",
                # # "uvicorn", 
                # # "img2dataset",
                # "more-itertools", "timm==0.6.12", "diffusers", "accelerate", "mmcv==1.7.0", "optimum",
                # "tensorboard","transformers==4.26.1", "sentencepiece", 
                # "ftfy>=6.1.1", "beautifulsoup4", "opencv-python","einops", "cloudpathlib[s3]","xformers",
                #  "--index-url https://download.pytorch.org/whl/cu118", "torch", "--index-url https://download.pytorch.org/whl/cu118", "torchvision",
                # ]}]}}
                "pip": [ "pandas", "fastparquet", "b2", "bs4", "fastapi", "uvicorn", "torch==2.0.1", "boto3", "s3fs" , "shortuuid" , "tensorboard", "transformers==4.26.1", "sentencepiece", "ftfy>=6.1.1", "beautifulsoup4", "opencv-python","einops", "cloudpathlib[s3]","xformers",
                        "more-itertools", "timm==0.6.12", "diffusers", "accelerate", "mmcv==1.7.0", "optimum",]}
                # "tensorboard","transformers==4.26.1", "sentencepiece", 
                # "ftfy>=6.1.1", "beautifulsoup4", "opencv-python","einops", "cloudpathlib[s3]","xformers", ]}

                    # "pip": ["git+https://github.com/rakataprime/at1.git@main#egg=llava&subdirectory=LLaVA", "pandas", "fastparquet", "b2", "bs4", "fastapi", "torch==2.0.1", "boto3", "s3fs" , "shortuuid"  ]
        if args.entrypoint_num_cpus >0 & args.entrypoint_memory >0 :
            if args.entrypoint_num_gpus> 0:
              ray_job_id = self.client.submit_job(
                    entrypoint=scriptcmd,  # 'pip3 install -r requirements.txt;'+
                    entrypoint_num_cpus= max(2, args.entrypoint_num_cpus),
                    entrypoint_num_gpus= args.entrypoint_num_gpus,
                    # entrypoint_memory= args.entrypoint_memory,
                    runtime_env=runtime_env
                )
            else:
                ray_job_id = self.client.submit_job(
                    entrypoint=scriptcmd,  # 'pip3 install -r requirements.txt;'+
                    entrypoint_num_cpus= args.entrypoint_num_cpus,
                    entrypoint_memory= args.entrypoint_memory,
                    runtime_env=runtime_env
                )
        else:
            ray_job_id = self.client.submit_job(
                entrypoint=scriptcmd,  # 'pip3 install -r requirements.txt;'+
                runtime_env=runtime_env
            )
        #    bitsandbytes 0.41.2
        
        print(ray_job_id)

    def TerminateJob(self, job_id):
        # rid = self.job_tracker_db.get_rayjobid(job_id)
        # if rid is None:
        #     return 'Invalid Job id'
        # self.jobs.pop(job_id)
        return self.client.stop_job(job_id)

    def GetJobStatus(self, job_id):
        rid = self.job_tracker_db.get_rayjobid(job_id)
        if rid is None:
            return 'Invalid Job id'
        return self.client.get_job_status(job_id)

    def MonitorJob(self, job_id):
        rid = self.job_tracker_db.get_rayjobid(job_id)
        if rid is None:
            return 'Invalid Job id'

        metrics = self.job_tracker_db.read_metrics(job_id)
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ray-address",  default='https://provider.akash-ai.com:31506', type=str)
    # parser.add_argument("--dash-address" ,  default='https://provider.akash-ai.com:31610', type=str)
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:30808', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:30311', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:31856', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:30728', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:30516', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:31816', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:31930', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:32446', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:32481', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:31917', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:30140', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:32754', type=str) #8265 default port
    # parser.add_argument("--ray-address",  default='provider.akash-ai.com:31970', type=str)
    # parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:31031', type=str) #8265 default port
    parser.add_argument("--ray-address",  default='provider.akash-ai.com:30749', type=str)
    parser.add_argument("--dash-address" ,  default='http://provider.akash-ai.com:32576', type=str) #8265 default port

    parser.add_argument("--job", default='submit', type=str)
    parser.add_argument("--jobname", default='test', type=str)
    parser.add_argument("--cmd", default='python i2dataset.py --local --smoketest', type=str)
    parser.add_argument("--env", default='llava', type=str)
    parser.add_argument("--entrypoint_num_cpus", default=-1, type=int)
    parser.add_argument("--entrypoint_num_gpus", default=-1, type=int)

    parser.add_argument("--entrypoint_memory", default=-1, type=int)

    args = parser.parse_args()
    print(args)
    # ray.init(address=f"ray://{args.ray_address}")

    scheduler = JobScheduler(address=args.ray_address, dashboard_address=args.dash_address)
    scheduler.schedule_job(config={
        "job_name":  args.jobname, #"test",
        "cmd" :  args.cmd,#"python3 i2dataset.py --smoketest "
        "env": args.env,
    },args= args)

    #  python3 launch.py --jobname create_dataset --cmd 'python i2dataset.py'

    # python3 launch.py --jobname createdataset --cmd 'python i2dataset.py'
    # python3 launch.py --jobname createdataset --cmd 'python i2dataset.py'

    # python3 launch.py --jobname createdataset --entrypoint_num_cpus 40 --entrypoint_memory 40000 --cmd 'python i2dataset.py --local --n_processes 4'

    # python -m llava.serve.cli_batch --model-path '/mnt/sabrent/llava-v1.5-7b' \ 
# --load-4bit \
# --max-new-tokens 128 \
# --user-prompt 'describe this image in a highly detailed manner' \
# --image-folder '/home/ray/train_data'  \
# --output-csv '/home/ray/train_data'

# python3 launch.py --jobname caption --cmd 'python -m llava.serve.cli_batch --model-path "/home/ray/train_data/llava-v1.5-7b/" --load-4bit --max-new-tokens 128 --user-prompt "describe this image in a highly detailed manner" --image-folder "/home/ray/train_data" # --output-csv "/home/ray/train_data"'
 


#  python3 launch.py --jobname caption --cmd 'python -m llava.serve.cli_batch --model-path "/home/ray/models/llava-v1.5-7b/" --max-new-tokens 128 --user-prompt "describe this image in a highly detailed manner" --image-folder "/home/ray/train_data" --output-csv "/home/ray/train_data/"'

#  python3 launch.py --jobname download --cmd 'python download_models.py'

	# "s3://akash-thumper-v1-training-images.s3.us-east-005.backblazeb2.com"

    # python3 launch.py --jobname createdataset --entrypoint_num_cpus 40 --entrypoint_memory 40000 --cmd 'python i2dataset.py --local --n_processes 4 --out_folder s3://akash-thumper-v1-training-images.s3.us-east-005.backblazeb2.com'

    # python3 launch.py --jobname createdataset --cmd 'pyt2dataset.py -n_processes 4 --out_folder s3://akash-thumper-v1-training-images.s3.us-east-005.backblazeb2.cohon im'

# python3 launch.py --jobname createdataset --entrypoint_num_cpus 140 --jobname createdataset --cmd 'b2 sync b2://akash-thumper-v1-training-images /home/ray/train_data/ --skipNewer --threads 148'

    # python3 launch.py --jobname createdataset --entrypoint_num_cpus 140 --entrypoint_memory 40000 --cmd 'python i2dataset.py --local --n_processes 10 --out_folder /home/ray/s3fs-bucket1'

# python i2dataset.py --local --n_processes 120 --out_folder /home/ray/training_data