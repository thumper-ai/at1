from ray.job_submission import JobSubmissionClient, JobStatus
import time
from ray import serve
import ray
import datetime
import uuid
# import global_config
import os
# from logger import Logger
# from config import settings

class JobScheduler:
    def __init__(self, address=f'ray://{os.environ.get("RAY_HOST", "http://127.0.0.1")}:{os.environ.get("RAY_PORT", "8265")}') -> None:
        dashboard_address = f'http://{os.environ.get("RAY_HOST", "http://127.0.0.1")}:{os.environ.get("RAY_DASHBOARD_PORT", "8265")}'
        print(f'Using ray address {address} dashboard address {dashboard_address}')
        self.client = JobSubmissionClient(dashboard_address)
        self.jobs = {}

    def schedule_job(self, config):
        # if isinstance(config, str):
        #     config = utils.load_config(config)
        print(config)
        job_id = str(uuid.uuid4())
        config["job_id"] = job_id
        job_name = config["job_name"]
        script_command = f'python diffusion/raycomposer_at.py --address http://127.0.0.1 --n 1 --use_gpu'

        print(f'existing command {script_command}')

        ray_job_id = self.client.submit_job(
            # Entrypoint shell command to execute
            entrypoint=script_command,  # 'pip3 install -r requirements.txt;'+
            # Runtime environment for the job, specifying a working directory and pip package
            # runtime_env={ "working_dir": log_dir, "pip": runtime_env },
            runtime_env = {"working_dir": 'diffusion', "pip": ["mosaicml==0.12.1"]}
            # entrypoint_num_gpus=1
        )

        self.jobs[job_id] = {
            'script_command': script_command, 'dependency': None, 'log_dir': log_dir, 'job_name': job_name}
        print(f"ray job id {ray_job_id}")
        return job_id

    def TerminateJob(self, job_id):
        rid = self.job_tracker_db.get_rayjobid(job_id)
        if rid is None:
            return 'Invalid Job id'
        self.jobs.pop(job_id)
        return self.client.stop_job(rid)

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

    def GenerateImages(self, job_id, prompts):
        with open('prompts.txt', 'w') as f:
            for line in prompts:
                f.write(f"{line}\n")
        try:
            # writer = Logger(os.path.join(
            #     global_config.S3_log_dir, job_id))
            writer = Logger(os.path.join(
               "s3://stable-diffusion-checkpoints/", job_id))
            writer.log_file('prompts.txt')
        except:
            return False

        return True


if __name__ == '__main__':
    client = JobScheduler("http://127.0.0.1")
    client.schedule_job({})