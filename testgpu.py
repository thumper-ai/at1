import os 
import ray 

ray.init("auto")

@ray.remote(num_gpus=1, num_cpus=6)
def testgpu():
    os.system("nvidia-smi ")


jobs = [testgpu.remote() for a in range(32)]
ray.get(jobs)