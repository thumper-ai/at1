import torch, ray 
ray.init("auto")
# @ray.remote(num_cpus=7, num_gpus=1)
# def get_gpu_count():
#     return torch.cuda.device_count()
    
# results = [get_gpu_count.remote() for a in range(32)]
# output = ray.get(results)
# print("here are the gpus",output)

@ray.remote(num_cpus=2, num_gpus=1)
class Actor:
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
    def hasgpu(self):
        return self.gpu_count >0
    def increment(self):
        # self.value += 1
        return  torch.cuda.device_count()

    # def get_counter(self):
    #     return self.value

# Create an actor from this class.
# counter = Counter.remote()
actors = [Actor.remote() for a in range(24)]
jobs = []
for actor in actors:
    if ray.get(actor.hasgpu.remote()) > 0:
        jobs.append(actor.increment.remote())

results = ray.get(jobs)
print(results)