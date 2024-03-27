from pathlib import Path
import ray 
import numpy as np

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
