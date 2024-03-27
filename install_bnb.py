from pathlib import Path
import ray 
import os

@ray.remote(num_gpus=6)
def install_bnb():
    # os.system("pip uninstall bitsandbytes -y && pip install bitsandbytes")
    os.system("python -m bitsandbytes")
    print(f"success")
    return 'success'
    # return len(images)
ray.init("auto")
results= ray.get([install_bnb.remote() for a in range(4)])
print(f"results {results} ")
# RUN pip3 uninstall -y bitsandbytes && \
#     cd /opt && \
#     git clone --depth=1 https://github.com/${BITSANDBYTES_REPO} bitsandbytes && \
#     cd bitsandbytes && \
#     CUDA_VERSION=114 make -j$(nproc) cuda11x && \
#     CUDA_VERSION=114 make -j$(nproc) cuda11x_nomatmul && \
#     python3 setup.py --verbose build_ext --inplace -j$(nproc) bdist_wheel && \
#     cp dist/bitsandbytes*.whl /opt && \
#     pip3 install --no-cache-dir --verbose /opt/bitsandbytes*.whl  && \
#     cd ../ && \
#     rm -rf bitsandbytes