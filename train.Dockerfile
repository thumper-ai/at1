FROM  rayproject/ray-ml:nightly-py39-cu118
# gpu
# cu118
EXPOSE 6380
EXPOSE 8265
# RUN sudo apt install -y iotop
# RUN pip install mosaicml-streaming pyarrow pandas boto3 fastparquet cloudpathlib[s3] mosaicml==0.15.1 hydra-core>=1.2 hydra-colorlog>=1.1.0 diffusers[torch]==0.19.3 transformers[torch]==4.31.0 wandb==0.15.4 xformers==0.0.21 torchmetrics[image]==0.11.4 clean-fid clip@git+https://github.com/openai/CLIP.git 
COPY /requirements.txt .
RUN conda create -n pixart python==3.9.0 -y
RUN source ~/anaconda3/etc/profile.d/conda.sh && conda activate pixart && conda config --add channels conda-forge && conda install s5cmd && pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117  && pip install -r requirements.txt  && pip install s3fs b2 cloudpathlib[s3] && pip install -U "ray[default,train] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"
RUN pip install -U "ray[default,train] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"
COPY /docker-gpu/download_metadata.py .
COPY /docker-gpu/download_test.py .
# COPY /docker-gpu/laion_cloudwriter.py .
# COPY /docker-gpu/starthead.py .
COPY /docker-gpu/starthead.sh .

COPY /docker-gpu/test_upload.py .
# COPY /download_train.py .
COPY /i2dataset.py .



# COPY ./diffusion/ ./diffusion/
COPY ./LLaVA/ ./LLaVA/
COPY ./PixArt-alpha/ ./PixArt-alpha/
RUN sudo chmod 777 /home/ray/LLaVA
RUN sudo chmod a+x /home/ray/LLaVA
RUN sudo chown ray /home/ray/LLaVA
RUN sudo chmod 777 /home/ray/PixArt-alpha
RUN sudo chmod a+x /home/ray/PixArt-alpha
RUN sudo chown ray /home/ray/PixArt-alpha/


# RUN cd ./LLaVA && conda create -n llava python=3.10 -y 
# RUN cd ./LLaVA && source activate llava && pip install --upgrade pip && pip install -e .

# RUN conda update conda

# RUN source activate llava
# ENV PATH /home/ray/anaconda3/bin:$PATH
# RUN conda info --all  | grep -i python
# RUN conda env list

# RUN conda create -n pixart python==3.9.0
# RUN source activate pixart && cd ./PixArt-alpha &&  pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 && pip install -r requirements.txt
# RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
# 
COPY /run.py .
COPY /launch.py .

COPY /run_eval.py .
COPY /i2dataset.py .
COPY /download_train.py .
COPY /install_conda_deps.sh .

RUN mkdir /home/ray/test_data
RUN mkdir /home/ray/train_data
RUN mkdir /home/ray/checkpoints
RUN mkdir /home/ray/models

# RUN sudo chmod 777 /home/ray/laion_cloudwriter.py
# RUN sudo chmod a+x /home/ray/laion_cloudwriter.py
# RUN sudo chown ray /home/ray/laion_cloudwriter.py
RUN sudo chmod 777 install_conda_deps.sh
RUN sudo chmod a+x install_conda_deps.sh
RUN sudo chown ray install_conda_deps.sh

RUN sudo chmod 777 /home/ray/download_metadata.py
RUN sudo chmod a+x /home/ray/download_metadata.py
RUN sudo chown ray /home/ray/download_metadata.py
RUN sudo chmod 777 /home/ray/download_test.py
RUN sudo chmod a+x /home/ray/download_test.py
RUN sudo chown ray /home/ray/download_test.py
RUN sudo chmod 777 /home/ray/starthead.sh
RUN sudo chmod a+x /home/ray/starthead.sh
RUN sudo chown ray /home/ray/starthead.sh
RUN sudo chmod 777 /home/ray/test_upload.py
RUN sudo chmod a+x /home/ray/test_upload.py
RUN sudo chown ray /home/ray/test_upload.py
RUN sudo chmod 777 /home/ray/run.py
RUN sudo chmod a+x /home/ray/run.py
RUN sudo chown ray /home/ray/run.py
RUN sudo chmod 777 /home/ray/launch.py
RUN sudo chmod a+x /home/ray/launch.py
RUN sudo chown ray /home/ray/launch.py
RUN sudo chmod 777 /home/ray/download_train.py
RUN sudo chmod a+x /home/ray/download_train.py
RUN sudo chown ray /home/ray/download_train.py

RUN sudo chmod 777 /home/ray
RUN sudo chmod 777 /home/ray/test_data
RUN sudo chmod 777 /home/ray/train_data
RUN sudo chown ray /home/ray/test_data
RUN sudo chown ray /home/ray/train_data

RUN sudo chown ray /home/ray/checkpoints
RUN sudo chown ray /home/ray/checkpoints

RUn sudo apt-get install git-lfs
RUN pip install ray[train,default]
RUN sudo apt install -y s3fs
RUN pip install s3fs
RUN pip install --upgrade pip
# RUN pip install git+https://github.com/thumper-ai/LLaVA.git

# CMD [ "ray", "start", "--head",  "--port=6380", "--dashboard-port=8265", "--dashboard-host=0.0.0.0", "--object-manager-port=8076", "--node-manager-port=8077", "--dashboard-agent-grpc-port=8078" , "--dashboard-agent-listen-port=8079", "--min-worker-port=10002" , "--max-worker-port=10005", "--block" ]

#docker build -t thumperai/rayakash:nighly-gpu-17 .
# ENTRYPOINT [ "python3" ]
# CMD [ "ray", "start", "--port=6380", "--dashboard-port=8265", "--dashboard-host=0.0.0.0", "--object-manager-port=8076", "--node-manager-port=8077", "--dashboard-agent-grpc-port=8078" , "--dashboard-agent-listen-port=8079", "--min-worker-port=10002" , "--max-worker-port=10005", "--block" ]
# CMD ["starthead.sh"]
# ENTRYPOINT ["/bin/sh -c"]
ENTRYPOINT ["/bin/bash"]
CMD ["/home/ray/starthead.sh"]

#composer run.py --config-path hydra-yamls --config-name update.yaml 