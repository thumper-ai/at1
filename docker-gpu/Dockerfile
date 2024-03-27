FROM  rayproject/ray-ml:nightly-py310-cu118
EXPOSE 6380
EXPOSE 8265

RUN pip install mosaicml-streaming pyarrow pandas boto3 fastparquet cloudpathlib[s3] mosaicml==0.15.1 hydra-core>=1.2 hydra-colorlog>=1.1.0 diffusers[torch]==0.19.3 transformers[torch]==4.31.0 wandb==0.15.4 xformers==0.0.21 torchmetrics[image]==0.11.4 clean-fid clip@git+https://github.com/openai/CLIP.git tensorflow===2.13.0

COPY /download_metadata.py .
COPY /download_test.py .
COPY /laion_cloudwriter.py .
COPY /starthead.py .
COPY /starthead.sh .

COPY /test_upload.py .
COPY /download_train.py .

COPY ./diffusion/ ./diffusion/
COPY ./yamls/ .

COPY /run.py .
COPY /run_eval.py .

RUN mkdir /home/ray/test_data
RUN mkdir /home/ray/train_data
RUN mkdir /home/ray/checkpoints

# RUN sudo chmod 777 /home/ray/laion_cloudwriter.py
# RUN sudo chmod a+x /home/ray/laion_cloudwriter.py
# RUN sudo chown ray /home/ray/laion_cloudwriter.py
RUN sudo chmod 777 /home/ray/download_metadata.py
RUN sudo chmod a+x /home/ray/download_metadata.py
RUN sudo chown ray /home/ray/download_metadata.py
RUN sudo chmod 777 /home/ray/download_test.py
RUN sudo chmod a+x /home/ray/download_test.py
RUN sudo chown ray /home/ray/download_test.py
RUN sudo chmod 777 /home/ray/starthead.py
RUN sudo chmod a+x /home/ray/starthead.py
RUN sudo chown ray /home/ray/starthead.py

RUN sudo chmod 777 /home/ray/starthead.sh
RUN sudo chmod a+x /home/ray/starthead.sh
RUN sudo chown ray /home/ray/starthead.sh

RUN sudo chmod 777 /home/ray/test_upload.py
RUN sudo chmod a+x /home/ray/test_upload.py
RUN sudo chown ray /home/ray/test_upload.py
RUN sudo chmod 777 /home/ray/run.py
RUN sudo chmod a+x /home/ray/run.py
RUN sudo chown ray /home/ray/run.py
RUN sudo chmod 777 /home/ray/download_train.py
RUN sudo chmod a+x /home/ray/download_train.py
RUN sudo chown ray /home/ray/download_train.py

RUN sudo chmod 777 /home/ray
RUN sudo chmod 777 /home/ray/test_data
RUN sudo chmod 777 /home/ray/train_data
RUN sudo chown ray /home/ray/test_data
RUN sudo chown ray /home/ray/train_data

RUN sudo chown ray /home/ray/checkpoints
# RUN sudo chown ray /home/ray/checkpoints
# CMD [ "ray", "start", "--head",  "--port=6380", "--dashboard-port=8265", "--dashboard-host=0.0.0.0", "--object-manager-port=8076", "--node-manager-port=8077", "--dashboard-agent-grpc-port=8078" , "--dashboard-agent-listen-port=8079", "--min-worker-port=10002" , "--max-worker-port=10005", "--block" ]

#docker build -t thumperai/rayakash:nighly-gpu-17 .
# ENTRYPOINT [ "python3" ]
ENTRYPOINT ["/bin/sh -c"]
# CMD [ "ray", "start", "--port=6380", "--dashboard-port=8265", "--dashboard-host=0.0.0.0", "--object-manager-port=8076", "--node-manager-port=8077", "--dashboard-agent-grpc-port=8078" , "--dashboard-agent-listen-port=8079", "--min-worker-port=10002" , "--max-worker-port=10005", "--block" ]
# CMD ["/home/ray/starthead.py"]
CMD ["starthead.sh"]

#composer run.py --config-path hydra-yamls --config-name update.yaml 