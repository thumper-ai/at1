FROM  rayproject/ray-ml:nightly-py310-cu118
RUN git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    CUDA_VERSION=118 make cuda11x && \
    python setup.py bdist_wheel

EXPOSE 6380
EXPOSE 8265
COPY /startworker.py .
RUN sudo chmod 777 /home/ray/startworker.py
RUN sudo chmod a+x /home/ray/startworker.py

RUN sudo chown ray /home/ray/startworker.py
RUN sudo chmod 777 /home/ray
RUN sudo apt-get install git-lfs s3fs -y
RUN git lfs install --skip-repo
RUN pip install s3fs
RUN ls -lia /home/ray

RUN mkdir /home/ray/models
RUN sudo chmod 777 /home/ray
RUN sudo chmod 777 /home/ray/models

ENTRYPOINT [ "python3" ]
# CMD [ "ray", "start", "--port=6380", "--dashboard-port=8265", "--dashboard-host=0.0.0.0", "--object-manager-port=8076", "--node-manager-port=8077", "--dashboard-agent-grpc-port=8078" , "--dashboard-agent-listen-port=8079", "--min-worker-port=10002" , "--max-worker-port=10005", "--block" ]
CMD ["/home/ray/startworker.py"]