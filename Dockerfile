FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN apt update && apt install ffmpeg libsm6 libxext6  -y
RUN pip install git+https://github.com/the-sauer/blob-matcher
