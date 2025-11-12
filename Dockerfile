FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y   # OpenCV Dependencies
RUN apt install git -y

RUN pip install git+https://github.com/the-sauer/blob-matcher


