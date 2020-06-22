FROM python:3.7-slim

COPY . /root

WORKDIR /root

RUN pip install flask gunicorn requests torch opencv_python torchvision