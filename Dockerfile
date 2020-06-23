FROM python:3.7-buster

COPY . /root

WORKDIR /root

RUN pip install -r requirements.txt