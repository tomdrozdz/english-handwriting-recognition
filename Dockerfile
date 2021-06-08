# syntax=docker/dockerfile:1

FROM python:3.9.5-buster

WORKDIR /service

RUN apt-get update
RUN apt-get install libgl1-mesa-glx libsm6 libxext6 build-essential -y

COPY requirements.txt requirements.txt

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt

COPY . .

ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:3000 --workers=1 --chdir /service/src"

EXPOSE 3000
CMD ["gunicorn", "server:app"]
