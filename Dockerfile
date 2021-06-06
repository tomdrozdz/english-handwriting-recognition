# syntax=docker/dockerfile:1

FROM python:3.9.5-buster

WORKDIR /service

RUN apt-get update
RUN apt-get install libgl1-mesa-glx libsm6 libxext6 -y

COPY requirements.txt requirements.txt

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt

COPY *.py .
COPY models/accuracy_best.pth models/main.pth
COPY static/ static/

EXPOSE 80
CMD ["python3", "server.py"]
