FROM python:3.8

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y
RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -U numpy
RUN pip install tensorboardx gym matplotlib pyyaml opencv-python
RUN curl https://raw.githubusercontent.com/TeaganLi/HouseExpo/refs/heads/master/HouseExpo/json.tar.gz > HouseExpo.tar.gz && tar -zvxf HouseExpo.tar.gz && mkdir HouseExpo && mv json HouseExpo/json
