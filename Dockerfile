FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ADD . /app
WORKDIR /app
RUN pip3 install -e .
