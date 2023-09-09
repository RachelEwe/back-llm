FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as build

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip

RUN groupadd -g 1000 dev && \
  useradd -rm -d /home/dev -s /bin/bash -g dev -u 1000 dev

USER 1000

COPY --chown=1000 . /home/dev

WORKDIR /home/dev

RUN pip install --no-warn-script-location --upgrade pip setuptools wheel \
    && pip install --no-warn-script-location -r requirements.txt

CMD ["/usr/bin/python3","./api.py"]
