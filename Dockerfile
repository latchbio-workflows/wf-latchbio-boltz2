# Base Image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu20.04

# Latch environment building
COPY --from=812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda:fe0b-main /bin/flytectl /bin/flytectl
WORKDIR /root

ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev build-essential procps rsync openssh-server

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        python3-pip \
        python3.10-venv \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN python3.10 -m ensurepip --upgrade

RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir awscli

RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz -o s5cmd_2.0.0_Linux-64bit.tar.gz &&\
    tar -xzvf s5cmd_2.0.0_Linux-64bit.tar.gz &&\
    mv s5cmd /bin/ &&\
    rm CHANGELOG.md LICENSE README.md

COPY --from=812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda:fe0b-main /root/Makefile /root/Makefile
COPY --from=812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda:fe0b-main /root/flytekit.config /root/flytekit.config

WORKDIR /tmp/docker-build/work/

SHELL [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-o", "verbose", \
    "-o", "errtrace", \
    "-O", "inherit_errexit", \
    "-O", "shift_verbose", \
    "-c" \
]

ENV TZ='Etc/UTC'

ENV LANG='en_US.UTF-8'
ARG DEBIAN_FRONTEND=noninteractive

# Install system requirements
RUN apt-get update --yes && \
    xargs apt-get install --yes aria2 git wget unzip curl fuse python3.9-dev && \
    apt-get install --fix-broken && \
    apt-get install -y git

# ObjectiveFS
RUN curl --location --fail --remote-name https://objectivefs.com/user/download/an7dzrz65/objectivefs_7.2_amd64.deb && \
    dpkg -i objectivefs_7.2_amd64.deb && \
    mkdir /etc/objectivefs.env

COPY credentials/* /etc/objectivefs.env/

RUN apt-get install --yes pkg-config libfuse-dev

# ObjectiveFS performance tuning
ENV CACHESIZE="50Gi"
ENV DISKCACHE_SIZE="200Gi"



RUN pip3 install torch torchvision torchaudio biopython

RUN pip3 install lightning[extra]

RUN git clone --depth 1 https://github.com/jwohlwend/boltz.git /opt/boltz
RUN pip install -e /opt/boltz 
# RUN pip3 install rdkit-pypi 
# RUN pip3 install --no-cache-dir "numpy==1.24.4"  "numba==0.61.0" "llvmlite>=0.44,<0.45" einops mashumaro scipy

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential            \
        libattr1-dev               \
        python3.10-dev             \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# Latch SDK
# DO NOT REMOVE
RUN pip install --no-cache-dir --upgrade pip latch
RUN mkdir /opt/latch

# Copy workflow data (use .dockerignore to skip files)
COPY . /root/

# Latch workflow registration metadata
# DO NOT CHANGE
ARG tag
# DO NOT CHANGE
ENV FLYTE_INTERNAL_IMAGE $tag

WORKDIR /root
