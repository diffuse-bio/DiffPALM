FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y wget && apt-get install -y git build-essential libssl-dev

# install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install pytorch==2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge

# installing google cloud SDK
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-386.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-386.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q



ARG username=$GIT_USERNAME
ARG password=$GIT_PASSWORD

WORKDIR /app

RUN git clone --single-branch -b si_2311_docker https://username:password@github.com/diffuse-bio/DiffPALM.git
#RUN git clone https://username:password@github.com/diffuse-bio/DiffPALM.git
RUN python -m pip install fair-esm@git+https://github.com/Bitbol-Lab/esm.git@oh_input#egg=fair-es

WORKDIR /app/DiffPALM

RUN python -m pip install .

ENTRYPOINT ["python", "run_diffPalm.py", "/app/data/6L5K_1.paralogs_processed.fasta", "/app/data/6L5K_2.paralogs_processed.fasta"]
~                                                                                                                                            