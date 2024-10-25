FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# apt packages
RUN apt-get update && apt-get install curl wget git pip vim tmux htop libxml2 kmod systemctl lsof python3.10 -y
RUN pip install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /root

# Python library
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers accelerate sentencepiece tokenizers texttable toml attributedict protobuf cchardet
RUN pip install matplotlib scikit-learn pandas

# Flush
RUN rm -rf /root/.cache/pip
