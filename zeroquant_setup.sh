#!/bin/bash
pip install deepspeed==0.16.0
apt install libopenmpi-dev
pip install mpi4py==4.0.1

# Set environment variables for DeepSpeed
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500