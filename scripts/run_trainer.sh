#!/bin/bash
uv run uvicorn src.lean_pool_api:app --port 8020 &

sleep 180

export CUDA_VISIBLE_DEVICES=0,1,2
uv run accelerate launch --num-processes 3 --num_machines 1 --deepspeed_config_file /home/tss52/rds/hpc-work/RLMtrain/src/deepspeed.conf --machine_rank $SLURM_PROCID /home/tss52/rds/hpc-work/RLMtrain/src/train.py --server-ip localhost &
wait
