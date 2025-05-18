#!/bin/bash

uv run accelerate launch --num-processes 4 --num_machines 1 --deepspeed_config_file /home/tss52/rds/hpc-work/RLMEval/src/deepspeed.conf --machine_rank $SLURM_PROCID /home/tss52/rds/hpc-work/RLMEval/src/train.py --server-ip $VLLM_NODE &
uv run uvicorn src.lean_pool_api:app --port 8080 &
wait
