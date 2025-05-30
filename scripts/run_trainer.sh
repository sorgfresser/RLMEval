#!/bin/bash
uv run uvicorn src.lean_pool_api:app --port 8020 &

sleep 180

uv run accelerate launch --num-processes 3 --num_machines 1 --deepspeed_config_file /home/tss52/rds/hpc-work/RLMtrain/src/deepspeedconf.json --machine_rank $SLURM_PROCID /home/tss52/rds/hpc-work/RLMtrain/src/train.py --server-ip localhost &
wait
