#!/bin/bash
ulimit -n 8192
uv run uvicorn src.lean_pool_api:app --port 8020 &

export CUDA_VISIBLE_DEVICES=0
uv run ./scripts/vllm_serve.py --model sorgfresser/testtrainsft --tensor_parallel_size 1 &

sleep 180

export CUDA_VISIBLE_DEVICES=1,2,3
uv run accelerate launch --num-processes 3 --num_machines 1 --deepspeed_config_file /home/tss52/rds/hpc-work/RLMtrain/src/deepspeedconf.json --machine_rank "${SLURM_PROCID:-0}" /home/tss52/rds/hpc-work/RLMtrain/src/train.py --server-ip localhost "$@" &
wait
