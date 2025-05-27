#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
uv run ./scripts/vllm_serve.py --model sorgfresser/testtrainsft --tensor_parallel_size 1
