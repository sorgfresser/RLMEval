#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
uv run trl vllm-serve --model sorgfresser/testtrainsft --tensor_parallel_size 1 &
