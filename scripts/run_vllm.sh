#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
uv run trl vllm-serve --model AI-MO/Kimina-Autoformalizer-7B --tensor_parallel_size 1 &
