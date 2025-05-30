#!/bin/bash
uv run ./scripts/vllm_serve.py --model sorgfresser/testtrainsft --tensor_parallel_size 1
