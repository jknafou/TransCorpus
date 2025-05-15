#!bin/zsh

# This script demonstrates how to use multiple GPUs for translation with the transcorpus CLI.
# From root -> zsh example/multi_GPU.sh

UV_INDEX_STRATEGY=unsafe-best-match rye sync

source .venv/bin/activate

transcorpus download-corpus bio
CUDA_VISIBLE_DEVICES=0 transcorpus translate bio -t es --num-splits 100 --max-tokens 8192 &
sleep 5
CUDA_VISIBLE_DEVICES=1 transcorpus translate bio -t es --num-splits 100 --max-tokens 2048 &

wait
