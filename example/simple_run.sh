#!/bin/bash
#SBATCH --job-name=TransCorpus
#SBATCH --error=TransCorpus-error.e%j
#SBATCH --output=TransCorpus-out.o%j
#SBATCH --ntasks=1
#SBATCH --partition=private-ruch-gpu
#SBATCH --gpus=3
#SBATCH --exclusive
#SBATCH --time=7-00:00:00

set -e  # Abort script on any error

ml GCCcore/14.2.0
ml CUDA/12.3.0

rm -f requirement*

UV_INDEX_STRATEGY=unsafe-best-match rye sync

if [ ! -d .venv ]; then
  echo "Virtual environment .venv not found!" >&2
  exit 1
fi

source .venv/bin/activate

for gpu in 0 1 2; do
  echo "Launching translation on GPU $gpu"
  nohup env CUDA_VISIBLE_DEVICES=$gpu transcorpus translate bio es --max-tokens 35000 --num-splits 1000 &> nohup_gpu${gpu}.out &
  sleep 5
done

wait  # Wait for all background jobs to finish


