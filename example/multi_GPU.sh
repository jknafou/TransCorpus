#!/usr/bin/env zsh
# -*- mode: zsh; sh-indentation: 2; indent-tabs-mode: nil; sh-basic-offset: 2; -*-
# vim: ft=zsh sw=2 ts=2 et

# Multi-GPU Translation Pipeline
# Usage: ./multi_GPU.sh [corpus_name] [target_language] [preprocess_workers] [num_splits]

# Enable strict error handling
set -euo pipefail
trap 'cleanup' EXIT INT TERM

# Configuration
MAX_TOKENS_GPU0=24320
MAX_TOKENS_GPU1=8192
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Track background process IDs
typeset -a bg_pids

# Color definitions
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
BOLD=$(tput bold)
RESET=$(tput sgr0)

cleanup() {
  echo -e "\n${YELLOW}⚠️  Cleaning up background processes...${RESET}"
  # Kill all child process groups
  for pid in $bg_pids; do
    kill -- -$pid 2>/dev/null || true
  done
  # Wait for processes to terminate
  wait 2>/dev/null || true
}

print_header() {
  echo -e "${BOLD}${BLUE}=== Multi-GPU Translation Pipeline ===${RESET}"
  echo -e "${BOLD}Started at:${RESET} $(date)"
  echo -e "${BOLD}Log directory:${RESET} ${LOG_DIR}/${TIMESTAMP}"
  echo -e "${BOLD}Corpus:${RESET} ${1}"
  echo -e "${BOLD}Target language:${RESET} ${2}"
}

validate_input() {
  if [[ $# -lt 4 ]]; then
    echo -e "${RED}${BOLD}Error:${RESET} Missing required arguments"
    print_usage
    exit 1
  fi
}

print_usage() {
  cat <<-EOF
${BOLD}Usage:${RESET}
  $0 [corpus_name] [target_language] [preprocess_workers] [num_splits]

${BOLD}Arguments:${RESET}
  1) Corpus name (e.g., bio)
  2) Target language code (e.g., de)
  3) Number of preprocessing workers
  4) Number of splits for parallel processing

${BOLD}Example:${RESET}
  $0 bio de 4 20
EOF
}

main() {
  validate_input "$@"

  local corpus_name=$1
  local target_lang=$2
  local num_worker_preprocess=$3
  local num_splits=$4

  print_header "$corpus_name" "$target_lang"
  mkdir -p "${LOG_DIR}/${TIMESTAMP}"

  echo -e "\n${GREEN}${BOLD}[1/4] Setting up environment...${RESET}"
  UV_INDEX_STRATEGY=unsafe-best-match rye sync
  source .venv/bin/activate

  echo -e "\n${GREEN}${BOLD}[2/4] Downloading corpus...${RESET}"
  transcorpus download-corpus "${corpus_name}" -d 2>&1 | tee "${LOG_DIR}/${TIMESTAMP}/download.log"

  echo -e "\n${GREEN}${BOLD}[3/4] Starting translation workers...${RESET}"
  echo "${BLUE}${BOLD}Starting GPU 0 worker (max_tokens=${MAX_TOKENS_GPU0})${RESET}"
  CUDA_VISIBLE_DEVICES=0 transcorpus translate "${corpus_name}" "${target_lang}" -d \
    --num-splits "${num_splits}" --max-tokens "${MAX_TOKENS_GPU0}" \
    2>&1 | tee "${LOG_DIR}/${TIMESTAMP}/translate_gpu0.log" &
  bg_pids+=($!)

  sleep 10  # Stagger GPU workers

  echo "${BLUE}${BOLD}Starting GPU 1 worker (max_tokens=${MAX_TOKENS_GPU1})${RESET}"
  CUDA_VISIBLE_DEVICES=1 transcorpus translate "${corpus_name}" "${target_lang}" -d \
    --num-splits "${num_splits}" --max-tokens "${MAX_TOKENS_GPU1}" \
    2>&1 | tee "${LOG_DIR}/${TIMESTAMP}/translate_gpu1.log" &
  bg_pids+=($!)

  echo -e "\n${GREEN}${BOLD}[4/4] Starting preprocessing workers...${RESET}"
  for ((worker=0; worker<num_worker_preprocess; worker++)); do
    echo "${BLUE}${BOLD}Starting preprocess worker ${worker}${RESET}"
    transcorpus preprocess "${corpus_name}" "${target_lang}" -d --num-splits "${num_splits}" \
      2>&1 | tee "${LOG_DIR}/${TIMESTAMP}/preprocess_${worker}.log" &
    bg_pids+=($!)
  done

  echo -e "\n${YELLOW}${BOLD}Waiting for all jobs to complete...${RESET}"
  wait  # Wait for all background processes
  echo -e "\n${GREEN}${BOLD}All jobs completed successfully!${RESET}"
}

main "$@"

