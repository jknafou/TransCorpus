# fairseq_generate_wrapper.py
import torch
import argparse
from fairseq_cli.generate import cli_main

if __name__ == "__main__":
    # Allow argparse.Namespace for PyTorch 2.6+
    torch.serialization.add_safe_globals([argparse.Namespace])
    cli_main()
