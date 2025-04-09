"""
TransCorpus Initialization Module.

This module initializes the TransCorpus package by providing utility functions
for directory management. The main function, `create_transcorpus_dir`, ensures
that a `.TransCorpus` directory exists in the user's home directory.

Functions:
    - create_transcorpus_dir: Creates and returns the `.TransCorpus` directory.

Typical usage example:
    from transcorpus import create_transcorpus_dir

    transcorpus_dir = create_transcorpus_dir()
"""

from pathlib import Path


def create_transcorpus_dir() -> Path:
    """
    Create the `.TransCorpus` directory in the user's home directory if it does not exist.

    This function checks whether the `.TransCorpus` directory exists in the user's home
    directory. If it does not exist, it creates the directory and prints a message indicating
    that it was created.

    Returns:
        Path: The absolute path to the `.TransCorpus` directory.

    Example:
        >>> create_transcorpus_dir()
        Created directory: /home/user/.TransCorpus
        PosixPath('/home/user/.TransCorpus')
    """
    transcorpus_dir = Path.home() / ".TransCorpus"
    if not transcorpus_dir.exists():
        transcorpus_dir.mkdir(parents=True)
        print(f"Created directory: {transcorpus_dir}")
    return transcorpus_dir
