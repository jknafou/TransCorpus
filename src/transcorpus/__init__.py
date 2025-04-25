"""
Initialize the TransCorpus project.

This script initializes the TransCorpus project by importing the necessary
utility function and creating the required directory structure.

Modules:
    transcorpus.utils: Provides utility functions for the TransCorpus project.

Functions:
    create_transcorpus_dir: A utility function that creates the TransCorpus
    directory.

Execution:
    When run as the main script, this module invokes the
    `create_transcorpus_dir` function to ensure the required directory structure
    is set up.
"""

from transcorpus.utils import create_transcorpus_dir

if __name__ == "__main__":
    # If this script is run directly, create the TransCorpus directory.
    create_transcorpus_dir()
