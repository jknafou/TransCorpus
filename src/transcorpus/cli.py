"""
TransCorpus Command-Line Interface (CLI).

This module provides the CLI entry point for the TransCorpus package. It
organizes and exposes commands for downloading databases, corpora, and corpus
IDs using the `retrieval` module.

Commands:
    - download_database: Download a database for a specific domain.
    - download_corpus: Download a corpus for a specific domain.
    - download_corpus_id: Download corpus IDs for a specific domain.

Typical usage example:
    $ transcorpus download_database bio --demo
    $ transcorpus download_corpus bio
    $ transcorpus download_corpus_id bio --demo
"""

import click

from transcorpus import preview, retrieval, translate


@click.group()
def cli():
    """
    Transcorpus CLI tool.

    This is the main entry point for the TransCorpus command-line interface. It
    groups all available commands (e.g., downloading databases, corpora, and
    IDs) into a single CLI tool.

    Example:
        $ python cli.py --help
        Usage: cli.py [OPTIONS] COMMAND [ARGS]...

        Options:
          --help  Show this message and exit.

        Commands:
          download_database  Download a database for a specific domain.
          download_corpus    Download a corpus for a specific domain.
          download_corpus_id Download corpus IDs for a specific domain.
    """


cli.add_command(retrieval.download_database)
cli.add_command(retrieval.download_corpus)

cli.add_command(preview.preview)

cli.add_command(translate.translate)

if __name__ == "__main__":
    cli()
