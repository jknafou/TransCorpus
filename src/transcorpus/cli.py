from transcorpus import retrieval
import click


@click.group()
def cli():
    """Transcorpus CLI tool."""


cli.add_command(retrieval.download_database)
cli.add_command(retrieval.download_corpus)

if __name__ == "__main__":
    cli()
