from transcorpus.retrieval import download_file
import click


@click.group()
def cli():
    """Transcorpus CLI tool."""


@cli.command()
@click.argument("database_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_database(database_name, demo):
    """
    Download a database by name.

    DATABASE_NAME: Name of the database to download.
    """
    click.echo(f"Downloading database: {database_name, demo}")
    # Add logic for downloading the database here.


@cli.command()
@click.argument("corpus_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_corpus(corpus_name, demo):
    """
    Download a corpus by name.

    CORPUS_NAME: Name of the corpus to download.
    """
    corpus_urls = {
        "bio": {
            "files": ["PMID.txt", "title_abstract_en.txt"],
            "files_demo": ["PMID_sample.txt", "title_abstract_en_sample.txt"],
            "endpoint": "https://transcorpus.s3.text-analytics.ch/",
        }
    }

    if corpus_name not in corpus_urls:
        raise ValueError(f"Unknown corpus name: {corpus_name}")

    if demo and "files_demo" not in corpus_urls[corpus_name]:
        raise ValueError(f"Demo files not available for corpus: {corpus_name}")
    files = "files_demo" if demo else "files"

    urls = [
        corpus_urls[corpus_name]["endpoint"] + f
        for f in corpus_urls[corpus_name][files]
    ]

    for url in urls:
        download_file(url, "data/corpus/")


if __name__ == "__main__":
    cli()
