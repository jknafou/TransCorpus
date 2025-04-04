from transcorpus.data_urls import data_urls
from transcorpus import transcorpus_dir
import click
import subprocess
import os


def download_file(url, directory):
    file_name = os.path.join(directory, os.path.basename(url))

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"File already downloaded: {file_name}")
        return

    try:
        subprocess.run(["wget", url, "-P", directory], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")


def download_data(data_type, domain_name, demo):
    """
    Download corpus or database from a specific domain.

    """
    if domain_name not in data_urls:
        raise ValueError(f"Unknown domain name: {domain_name}")

    if demo and data_type + "_demo" not in data_urls[domain_name]:
        raise ValueError(f"No {data_type} demos available for: {domain_name}")

    url = (
        data_urls[domain_name]["endpoint"]
        + data_urls[domain_name][f"{data_type}_demo" if demo else data_type]
    )

    download_file(url, transcorpus_dir + "/" + domain_name)


@click.command()
@click.argument("corpus_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_corpus(corpus_name, demo):
    download_data("corpus", corpus_name, demo)


@click.command()
@click.argument("database_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_database(database_name, demo):
    download_data("database", database_name, demo)
