"""TransCorpus Retrieval Module.

This module provides functionality for downloading files and data from specific
domains. It includes utilities to handle file downloads, domain configurations,
and CLI commands for downloading different types of data (e.g., corpus, IDs,
databases).

Functions:
- download_file: Downloads a file from a given URL to a specified directory.
- download_data: Downloads corpus, IDs, or databases from a specific domain.
- download_corpus: CLI command to download a corpus.
- download_database: CLI command to download a database.

Classes:
- SuffixModel: A helper model to determine the file suffix based on a flag.

Typical usage example:
$ python retrieval.py download_corpus bio --demo
"""

from pathlib import Path
from typing import Optional
import os, time

import click
import requests
from pydantic import BaseModel, HttpUrl
from tqdm.auto import tqdm

from transcorpus.models import DataType, FileSuffix
from transcorpus.utils import get_domain_url, url_dir2path


# def download_file(url: HttpUrl, directory: Path, v: bool = True) -> Path:
#     """Download a file from the specified URL to the given directory.
#
#     This function downloads a file from the provided URL and saves it in the
#     specified directory. If the file already exists, it skips the download.
#     The function also handles errors during the download process, such as
#     network issues or user interruptions.
#
#         Args:
#             url (HttpUrl): The URL of the file to be downloaded.
#             directory (Path): The directory where the file should be saved.
#
#         Returns:
#             Optional[Path]: The path to the downloaded file if successful, or
#             None if the download failed.
#
#         Raises:
#             KeyboardInterrupt: If the user interrupts the download process.
#
#         Example:
#             >>> from pathlib import Path
#             >>> download_file("http://example.com/file", Path("/tmp"))
#             Downloaded: /tmp/file
#             PosixPath('/tmp/file')
#     """
#     file_path, file_name = url_dir2path(url, directory)
#     if file_path.exists():
#         if v:
#             print(f"File already downloaded: {file_name}")
#         return file_path
#     try:
#         response = requests.get(str(url), stream=True, timeout=10)
#         response.raise_for_status()
#         total_size = int(response.headers.get("Content-Length", 0))
#         with open(file_path, "wb") as f:
#             with tqdm(
#                 total=total_size, unit="B", unit_scale=True, desc=file_name
#             ) as pbar:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#                     pbar.update(len(chunk))
#         print(f"Downloaded: {file_path}")
#         return file_path
#     except requests.exceptions.RequestException as e:
#         if file_path.exists():
#             file_path.unlink()
#         raise click.ClickException(f"Failed to download {file_name}: {e}")
#     except KeyboardInterrupt:
#         if file_path.exists():
#             file_path.unlink()
#         raise click.ClickException("Download interrupted by user.")


def download_file(
    url: HttpUrl, directory: Path, v: bool = True, wait_interval: float = 2.0
) -> Path:
    """Download a file with atomic locking and integrity checks."""
    file_path, file_name = url_dir2path(url, directory)
    lock_path = file_path.with_suffix(file_path.suffix + ".lock")
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    while lock_path.exists():
        if v:
            print(f"Download in progress by another process: {file_name}")
        time.sleep(wait_interval)
        if file_path.exists():
            if v:
                print(f"File already available: {file_name}")
            return file_path
    try:
        if file_path.exists():
            if v:
                print(f"File already downloaded: {file_name}")
            return file_path
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return download_file(url, directory, v, wait_interval)
    try:
        response = requests.get(str(url), stream=True, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded_size = 0
        with open(temp_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=file_name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    pbar.update(len(chunk))
        temp_path.rename(file_path)
        if v:
            print(f"Successfully downloaded: {file_name}")
        return file_path
    except requests.exceptions.RequestException as e:
        for path in [temp_path, file_path]:
            if path.exists():
                path.unlink()
        raise click.ClickException(f"Download failed: {file_name} - {e}")
    except KeyboardInterrupt:
        for path in [temp_path, file_path]:
            if path.exists():
                path.unlink()
        raise click.ClickException(" Download interrupted by user")
    finally:
        if lock_path.exists():
            lock_path.unlink()


def download_data(
    data_type: DataType,
    domain_name: str,
    file_suffix: FileSuffix,
) -> None:
    """Download data (corpus, IDs, or database) for a specific domain.

    This function retrieves data from a specified domain and downloads it based
    on the provided data type and file suffix. It validates input parameters
    and ensures that the target directory exists.

    Args:
        data_type (DataType): The type of data to be downloaded
        (e.g., "corpus").
        domain_name (str): The name of the domain from which to retrieve data.
        file_suffix (FileSuffix): The suffix of the file to be downloaded
        (e.g., "file" or "demo").

    Raises:
        ValueError: If the domain name or data type is invalid, or if no URL is
        found for the specified suffix.

    Example:
        >>> download_data("corpus", "bio", "file")
    """
    url, transcorpus_dir, domains_dict = get_domain_url(
        domain_name=domain_name, data_type=data_type, file_suffix=file_suffix
    )
    download_file(
        url,
        transcorpus_dir / domain_name / domains_dict[domain_name].language,
    )


class SuffixModel(BaseModel):
    """A helper model to determine the appropriate file suffix based on a flag.

    Attributes:
        flag (bool): A boolean flag indicating whether to use "demo" mode.

    Methods:
        get_suffix(): Returns "demo" if the flag is True; otherwise "file".

    Example:
        >>> model = SuffixModel(flag=True)
        >>> model.get_suffix()
        'demo'
    """

    flag: bool

    def get_suffix(self) -> FileSuffix:
        """Get the appropriate file suffix based on the flag.

        Returns:
            FileSuffix: "demo" if the flag is True; otherwise "file".

        Example:
            >>> model = SuffixModel(flag=False)
            >>> model.get_suffix()
            'file'
        """
        return "demo" if self.flag else "file"


@click.command()
@click.argument("corpus_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_corpus(corpus_name: str, demo: bool) -> None:
    """CLI command to download a corpus for a specific domain.

    Args:     corpus_name (str): The name of the corpus to be
    downloaded. demo (bool): Whether to run in demo mode.

    Example:     $ python retrieval.py download_corpus bio --demo
    """
    file_suffix = SuffixModel(flag=demo)
    download_data("corpus", corpus_name, file_suffix.get_suffix())
    download_data("id", corpus_name, file_suffix.get_suffix())


@click.command()
@click.argument("database_name")
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def download_database(database_name: str, demo: bool):
    """CLI command to download a database for a specific domain.

    Args:     database_name (str): The name of the database to be
    downloaded. demo (bool): Whether to run in demo mode.

    Example:     $ python retrieval.py download_database bio --demo
    """
    file_suffix = SuffixModel(flag=demo)
    download_data("database", database_name, file_suffix.get_suffix())
