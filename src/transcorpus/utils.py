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

from transcorpus.config import load_domains, load_translation_models


def create_transcorpus_dir() -> Path:
    """
    Create the `.TransCorpus` directory in the user's home directory if it does not exist.

    This function checks whether the `.TransCorpus` directory exists in the
    user's home directory. If it does not exist, it creates the directory and
    prints a message indicating that it was created.

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


def get_domain_url(domain_name: str, data_type: str, file_suffix: str) -> tuple:
    """
    Get the URL and directory for a specific domain and data type.

    This function retrieves the URL and directory for a specific domain and
    data type (e.g., corpus, IDs) based on the provided domain name and file
    suffix. It validates the input parameters and ensures that the target
    directory exists.

    Args:
        domain_name (str): The name of the domain.
        data_type (str): The type of data (e.g., "corpus", "id").
        file_suffix (str): The file suffix to be used.

    Returns:
        tuple: A tuple containing the URL, the directory path, and the
        domains dictionary.

    Raises:
        ValueError: If the domain name or data type is invalid, or if no URL is
        found for the specified suffix.

    Example:
        >>> get_domain_url("bio", "corpus", "demo")
        ('https://example.com/bio/demo.txt',
        PosixPath('/home/user/.TransCorpus/bio/demo.txt'), {'bio': {...}})
    """
    domains_dict = load_domains(
        Path(__file__).resolve().parent / "domains.json"
    )
    if domain_name not in domains_dict:
        raise ValueError(f"Unknown domain name: {domain_name}")

    data_entry = getattr(domains_dict[domain_name], data_type, None)
    if not data_entry:
        raise ValueError(
            f"Invalid data type '{data_type}' for domain '{domain_name}'"
        )
    url = getattr(data_entry, file_suffix, None)
    if not url:
        raise ValueError(
            f"No '{file_suffix}' found for {data_type} in {domain_name}"
        )
    transcorpus_dir = create_transcorpus_dir()
    return url, transcorpus_dir, domains_dict


def get_model_translation_url():
    return load_translation_models(
        Path(__file__).resolve().parent / "translation_urls.json"
    )


abbreviation = [
    "a",
    "å",
    "Ǻ",
    "Å",
    "b",
    "c",
    "d",
    "e",
    "ɛ",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "Ö",
    "Ø",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "µm",
    "abs",
    "al",
    "approx",
    "bp",
    "ca",
    "cap",
    "cf",
    "co",
    "d.p.c",
    "dr",
    "e.g",
    "et",
    "etc",
    "er",
    "eq",
    "fig",
    "figs",
    "h",
    "i.e",
    "it",
    "inc",
    "min",
    "ml",
    "mm",
    "mol",
    "ms",
    "no",
    "nt",
    "ref",
    "r.p.m",
    "sci",
    "s.d",
    "sd",
    "sec",
    "s.e.m",
    "sp",
    "ssp",
    "st",
    "supp",
    "vs",
    "wt",
]
