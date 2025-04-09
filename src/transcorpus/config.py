"""
TransCorpus Configuration Module.

This module provides functionality for loading domain configurations from a JSON file.
The main function, `load_domains`, reads and validates the configuration file, ensuring
it adheres to the expected structure defined by the `DomainData` model.

Functions:
    - load_domains: Loads and validates domain configurations from a JSON file.

Typical usage example:
    from transcorpus.config import load_domains
    from pathlib import Path

    domains = load_domains(Path("domains.json"))
"""

import json
from pathlib import Path
from pydantic import ValidationError

from transcorpus.models import DomainData


def load_domains(json_path: Path) -> dict[str, DomainData]:
    """
    Load and validate domain configurations from a JSON file.

    This function reads a JSON file containing domain-specific configurations,
    validates the data using the `DomainData` model, and returns a dictionary
    mapping domain names to their corresponding configurations.

    Args:
        json_path (Path): The path to the JSON configuration file.

    Returns:
        dict[str, DomainData]: A dictionary where keys are domain names and values
        are validated `DomainData` objects.

    Raises:
        ValueError: If the configuration is invalid or the file is not found.
            - Invalid configuration: Raised when the JSON data does not match
              the expected structure defined by the `DomainData` model.
            - File not found: Raised when the specified JSON file does not exist.

    Example:
        >>> from pathlib import Path
        >>> domains = load_domains(Path("domains.json"))
        >>> print(domains["bio"])
        DomainData(language="en", database={"file": "http://example.com/file"})
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            domain_name: DomainData(**domain_data)
            for domain_name, domain_data in data.items()
        }
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e

    except FileNotFoundError as e:
        raise ValueError(f"Configuration file not found at {json_path}") from e
