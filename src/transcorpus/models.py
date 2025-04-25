"""
TransCorpus Models Module.

This module defines the data models used throughout the TransCorpus project.
These models are built using Pydantic to ensure validation of input data and
enforce type safety.

Models:
    - DataEntry: Represents a single entry in a domain configuration.
    - DomainData: Represents the configuration for a specific domain, including
    its database, corpus, ID, and language.

Type Aliases:
    - FileSuffix: Literal type for allowed file suffixes.
    - DataType: Literal type for allowed data types.

Typical usage example:
    from transcorpus.models import DomainData, DataEntry

    data_entry = DataEntry(file="http://example.com/file",
    demo="http://example.com/demo")
    domain_data = DomainData(
        database=data_entry,
        corpus=data_entry,
        id=None,
        language="en"
    )
"""

from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl

from transcorpus.languages import M2M100_Languages

FileSuffix = Literal["file", "demo"]
DataType = Literal["database", "corpus", "id"]


class DataEntry(BaseModel):
    """
    Represents a single entry in a domain configuration.

    Attributes:
        file (HttpUrl): The URL pointing to the main file for this entry.
        demo (Optional[HttpUrl]): An optional URL pointing to a demo version of
        the file.

    Example:
        >>> from transcorpus.models import DataEntry
        >>> data_entry = DataEntry(file="http://example.com/file",
        demo="http://example.com/demo")
        >>> print(data_entry.file)
        http://example.com/file
    """

    file: HttpUrl
    demo: Optional[HttpUrl] = None


class DomainData(BaseModel):
    """
    Represents the configuration for a specific domain.

    Attributes:
        database (DataEntry): The database entry for this domain.
        corpus (DataEntry): The corpus entry for this domain.
        id (Optional[DataEntry]): An optional ID entry for this domain.
        language (M2M100_Languages): The language associated with this domain.

    Example:
        >>> from transcorpus.models import DomainData, DataEntry
        >>> data_entry = DataEntry(file="http://example.com/file",
        demo="http://example.com/demo")
        >>> domain_data = DomainData(
        ...     database=data_entry,
        ...     corpus=data_entry,
        ...     id=None,
        ...     language="en"
        ... )
        >>> print(domain_data.language)
        en
    """

    database: DataEntry
    corpus: DataEntry
    id: Optional[DataEntry] = None
    language: M2M100_Languages
