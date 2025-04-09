"""
Test suite for validating the `load_domains` function in the `transcorpus.config` module.

Functions:
- `test_load_invalid_domains`: Tests the `load_domains` function with various invalid configurations to ensure
    proper error handling and validation. Uses parameterized test cases to cover scenarios such as missing fields,
    invalid URL formats, unsupported language codes, and incorrect data types.

    Parameters:
        - `tmp_path` (Path): Temporary directory provided by pytest for creating test files.
        - `invalid_data` (dict): The invalid configuration data to be tested.
        - `expected_error` (str): The expected error message or substring to be raised by the function.

- `test_load_missing_file`: Tests the behavior of the `load_domains` function when the configuration file is missing.
    Ensures that the function raises a `ValueError` with the appropriate error message.

Dependencies:
- `pytest`: Used for writing and executing test cases.
- `json`: Used for serializing test data into JSON format.
- `pathlib.Path`: Used for handling file paths in a platform-independent manner.
"""

import json
import pytest
from pathlib import Path

from transcorpus.config import load_domains


@pytest.mark.parametrize(
    "invalid_data, expected_error",
    [
        # Missing required field (e.g., "database")
        (
            {
                "bio": {
                    "corpus": {"file": "https://example.com/corpus.txt"},
                    "language": "en",
                }
            },
            "Field required",
        ),
        # Invalid URL format
        (
            {
                "bio": {
                    "database": {"file": "invalid_url"},
                    "corpus": {"file": "https://example.com/corpus.txt"},
                    "language": "en",
                }
            },
            "Input should be a valid URL, relative URL without a base",
        ),
        # Invalid language code
        (
            {
                "bio": {
                    "database": {"file": "https://example.com/database.tar.gz"},
                    "corpus": {"file": "https://example.com/corpus.txt"},
                    "language": "xx",
                }
            },
            "Invalid configuration: 1 validation error for DomainData\nlanguage\n  Input should be 'af', 'am', 'ar', 'ast', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'ns', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh' or 'zu' [type=literal_error, input_value='xx', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.11/v/literal_error",
        ),
        # Invalid type for "database" (should be a dict)
        (
            {
                "bio": {
                    "database": "not_a_dict",
                    "corpus": {"file": "https://example.com/corpus.txt"},
                    "language": "en",
                }
            },
            "Input should be a valid dictionary",
        ),
    ],
)
def test_load_invalid_domains(tmp_path: Path, invalid_data: dict, expected_error: str):
    """
    Test the `load_domains` function with invalid domain data.

    This test verifies that the `load_domains` function raises a `ValueError`
    when provided with invalid domain data. It also checks that the error
    message contains the expected substring.

    Args:
        tmp_path (Path): A temporary directory path provided by pytest for
            creating temporary files.
        invalid_data (dict): A dictionary representing invalid domain data
            to be written to a JSON file for testing.
        expected_error (str): The expected substring to be found in the
            error message raised by the `load_domains` function.

    Raises:
        ValueError: Expected to be raised by the `load_domains` function
            when invalid domain data is provided.
    """
    test_json_path = tmp_path / "invalid_domains.json"
    test_json_path.write_text(json.dumps(invalid_data))

    with pytest.raises(ValueError) as exc_info:
        load_domains(test_json_path)

    assert expected_error in str(exc_info.value)


def test_load_missing_file(tmp_path: Path):
    """
    Test the behavior of the `load_domains` function when attempting to load
    a non-existent configuration file.

    This test ensures that:
    1. The specified file does not exist.
    2. The `load_domains` function raises a `ValueError` when called with the
       path to the missing file.
    3. The error message matches the expected format, indicating that the
       configuration file was not found.

    Args:
        tmp_path (Path): A temporary directory provided by pytest to create
                         test files and directories.
    """
    missing_json_path = tmp_path / "nonexistent.json"

    assert not missing_json_path.exists()

    with pytest.raises(ValueError) as exc_info:
        load_domains(missing_json_path)

    expected_error = f"Configuration file not found at {missing_json_path}"
    assert str(exc_info.value) == expected_error
