"""
Unit tests for the `transcorpus.retrieval` module.

This test suite includes various test cases to validate the functionality of the
retrieval-related functions, including downloading files, handling errors, and
ensuring proper behavior under different scenarios.

Fixtures:
    - `_domains_fixture`: Loads domain data from a test JSON file.
    - `_mock_download_setup`: Sets up mocks for `requests.get` and provides
      necessary data for download-related tests.
    - `_mock_dependencies`: Mocks dependencies like directory creation, file
      downloading, and domain loading.

Test Cases:
    - `test_download_file_already_exists`: Verifies that the function correctly
      identifies and skips downloading files that already exist.
    - `test_download_file_successful`: Ensures successful file download and
      proper handling of streamed content.
    - `test_download_file_deletes_partial_file_on_error`: Validates that partial
      files are deleted when an error occurs during download.
    - `test_download_file_deletes_partial_file_on_keyboard_interrupt`: Confirms
      that partial files are deleted when a `KeyboardInterrupt` is raised during
      download.
    - `test_download_data`: Tests the `download_data` function for various
      scenarios, including invalid domain names, data types, and file suffixes.
    - `test_download_database`: Tests the `download_database` CLI command.
    - `test_download_corpus`: Tests the `download_corpus` CLI command.
    - `test_download_corpus_id`: Tests the `download_corpus_id` CLI command.

Constants:
    - `COMMON_PARAMS`: A list of common parameters used for parameterized tests.

Dependencies:
    - `pytest`: For writing and running test cases.
    - `unittest.mock`: For mocking external dependencies.
    - `requests`: For simulating HTTP requests.
    - `click.testing.CliRunner`: For testing CLI commands.
"""

from pathlib import Path
from typing import Any
from unittest import mock

from pydantic import HttpUrl
import pytest
import requests


from click.testing import CliRunner

from transcorpus.retrieval import (
    download_corpus,
    download_corpus_id,
    download_database,
    download_data,
    download_file,
)
from transcorpus.config import load_domains
from transcorpus.models import DomainData
from pytest_mock import MockerFixture


@pytest.fixture(name="_domains_fixture")
def domains_fixture() -> dict[str, DomainData]:
    """
    Loads domain data from a JSON file for testing purposes.

    The function constructs the path to a JSON file named "test_domains.json"
    located in the same directory as the current test file. It then loads
    and returns the domain data as a dictionary.

    Returns:
        dict[str, DomainData]: A dictionary where the keys are strings
        representing domain identifiers and the values are DomainData objects
        containing the corresponding domain information.
    """
    test_json_path = Path(__file__).resolve().parent / "test_domains.json"
    return load_domains(test_json_path)


@pytest.fixture(name="_mock_download_setup")
def mock_download_setup(
    _domains_fixture: dict[str, DomainData], data_type, tmp_path: Path, file_name
):
    """
    Fixture to mock the download setup for testing purposes.

    This fixture patches the `requests.get` method to simulate a download process
    without making actual HTTP requests. It provides a mocked response object,
    a mocked `requests.get` function, a data entry from the `_domains_fixture`,
    and a temporary file path for testing.

    Args:
        _domains_fixture (dict[str, DomainData]): A dictionary containing domain data
            used for testing. The specific data entry is determined by the `data_type`.
        data_type (str): The type of data to retrieve from the `_domains_fixture`.
        tmp_path (Path): A temporary directory path provided by pytest for storing
            temporary files during the test.
        file_name (str): The name of the file to be used in the temporary path.

    Yields:
        tuple: A tuple containing:
            - mock_response (Mock): A mocked response object with predefined headers
            and behavior.
            - mock_get (Mock): A mocked `requests.get` function.
            - data_entry: The specific data entry from `_domains_fixture` based on `data_type`.
            - file_path (Path): The full path to the temporary file.
    """
    with mock.patch("requests.get") as mock_get:
        mock_response = mock.Mock()
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        data_entry = getattr(_domains_fixture["test"], data_type)
        file_path = tmp_path / file_name

        yield mock_response, mock_get, data_entry, file_path


COMMON_PARAMS = [
    ("database", "test.txt"),
    ("corpus", "test.txt"),
    ("id", "test.txt"),
]


@pytest.mark.parametrize(("data_type", "file_name"), COMMON_PARAMS)
def test_download_file_already_exists(
    _domains_fixture: dict[str, DomainData],
    tmp_path: Path,
    data_type: str,
    file_name: str,
):
    """
    Test that the `download_file` function correctly handles the case where the file
    already exists in the target directory.

    This test uses the `COMMON_PARAMS` parameterized values to check multiple
    combinations of `data_type` and `file_name`. It ensures that the function does
    not attempt to re-download the file and instead logs a message indicating that
    the file is already downloaded.

    Args:
        _domains_fixture (dict[str, DomainData]): A fixture providing domain-specific
            data for testing.
        tmp_path (Path): A temporary directory path provided by pytest for testing
            file operations.
        data_type (str): The type of data being tested (e.g., "text", "image").
        file_name (str): The name of the file to be checked.

    Setup:
        - Creates a dummy file in the temporary directory to simulate an already
        downloaded file.

    Test Steps:
        1. Mock the `print` function to capture its output.
        2. Retrieve the data entry corresponding to the `data_type` from the
        `_domains_fixture`.
        3. Call the `download_file` function with the file path and temporary
        directory.
        4. Assert that the mocked `print` function was called with the expected
        message indicating the file already exists.
    """
    dummy_file = tmp_path / file_name
    dummy_file.touch()

    with mock.patch("builtins.print") as mock_print:
        data_entry = getattr(_domains_fixture["test"], data_type)
        download_file(data_entry.file, tmp_path)
        mock_print.assert_called_with(f"File already downloaded: {file_name}")


@pytest.mark.parametrize(("data_type", "file_name"), COMMON_PARAMS)
def test_download_file_successful(
    _mock_download_setup: tuple[mock.Mock, mock.MagicMock | mock.AsyncMock, Any, Any],
    tmp_path: Path,
):
    """
    Test the successful download of a file.

    This test verifies that the `download_file` function correctly downloads a file
    and performs the expected operations, such as printing a success message and
    making the appropriate HTTP GET request.

    Args:
        _mock_download_setup (tuple[mock.Mock, mock.MagicMock | mock.AsyncMock, Any, Any]):
            A tuple containing mocked objects and data for setting up the test:
            - mock_response: A mocked HTTP response object.
            - mock_get: A mocked HTTP GET function (synchronous or asynchronous).
            - data_entry: An object representing the file to be downloaded.
            - file_path: The expected file path where the file will be saved.
        tmp_path (Path): A temporary directory path provided by pytest for storing
            test files.

    Asserts:
        - The `print` function is called with the expected success message.
        - The HTTP GET request is made with the correct URL, stream enabled, and
        a timeout of 10 seconds.
    """
    mock_response, mock_get, data_entry, file_path = _mock_download_setup
    mock_response.iter_content.return_value = iter([b"chunk1", b"chunk2"])

    with mock.patch("builtins.print") as mock_print:
        download_file(data_entry.file, tmp_path)
        mock_print.assert_called_with(f"Downloaded: {file_path}")
        mock_get.assert_called_once_with(str(data_entry.file), stream=True, timeout=10)


@pytest.mark.parametrize(("data_type", "file_name"), COMMON_PARAMS)
def test_download_file_deletes_partial_file_on_error(
    _mock_download_setup: tuple[mock.Mock, mock.MagicMock | mock.AsyncMock, Any, Any],
    tmp_path: Path,
):
    """
    Test that `download_file` deletes the partially downloaded file when an error occurs during the download process.

    This test uses parameterized inputs for different data types and file names, and mocks the download setup to simulate
    a scenario where an exception is raised during the file download. It ensures that the function handles the error
    gracefully by returning `None` and removing any partially downloaded file.

    Args:
        _mock_download_setup (tuple): A tuple containing mocked objects and data for the download setup, including:
            - mock_response: A mocked response object with a side effect to simulate an error during `iter_content`.
            - Other elements used for testing purposes.
        tmp_path (Path): A temporary directory path provided by pytest for storing test files.

    Asserts:
        - The result of `download_file` is `None` when an error occurs.
        - The partially downloaded file does not exist after the error.
    """
    mock_response, _, data_entry, file_path = _mock_download_setup

    def iter_content_side_effect(*args, **kwargs):
        yield b"data"
        raise requests.exceptions.RequestException("Simulated error")

    mock_response.iter_content.side_effect = iter_content_side_effect

    result = download_file(data_entry.file, tmp_path)
    assert result is None
    assert not file_path.exists()


@pytest.mark.parametrize(("data_type", "file_name"), COMMON_PARAMS)
def test_download_file_deletes_partial_file_on_keyboard_interrupt(
    _mock_download_setup: tuple[mock.Mock, mock.MagicMock | mock.AsyncMock, Any, Any],
    _domains_fixture: dict[str, DomainData],
    tmp_path: Path,
    mocker: MockerFixture,
):
    """
    Test that the `download_file` function deletes a partially downloaded file
    when a `KeyboardInterrupt` is raised during the download process.

    This test uses parameterized inputs for `data_type` and `file_name` from
    `COMMON_PARAMS`. It mocks the download setup, domain fixture, and file
    system path to simulate the behavior of the `download_file` function.

    Args:
        _mock_download_setup (tuple): A tuple containing mocked objects for
            the download setup, including the response object.
        _domains_fixture (dict): A dictionary mapping domain names to their
            corresponding `DomainData` objects.
        tmp_path (Path): A temporary directory path for storing test files.
        mocker (MockerFixture): A pytest fixture for mocking objects.

    Test Steps:
    1. Configure the mocked response's `iter_content` method to raise a
    `KeyboardInterrupt` after yielding some data.
    2. Call the `download_file` function with the test URL and temporary path.
    3. Assert that a `KeyboardInterrupt` is raised during the download.
    4. Verify that the partially downloaded file is deleted from the
    temporary path.
    """
    mock_response, _, _, _ = _mock_download_setup

    def iter_content_side_effect(*args, **kwargs):
        yield b"data"
        raise KeyboardInterrupt

    mock_response.iter_content.side_effect = iter_content_side_effect

    url = _domains_fixture["test"].corpus.file
    with pytest.raises(KeyboardInterrupt):
        download_file(url, tmp_path)

    assert not (tmp_path / Path(str(url)).name).exists()


@pytest.mark.parametrize(
    ("data_type", "domain_name", "file_suffix", "expected_error"),
    [
        ("database", "unknown_domain", "file", "Unknown domain name:"),
        ("unknown_database", "test", "file", "Invalid data type"),
        ("database", "test", "demo", "No 'demo' found for"),
        ("database", "test", "file", ""),
    ],
)
def test_download_data(
    data_type, domain_name, file_suffix, expected_error, _domains_fixture
):
    """
    Test the `download_data` function to ensure it behaves correctly under various scenarios.

    Args:
        data_type (str): The type of data to be downloaded.
        domain_name (str): The name of the domain for which data is being downloaded.
        file_suffix (str): The suffix of the file to be downloaded.
        expected_error (str or None): The expected error message if an exception is raised.
                                      If None, no exception is expected.
        _domains_fixture (dict): A fixture providing mock domain data for testing.

    Mocks:
        - `transcorpus.retrieval.create_transcorpus_dir`: Mocked to simulate directory creation.
        - `transcorpus.retrieval.download_file`: Mocked to simulate file download.
        - `transcorpus.retrieval.load_domains`: Mocked to simulate loading domain data.

    Test Cases:
        - If `expected_error` is provided, the function should raise a `ValueError` with
          the expected error message.
        - If no error is expected, the function should:
            - Call `create_transcorpus_dir` once to create the directory.
            - Call `download_file` with the correct URL and file path.

    Assertions:
        - Verifies that the expected exception is raised when `expected_error` is provided.
        - Verifies that the mocked functions are called with the correct arguments when
          no error is expected.
    """
    with (
        mock.patch("transcorpus.retrieval.create_transcorpus_dir") as mock_create_dir,
        mock.patch("transcorpus.retrieval.download_file") as mock_download_file,
        mock.patch("transcorpus.retrieval.load_domains") as mock_load_domains,
    ):
        mock_create_dir.return_value = Path("/mock/transcorpus")
        mock_load_domains.return_value = _domains_fixture

        if expected_error:
            with pytest.raises(ValueError) as exc_info:
                download_data(data_type, domain_name, file_suffix)
            assert expected_error in str(exc_info.value)

        else:
            download_data(data_type, domain_name, file_suffix)

            mock_create_dir.assert_called_once()
            mock_download_file.assert_called_once_with(
                HttpUrl("http://example.com/test.txt"),
                Path("/mock/transcorpus/test/en"),
            )


@pytest.fixture(name="_mock_dependencies")
def mock_dependencies(_domains_fixture):
    """
    Mock dependencies for testing the retrieval module.

    This function uses `unittest.mock.patch` to mock the following functions:
    - `transcorpus.retrieval.create_transcorpus_dir`
    - `transcorpus.retrieval.download_file`
    - `transcorpus.retrieval.load_domains`

    The mocked functions are configured as follows:
    - `create_transcorpus_dir`: Returns a mocked `Path` object pointing to "/mock/transcorpus".
    - `load_domains`: Returns the `_domains_fixture` provided as an argument.

    Yields:
        dict: A dictionary containing the mocked objects:
            - "mock_create_dir": Mock for `create_transcorpus_dir`.
            - "mock_download_file": Mock for `download_file`.
            - "mock_load_domains": Mock for `load_domains`.
    """
    with (
        mock.patch("transcorpus.retrieval.create_transcorpus_dir") as mock_create_dir,
        mock.patch("transcorpus.retrieval.download_file") as mock_download_file,
        mock.patch("transcorpus.retrieval.load_domains") as mock_load_domains,
    ):
        mock_create_dir.return_value = Path("/mock/transcorpus")
        mock_load_domains.return_value = _domains_fixture

        yield {
            "mock_create_dir": mock_create_dir,
            "mock_download_file": mock_download_file,
            "mock_load_domains": mock_load_domains,
        }


def test_download_database(_mock_dependencies):
    """
    Test the `download_database` function using a CLI runner.

    This test verifies that the `download_database` command executes successfully
    and that the `mock_download_file` dependency is called with the correct arguments.

    Args:
        _mock_dependencies (dict): A dictionary containing mocked dependencies,
            including `mock_download_file`.

    Assertions:
        - The command's exit code is 0, indicating successful execution.
        - The `mock_download_file` function is called exactly once with the expected
          URL and file path.
    """
    runner = CliRunner()
    result = runner.invoke(download_database, ["test"])

    assert result.exit_code == 0
    _mock_dependencies["mock_download_file"].assert_called_once_with(
        HttpUrl("http://example.com/test.txt"),
        Path("/mock/transcorpus/test/en"),
    )


def test_download_corpus(_mock_dependencies):
    """
    Test the `download_corpus` function to ensure it correctly downloads a corpus file.

    This test uses a mocked CLI runner to invoke the `download_corpus` command with a test argument.
    It verifies that the command exits successfully and that the `mock_download_file` dependency
    is called with the expected URL and file path.

    Args:
        _mock_dependencies (dict): A dictionary of mocked dependencies, including:
            - "mock_download_file": A mock object for the file download function.

    Assertions:
        - The command's exit code is 0, indicating success.
        - The `mock_download_file` is called exactly once with the expected URL and file path.
    """
    runner = CliRunner()
    result = runner.invoke(download_corpus, ["test"])

    assert result.exit_code == 0
    _mock_dependencies["mock_download_file"].assert_called_once_with(
        HttpUrl("http://example.com/test.txt"),
        Path("/mock/transcorpus/test/en"),
    )


def test_download_corpus_id(_mock_dependencies):
    """
    Test the `download_corpus_id` function to ensure it correctly downloads a corpus
    file and saves it to the expected location.

    This test uses a mocked dependency to simulate the download process and verifies:
    - The command-line interface (CLI) invocation completes successfully with an exit code of 0.
    - The `mock_download_file` method is called exactly once with the expected URL and file path.

    Args:
        _mock_dependencies (dict): A dictionary of mocked dependencies, including:
            - "mock_download_file": A mock object for the file download function.
    """
    runner = CliRunner()
    result = runner.invoke(download_corpus_id, ["test"])

    assert result.exit_code == 0
    _mock_dependencies["mock_download_file"].assert_called_once_with(
        HttpUrl("http://example.com/test.txt"),
        Path("/mock/transcorpus/test/en"),
    )
