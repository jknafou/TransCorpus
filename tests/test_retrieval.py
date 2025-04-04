import subprocess
import os
import pytest
from transcorpus.retrieval import download_file, download_data
from transcorpus import transcorpus_dir
from transcorpus.data_urls import data_urls
from unittest.mock import patch, Mock


def test_file_already_exists(tmp_path):
    # Create a dummy file in the temporary directory
    dummy_file = tmp_path / "test.txt"
    dummy_file.touch()  # Create an empty file

    # Call the function
    result = download_file(data_urls["test"], tmp_path)

    # Assert the returned file path is correct
    expected_file_path = tmp_path / "test.txt"
    assert result == str(expected_file_path)


def test_successful_download(tmp_path):
    # Mock subprocess.run to simulate successful download
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock()  # Simulate successful subprocess execution

        result = download_file(data_urls["test"], str(tmp_path))

        # Assert subprocess.run was called correctly
        mock_run.assert_called_once_with(
            ["wget", data_urls["test"], "-P", str(tmp_path)], check=True
        )

        # Assert the returned file path is correct
        expected_file_path = tmp_path / "test.txt"
        assert result == str(expected_file_path)


def test_failed_download(tmp_path):
    # Mock subprocess.run to simulate a CalledProcessError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, cmd=["wget", transcorpus_dir, "-P", str(tmp_path)]
        )

        result = download_file(transcorpus_dir, str(tmp_path))

        # Assert subprocess.run was called correctly
        mock_run.assert_called_once_with(
            ["wget", transcorpus_dir, "-P", str(tmp_path)], check=True
        )

        # Assert the function handles the error gracefully and returns None
        assert result is None


# Test: Unknown domain name
def test_unknown_domain():
    with pytest.raises(ValueError, match="Unknown domain name: unknown"):
        download_data("corpus", "unknown", demo=True)


# Test: No demo available for the domain
def test_no_demo_available():
    with pytest.raises(ValueError, match="No corpus demos available for: test"):
        download_data("corpus", "test", demo=True)


@pytest.mark.parametrize(
    "data_type, demo",
    [
        ("corpus", True),
        ("id", True),
        ("database", True),
        ("corpus", False),
        ("id", False),
        ("database", False),
    ],
)
@patch("transcorpus.retrieval.download_file")
def test_path_format(mock_download_file, data_type, demo):
    domain_name = "bio"

    expected_url = (
        data_urls[domain_name]["endpoint"] + data_urls[domain_name][f"{data_type}_demo"]
    )
    expected_path = os.path.join(
        transcorpus_dir, domain_name, data_urls[domain_name]["language"]
    )

    # Call the function
    download_data(data_type, domain_name, True)

    # Assert that download_file was called with the correct arguments
    mock_download_file.assert_called_once_with(expected_url, expected_path)
