import subprocess
from transcorpus.retrieval import download_file
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
