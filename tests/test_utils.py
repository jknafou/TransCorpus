"""
Test the `create_transcorpus_dir` function to ensure it creates the expected
directory in the user's home directory.

This test uses the `unittest.mock` library to patch the `Path.home` method,
redirecting it to a temporary path (`tmp_path`) provided by pytest. It verifies
the following:
1. The function returns the expected path (`~/.TransCorpus`).
2. The directory is created if it does not exist.
3. The function behaves idempotently, returning the same path without creating
    a new directory if it already exists.

Args:
     tmp_path (Path): A pytest fixture providing a temporary directory for testing.
"""

from transcorpus.utils import create_transcorpus_dir
from unittest import mock
from pathlib import Path


def test_create_transcorpus_dir(tmp_path):
    """
    Test the `create_transcorpus_dir` function to ensure it creates the expected
    directory in the user's home path.

    This test uses a temporary path to mock the user's home directory and verifies:
    - The directory is created at the expected location.
    - The created path exists and is a directory.
    - Subsequent calls to `create_transcorpus_dir` return the same path without
      creating a new directory.

    Args:
        tmp_path (pathlib.Path): A temporary directory provided by pytest for testing.
    """
    with mock.patch.object(Path, "home") as mock_home:
        mock_home.return_value = tmp_path
        expected_path = tmp_path / ".TransCorpus"

        result = create_transcorpus_dir()

        assert result == expected_path
        assert expected_path.exists()
        assert expected_path.is_dir()

        result = create_transcorpus_dir()
        assert result == expected_path
