from transcorpus.__init__ import create_transcorpus_dir
import os
from unittest import mock


# Test function
def test_create_transcorpus_dir():
    # Mock os.path.exists and os.makedirs
    with (
        mock.patch("os.path.exists") as mock_exists,
        mock.patch("os.makedirs") as mock_makedirs,
    ):
        # Set up the return value for os.path.exists
        mock_exists.return_value = False

        # Call the function to test
        create_transcorpus_dir()

        # Assert that os.path.exists was called with the correct path
        home_dir = os.path.expanduser("~")
        expected_path = os.path.join(home_dir, ".TransCorpus")
        mock_exists.assert_called_with(expected_path)

        # Assert that os.makedirs was called with the correct path
        mock_makedirs.assert_called_with(expected_path)
