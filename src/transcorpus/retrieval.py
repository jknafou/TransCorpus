import subprocess
import os


def download_file(url, directory):
    file_name = os.path.join(directory, os.path.basename(url))

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"File already downloaded: {file_name}")
        return

    try:
        subprocess.run(["wget", url, "-P", directory], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
