import subprocess
import os


def download_file(url, dir):
    file_name = os.path.join(dir, os.path.basename(url))

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"File already downloaded: {file_name}")
        return

    try:
        subprocess.run(["wget", url, "-P", dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
