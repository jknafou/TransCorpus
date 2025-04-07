import os

home_dir = os.path.expanduser("~")
transcorpus_dir = os.path.join(home_dir, ".TransCorpus")


def create_transcorpus_dir():
    if not os.path.exists(transcorpus_dir):
        os.makedirs(transcorpus_dir)
        print(f"Created directory: {transcorpus_dir}")


if __name__ == "__main__":
    create_transcorpus_dir()
