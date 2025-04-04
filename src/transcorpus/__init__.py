import os

home_dir = os.path.expanduser("~")
transcorpus_dir = os.path.join(home_dir, ".TransCorpus")
if not os.path.exists(transcorpus_dir):
    os.makedirs(transcorpus_dir)
    print(f"Created directory: {transcorpus_dir}")
