import os

home_dir = os.path.expanduser("~")
trans_corpus_dir = os.path.join(home_dir, ".TransCorpus")
if not os.path.exists(trans_corpus_dir):
    os.makedirs(trans_corpus_dir)
    print(f"Created directory: {trans_corpus_dir}")
