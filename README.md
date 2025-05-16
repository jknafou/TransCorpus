# TransCorpus
TransCorpus is a scalable, production-ready API and CLI toolkit for large-scale parallel translation, preprocessing, and corpus management. It supports multi-GPU translation, robust checkpointing, and safe concurrent downloads, making it ideal for research and industry-scale machine translation workflows.

# Features
- 🚀 Multi-GPU and multi-process translation
- 📦 Corpus downloading and preprocessing
- 🔒 Safe, resumable, and concurrent file downloads
- 🧩 Split and checkpoint management for large corpora
- 🛠️ Easy deployment and extensibility
- 🖥️ Cross-platform: Linux, macOS, Windows

# Quick Start
1. Clone and Install
```bash
git clone git@github.com:jknafou/TransCorpus.git
cd TransCorpus
UV_INDEX_STRATEGY=unsafe-best-match rye sync
source .venv/bin/activate
```
if you prefer pip:
```bash
git clone git@github.com:jknafou/TransCorpus.git
cd TransCorpus
python3.10 -m venv .venv
source .venv/bin/activate
pip install .
```





# Work in Progress
- [ ] create a pypi package
- [ ] write tests
- [ ] avoid unloading the GPU between split and give the model a new binary file to translate
- [ ] limit fairseq verbose
- [ ] clean code + pass test
