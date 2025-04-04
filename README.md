# Notes for README:

https://ftp.ncbi.nlm.nih.gov/pubmed/

## Corpus Retrieval

Data can be retrieve from the following ftp server:

https://ftp.ncbi.nlm.nih.gov/pubmed/

The version used for TransBERT training are stored in the following S3 bucket:

```bash
wget -P ../data/bibmed https://transcorpus.s3.text-analytics.ch/bibmed.tar.gz
tar -xzvf ../data/bibmed/bibmed.tar.gz
```

## Corpus Preprocessing

````bash

## Corpus Translation

```bash
wget https://transcorpus.s3.text-analytics.ch/PMID.txt
wget https://transcorpus.s3.text-analytics.ch/title_abstract_en.txt
````

## Model Download

Commands to be add in CLI

TODO:

```bash
# transcorpus upload law

transcorpus download-database bio -d
transcorpus download-corpus bio -d
transcorpus download-corpus-id bio -d

transcorpus preview bio -d --count 1 --start-at 0 --language en

transcorpus process-database bio # just to show and give people the possibility to modify a few steps...
transcorpus process-corpus bio -d
# database would have one step more so if people want to tweak it

transcorpus split-corpus bio --n_split 16

transcorpus translate bio --target fr --split 1

transcorpus merge-translated-corpus bio

```
