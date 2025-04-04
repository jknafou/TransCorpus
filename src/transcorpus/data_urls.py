"""
Here is the current state of publicly available data for each domain
"""

data_urls = {
    "bio": {
        "database": "bibmed.tar.gz",
        "database_demo": "bibmed_sample.tar.gz",
        "corpus": "title_abstract_en.txt",
        "id": "PMID.txt",
        "corpus_demo": "title_abstract_en_sample.txt",
        "id_demo": "PMID_sample.txt",
        "endpoint": "https://transcorpus.s3.text-analytics.ch/",
        "language": "en",
    },
    "test": "https://transcorpus.s3.text-analytics.ch/test.txt",
}
