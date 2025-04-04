"""
Here is the current state of publicly available data for each domain
"""

corpus_urls = {
    "bio": {
        # TODO:
        # later, normalikze the names so it would not have PMIDs, or title_abstracts...
        "files": ["PMID.txt", "title_abstract_en.txt"],
        "files_demo": ["PMID_sample.txt", "title_abstract_en_sample.txt"],
        "endpoint": "https://transcorpus.s3.text-analytics.ch/",
        "language": "en",
    }
}
