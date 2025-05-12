from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import click

import sentencepiece as spm
from typing import List
from pathlib import Path


def spm_encode(
    model: Path,
    doc_sentences: List[str],
):
    sp = spm.SentencePieceProcessor()
    sp.Load(model.as_posix())

    def encode(input):
        return sp.EncodeAsPieces(input)

    stats = {
        "num_empty": 0,
    }

    def encode_line(line):
        line = line.strip()
        if len(line) > 0:
            line = encode(line)
            return line
        else:
            stats["num_empty"] += 1
        return None

    for i, doc_line in enumerate(doc_sentences):
        enc_line = encode_line(doc_line)
        if enc_line is not None:
            doc_sentences[i] = " ".join(enc_line)
        else:
            click.secho(
                f"Empty line found. Skipping.",
                fg="yellow",
            )

    return doc_sentences, stats
