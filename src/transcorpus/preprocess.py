from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import click

from typing import Optional, get_args
import sentencepiece as spm
from typing import List
from pathlib import Path
from transcorpus.languages import M2M100_Languages


def run_preprocess(
    corpus_name: str,
    target: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
):
    pass


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


@click.command()
@click.argument("corpus_name")
@click.option(
    "--target",
    "-t",
    required=True,
    type=click.Choice(get_args(M2M100_Languages)),
    help="Target language for translation.",
)
@click.option(
    "--split-index",
    "-i",
    type=int,
    default=None,
    help="Index of the split to process (1-based).",
)
@click.option(
    "--num-splits",
    "-n",
    type=int,
    default=1,
    help="Number of splits to divide the file into.",
)
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def preprocess(
    corpus_name: str,
    target: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
):
    run_preprocess(
        corpus_name=corpus_name,
        target=target,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
    )
