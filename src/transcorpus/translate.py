import sys, re
from tqdm import tqdm
import logging

import nltk
from pydantic import HttpUrl
import logging

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

import os
import warnings
from pathlib import Path
from typing import Optional, get_args

import click
from fairseq.options import (
    get_generation_parser,
    get_preprocessing_parser,
    parse_args_and_arch,
)
from fairseq_cli.generate import main as generate_main
from fairseq_cli.preprocess import main as preprocess_main
from omegaconf import OmegaConf
from fairseq.dataclass.configs import FairseqConfig
import torch
import argparse


from transcorpus.languages import M2M100_Languages
from transcorpus.preprocess import spm_encode
from transcorpus.retrieval import SuffixModel, download_file
import sys
from transcorpus.utils import (
    abbreviation,
    get_domain_url,
    get_model_translation_url,
)


def process_split(filename, split_index, num_splits):
    filesize = os.path.getsize(filename)
    split_index -= 1
    start = int(split_index * filesize / num_splits)
    end = int((split_index + 1) * filesize / num_splits)

    with open(filename, "rb") as f:
        if start > 0:
            f.seek(start)
            f.readline()

        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            yield line.decode("utf-8").strip()


def too_small_sentence_concat(sentences, too_small_sentences_length=10):
    # has_sth_changed = False
    no_more_small_sentence = True
    while True:
        if len(sentences) == 1:
            break
        for i in range(len(sentences)):
            if len(sentences[i]) <= too_small_sentences_length:
                if i == 0:
                    sentences[i + 1] += sentences[i] + " " + sentences[i + 1]
                else:
                    sentences[i - 1] = sentences[i - 1] + " " + sentences[i]
                sentences.remove(sentences[i])
                no_more_small_sentence = False
                # has_sth_changed = True
                break
        if no_more_small_sentence:
            break
        no_more_small_sentence = True
    # return sentences, has_sth_changed
    return sentences


def sentence_splitter(text, tokenizer):
    text = list(tokenizer.tokenize(text))
    text = too_small_sentence_concat(text)
    abstract = []
    for sentence in text:
        abstract.append(sentence)
    return abstract


def preprocess_data(
    corpus_path: Path,
    source_lang: str,
    target_lang: str,
    model_dict_path: Path,
    dest_dir: Path,
):
    args = get_preprocessing_parser().parse_args(
        [
            "--workers",
            "1",
            "--source-lang",
            source_lang,
            "--target-lang",
            target_lang,
            "--only-source",
            "--testpref",
            corpus_path.as_posix(),
            "--thresholdsrc",
            "0",
            "--thresholdtgt",
            "0",
            "--destdir",
            dest_dir.as_posix(),
            "--srcdict",
            model_dict_path.as_posix(),
            "--tgtdict",
            model_dict_path.as_posix(),
        ]
    )
    preprocess_main(args)


def generate_translation(
    data_bin_dir: Path,  # Binarized data directory (positional arg)
    model_path: Path,  # Model checkpoint path
    source_lang: str,
    target_lang: str,
    results_path: Path,
    fixed_dictionary: Path,
    lang_pairs: Path,
):
    args_list = [
        str(data_bin_dir),
        "--path",
        str(model_path),
        "--fixed-dictionary",
        str(fixed_dictionary),
        "-s",
        source_lang,
        "-t",
        target_lang,
        "--results-path",
        str(results_path),
        "--num-workers",
        "0",
        "--no-progress-bar",
        "--fp16",
        "--batch-size",
        "1",
        # "--max-tokens",
        # "4096",
        "--max-len-a",
        "1.5",
        "--remove-bpe",
        "sentencepiece",
        "--task",
        "translation_multi_simple_epoch",
        "--lang-pairs",
        str(lang_pairs),
        "--decoder-langtok",
        "--encoder-langtok",
        "src",
    ]

    parser = get_generation_parser()
    args = parse_args_and_arch(parser, args_list)

    with torch.serialization.safe_globals([argparse.Namespace]):
        generate_main(args)


def retrieve_translation(
    generated_file: Path,
    documents_sentences_ids: list[str],
):
    translated_pattern = re.compile("^D-[0-9]+")
    en_pattern = re.compile("^S-[0-9]+")

    translation_dict = {}
    with open(generated_file, encoding="utf-8") as f:
        for line in f:
            if translated_pattern.search(line):
                id, score, sentence = line.split("\t")
                id = int(id.split("-")[-1])
                if id not in translation_dict:
                    translation_dict[id] = {}
                translation_dict[id]["translated"] = sentence.rstrip()

            elif en_pattern.search(line):
                id, sentence = line.split("\t")
                id = int(id.split("-")[-1])
                if id not in translation_dict:
                    translation_dict[id] = {}
                translation_dict[id]["en"] = sentence.rstrip()[7:]

    all_ids = sorted(list(translation_dict.keys()))
    # with open(PMID_INDEX_PATH) as f:
    #     documents_sentences_ids = f.read()
    # documents_sentences_ids = [line.rstrip().split("\t") for line in open(PMID_INDEX_PATH)]

    print("documents_sentences_ids", documents_sentences_ids)
    print("length of documents_sentences_ids", len(documents_sentences_ids))
    print("translation_dict", translation_dict)
    print("length of translation_dict", len(translation_dict))
    assert len(all_ids) == len(documents_sentences_ids), (
        "The number of generated translations does not match the number of sentences in the input file. "
        # "This can be due to a problem with the translation model not fitting your machine."
    )
    # current_pmid = documents_sentences_ids[0][0]
    # current_section = "title"
    # text_to_write = current_pmid + "\t"
    # for [pmid, section, has_cut_at_max], translated_sentence_id in tqdm(
    #     zip(documents_sentences_ids, all_ids), total=len(pmid_index)
    # ):
    #     if pmid != current_pmid:
    #         text_to_write += "\n" + pmid + "\t"
    #     elif current_section != section:
    #         text_to_write += "\t"
    #     else:
    #         text_to_write += " "
    #
    #     text_to_write += translation_dict[translated_sentence_id]["fr"]
    #     current_section = section
    #     current_pmid = pmid
    #
    # assert len(text_to_write.rstrip().split("\n")) == sum(
    #     1 for _ in open(PMID_PATH)
    # )
    # with open(TRANSLATION_PATH, encoding="utf-8", mode="w") as f:
    #     f.write(text_to_write.rstrip())
    #
    # logging.warning(
    #     "translation finsished for run number "
    #     + index_run
    #     + " and GPU number "
    #     + index_gpu
    # )


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
def translate(
    corpus_name: str,
    target: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
):
    file_suffix = SuffixModel(flag=demo)
    corpus_url, transcorpus_dir, domains_dict = get_domain_url(
        domain_name=corpus_name,
        data_type="corpus",
        file_suffix=file_suffix.get_suffix(),
    )
    domain = domains_dict[corpus_name]
    original_language = domain.language

    if original_language == target:
        raise click.UsageError(
            f"Original language '{original_language}' is the same as target '{target}'."
        )

    if num_splits < 1:
        raise click.UsageError(
            f"Number of splits must be at least 1, but got {num_splits}."
        )
    elif num_splits == 1:
        click.secho(
            f"Processing the entire file at once. No splits applied.",
            fg="yellow",
        )

    if split_index is None and num_splits > 1:
        raise click.UsageError(
            "Split index must be provided when using multiple splits."
        )

    if split_index is not None and (
        split_index < 1 or split_index > num_splits
    ):
        raise click.UsageError(
            f"Split index must be between 1 and {num_splits}, but got {split_index}."
        )

    if split_index is None:
        split_index = 1

    corpus_path = (
        transcorpus_dir
        / corpus_name
        / original_language
        / Path(str(corpus_url)).name
    )

    if not corpus_path.exists():
        raise click.UsageError(
            f"Corpus file '{corpus_path}' does not exist. Please run the download command first."
        )

    model_path = get_model_translation_url()
    sentence_tokenizer_path = download_file(
        url=HttpUrl(model_path["sentence_tokenizer_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if sentence_tokenizer_path is None:
        raise click.UsageError(
            f"Sentence tokenizer file not found at {sentence_tokenizer_path}."
        )
    sentence_tokenizer = nltk.data.load(sentence_tokenizer_path.as_posix())
    sentence_tokenizer._params.abbrev_types.update(abbreviation)

    model_tokenizer_path = download_file(
        url=HttpUrl(model_path["model_tokenizer_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if model_tokenizer_path is None:
        raise click.UsageError(
            f"Model tokenizer file not found at {model_tokenizer_path}."
        )

    file_stem = Path(str(corpus_url)).stem
    file_new_name = (
        f"{file_stem}.{split_index}_{num_splits}.{original_language}"
    )
    output_path = (
        transcorpus_dir / corpus_name / original_language / file_new_name
    )

    click.secho(
        f"Processing split {split_index} of {num_splits} for corpus '{corpus_name}'.",
        fg="green",
    )
    documents_sentences_ids = []
    with open(output_path, "w", encoding="utf-8") as f:
        for i, document in enumerate(
            process_split(
                corpus_path,
                split_index,
                num_splits,
            )
        ):
            document = document.strip()
            if document:
                document = sentence_splitter(document, sentence_tokenizer)
                documents_sentences_ids.extend([f"{i}"] * len(document))
                document, stats = spm_encode(model_tokenizer_path, document)
                for sentence in document:
                    f.write(sentence + "\n")

            else:
                click.secho(
                    f"Empty document found in split {split_index}. Skipping.",
                    fg="yellow",
                )
                continue

    if stats["num_empty"] > 0:
        click.secho(
            f"Number of empty lines: {stats['num_empty']}",
            fg="yellow",
        )

    model_tokenizer_dictionary_path = download_file(
        url=HttpUrl(model_path["model_tokenizer_dictionary_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if model_tokenizer_dictionary_path is None:
        raise click.UsageError(
            f"Model dictionary file not found at {model_tokenizer_dictionary_path}."
        )

    file_new_name = f"{file_stem}.{split_index}_{num_splits}"
    corpus_path = (
        transcorpus_dir / corpus_name / original_language / file_new_name
    )
    dest_dir = transcorpus_dir / corpus_name / target / file_new_name
    preprocess_data(
        corpus_path=corpus_path,
        source_lang=original_language,
        target_lang=target,
        model_dict_path=model_tokenizer_dictionary_path,
        dest_dir=dest_dir,
    )

    model_translation_path = download_file(
        url=HttpUrl(model_path["model_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if model_translation_path is None:
        raise click.UsageError(
            f"Model dictionary file not found at {model_tokenizer_path}."
        )

    model_dictionary_path = download_file(
        url=HttpUrl(model_path["model_dictionary_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if model_dictionary_path is None:
        raise click.UsageError(
            f"Model dictionary file not found at {model_tokenizer_path}."
        )

    language_pairs_path = download_file(
        url=HttpUrl(model_path["model_language_pairs_url"]),
        directory=transcorpus_dir / "models",
        v=False,
    )
    if language_pairs_path is None:
        raise click.UsageError(
            f"Language pairs file not found at {model_tokenizer_path}."
        )

    click.secho(
        f"Processed {len(documents_sentences_ids)} sentences.",
        fg="green",
    )

    generate_translation(
        data_bin_dir=dest_dir,
        model_path=model_translation_path,
        source_lang=original_language,
        target_lang=target,
        results_path=dest_dir.parent,
        fixed_dictionary=model_dictionary_path,
        lang_pairs=language_pairs_path,
    )

    retrieve_translation(
        dest_dir.parent / "generate-test.txt",
        documents_sentences_ids,
    )
