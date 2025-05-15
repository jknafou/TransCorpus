from fairseq.checkpoint_utils import checkpoint_paths
import shutil
import psutil
import time

from transcorpus.translation_checkpointing import TranslationCheckpointing
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
from pydocstyle.wordlists import stem
from transcorpus.utils import (
    abbreviation,
    get_domain_url,
    get_model_translation_url,
)


def preview_proposal_messsage(
    corpus_name: str, original_language: str, target: str, demo: bool
):
    message = "You can now run the command preview to see the translation."
    if demo:
        message += f"\ntranscorpus preview {corpus_name} -l {original_language} -l {target} -c 100 -d"
    else:
        message += f"\ntranscorpus preview {corpus_name} -l {original_language} -l {target} -c 100"
    click.secho(message, fg="green")


def kill_data_workers(process_list):
    """Force terminate a list of PyTorch DataLoader workers and their descendants."""
    # First try SIGTERM (polite request to terminate)
    for proc in process_list:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    # Wait up to 3 seconds for processes to exit
    gone, alive = psutil.wait_procs(process_list, timeout=3)

    # Force kill any remaining processes with SIGKILL
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    # Final verification
    remaining = [p for p in process_list if p.is_running()]
    return len(remaining) == 0


def process_split(filename, split_index, num_splits, message=""):
    filesize = os.path.getsize(filename)
    split_index -= 1
    start = int(split_index * filesize / num_splits)
    end = int((split_index + 1) * filesize / num_splits)

    with open(filename, "rb") as f:
        if start > 0:
            f.seek(start)
            f.readline()

        pbar = tqdm(
            total=end - start,
            unit="B",
            unit_scale=True,
            desc=message,
        )
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            pbar.update(len(line))
            yield line.decode("utf-8").strip()
    pbar.close()


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
    click.secho(
        "Data pre-processing: Building the dictionary and binarizing the data.",
        fg="green",
    )
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
    max_tokens: int,
):
    click.secho(
        "Generating translations using the trained model.",
        fg="green",
    )
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
        "8",
        "--fp16",
        "--required-batch-size-multiple",
        "1",
        "--max-tokens",
        str(max_tokens),
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

    parent_process = psutil.Process()
    with torch.serialization.safe_globals([argparse.Namespace]):
        generate_main(args)

    timeout = 60
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_children = parent_process.children(recursive=True)
        if not current_children:
            break
        time.sleep(2)
        kill_data_workers(current_children)
    else:
        click.secho(
            "Warning: Subprocesses did not terminate within timeout!",
            fg="yellow",
        )


def retrieve_translation(
    dest_dir: Path,
    documents_sentences_ids: list[str],
):
    click.secho(
        "Retrieving translations from the generated output.",
        fg="green",
    )
    generated_translation_path = dest_dir / "generate-test.txt"
    filesize = os.path.getsize(generated_translation_path)
    translated_pattern = re.compile("^D-[0-9]+")
    translation_dict = {}
    with open(generated_translation_path, encoding="utf-8") as f:
        pbar = tqdm(
            total=filesize,
            unit="B",
            unit_scale=True,
            desc="Retrieving translations",
        )
        for line in f:
            pbar.update(len(line))
            if translated_pattern.search(line):
                i, score, sentence = line.split("\t")
                translation_dict[int(i.split("-")[-1])] = sentence.rstrip()

    assert len(translation_dict) == len(documents_sentences_ids), (
        "The number of generated translations does not match the number of sentences in the input file."
        f"Length of generated translations: {len(translation_dict)}, Length of input sentences: {len(documents_sentences_ids)}"
    )
    current_id = None
    with open(dest_dir.as_posix() + ".txt", encoding="utf-8", mode="w") as f:
        for i, doc_id in enumerate(
            tqdm(documents_sentences_ids, desc="Writing translations")
        ):
            if current_id is not None and doc_id != current_id:
                f.write("\n")  # Add newline AFTER the previous document ends
            f.write(
                translation_dict[i].strip() + " "
            )  # Trim existing newlines in sentences
            current_id = doc_id
        f.write("\n")  # Add newline AFTER the last document ends


def merge_splits(
    dest_dir: Path,
    num_splits: int,
):
    click.secho(
        f"Merging split files into a single file in {dest_dir.parent}.",
        fg="green",
    )
    expected_files = [
        Path(f"{dest_dir.parent / dest_dir.stem}.{i}_{num_splits}.txt")
        for i in range(1, num_splits + 1)
    ]
    with open(
        dest_dir.parent / (dest_dir.stem + ".txt"), "w", encoding="utf-8"
    ) as outfile:
        for split_path in expected_files:
            with open(
                split_path,
                "r",
                encoding="utf-8",
            ) as infile:
                for line in infile:
                    outfile.write(line)


def run_translation(
    corpus_name: str,
    target: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    max_tokens: int,
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

    corpus_path = (
        transcorpus_dir
        / corpus_name
        / original_language
        / Path(str(corpus_url)).name
    )

    dest_path = (
        transcorpus_dir / corpus_name / target / Path(str(corpus_url)).name
    )
    if dest_path.exists():
        click.secho(
            f"Corpus already translated.",
            fg="yellow",
        )
        preview_proposal_messsage(
            corpus_name=corpus_name,
            original_language=original_language,
            target=target,
            demo=demo,
        )
        exit(0)

    if num_splits < 1:
        raise click.UsageError(
            f"Number of splits must be at least 1, but got {num_splits}."
        )
    elif num_splits == 1:
        click.secho(
            f"Processing the entire file at once (num_splits=1).",
            fg="yellow",
        )

    if split_index is None and num_splits > 1:
        click.secho(
            f"Split index not provided. Will process the whole iteratively by split {num_splits} times.",
            fg="yellow",
        )

    if split_index is not None and (
        split_index < 1 or split_index > num_splits
    ):
        raise click.UsageError(
            f"Split index must be between 1 and {num_splits}, but got {split_index}."
        )

    if not corpus_path.exists():
        raise click.UsageError(
            f"Corpus file '{corpus_path}' does not exist. Please run the download command first."
        )

    file_size_bytes = os.path.getsize(corpus_path)
    size_gb = file_size_bytes / (1024**3)
    size_mb = file_size_bytes / (1024**2)

    if size_gb >= 1:
        size_str = f"{round(size_gb, 2)} GB"
    else:
        size_str = f"{round(size_mb, 2)} MB"

    click.secho(
        f"Corpus file size to be processed: {size_str}",
        fg="yellow",
    )

    if split_index is None:
        checkpoint_path = Path(
            transcorpus_dir
            / corpus_name
            / target
            / f"checkpoint_{file_suffix.get_suffix()}_{num_splits}.db"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_db = TranslationCheckpointing(checkpoint_path, num_splits)
        split_index = checkpoint_db.claim_next_split()
        if split_index:
            click.secho(
                f"Claimed split number: {split_index}",
                fg="green",
            )
        else:
            click.secho(
                f"All splits are already completed. No more splits to process.",
                fg="yellow",
            )
            preview_proposal_messsage(
                corpus_name=corpus_name,
                original_language=original_language,
                target=target,
                demo=demo,
            )
            exit(0)

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
    tokenized_split_file = (
        f"{file_stem}.{split_index}_{num_splits}.{original_language}"
    )
    tokenized_split_path = (
        transcorpus_dir / corpus_name / original_language / tokenized_split_file
    )
    document_sentences_ids_file = f"{file_stem}.{split_index}_{num_splits}.ids"
    documents_sentences_ids_path = (
        transcorpus_dir
        / corpus_name
        / original_language
        / document_sentences_ids_file
    )

    split_stage = checkpoint_db.get_stage(split_index)
    if split_stage == 0:
        if (
            not tokenized_split_path.exists()
            or not documents_sentences_ids_path.exists()
        ):
            try:
                temp_tokenized = tokenized_split_path.with_name(
                    f"{tokenized_split_path.name}.tmp"
                )
                temp_ids = documents_sentences_ids_path.with_name(
                    f"{documents_sentences_ids_path.name}.tmp"
                )
                click.secho(
                    f"Processing split {split_index} of {num_splits} for corpus '{corpus_name}'.",
                    fg="green",
                )
                documents_sentences_ids = []
                with open(
                    temp_tokenized,
                    "w",
                    encoding="utf-8",
                ) as f:
                    with open(
                        temp_ids,
                        "w",
                        encoding="utf-8",
                    ) as f_ids:
                        for i, document in enumerate(
                            process_split(
                                corpus_path,
                                split_index,
                                num_splits,
                                message=f"Tokenization split {split_index} of {num_splits} for corpus '{corpus_name}'",
                            )
                        ):
                            document = document.strip()
                            if document:
                                document = sentence_splitter(
                                    document, sentence_tokenizer
                                )
                                documents_sentences_ids.extend(
                                    [f"{i}"] * len(document)
                                )
                                document, stats = spm_encode(
                                    model_tokenizer_path, document
                                )
                                for sentence in document:
                                    f.write(sentence + "\n")

                            else:
                                click.secho(
                                    f"Empty document found in split {split_index}. Skipping.",
                                    fg="yellow",
                                )
                                continue
                        f_ids.write("_".join(documents_sentences_ids))

                if stats["num_empty"] > 0:
                    click.secho(
                        f"Number of empty lines: {stats['num_empty']}",
                        fg="yellow",
                    )
                if temp_tokenized.exists() and temp_ids.exists():
                    temp_tokenized.rename(tokenized_split_path)
                    temp_ids.rename(documents_sentences_ids_path)
                else:
                    raise FileNotFoundError(
                        "Temp files missing after processing"
                    )

                checkpoint_db.update_stage(split_index, 1)
            except Exception as e:
                click.secho(
                    f"Error during tokenization: {e}",
                    fg="red",
                )
                temp_tokenized.unlink(missing_ok=True)
                temp_ids.unlink(missing_ok=True)
                checkpoint_db.update_stage(split_index, 0)
            except KeyboardInterrupt:
                click.secho(
                    "Keyboard interrupt detected. Exiting.",
                    fg="red",
                )
                temp_tokenized.unlink(missing_ok=True)
                temp_ids.unlink(missing_ok=True)
                checkpoint_db.update_stage(split_index, 0)
                exit(0)
        else:
            click.secho(
                f"Tokenized split file '{tokenized_split_path}' and document sentences IDs file '{documents_sentences_ids_path}' already exists. Skipping tokenization.",
                fg="yellow",
            )
            with open(documents_sentences_ids_path, "r", encoding="utf-8") as f:
                documents_sentences_ids = f.read().strip().split("_")

    file_new_name = f"{file_stem}.{split_index}_{num_splits}"
    dest_dir = transcorpus_dir / corpus_name / target / file_new_name
    if split_stage > 1:
        click.secho(
            f"Resuming split {split_index} at stage {split_stage}.",
            fg="yellow",
        )
    if split_stage < 2:
        model_tokenizer_dictionary_path = download_file(
            url=HttpUrl(model_path["model_tokenizer_dictionary_url"]),
            directory=transcorpus_dir / "models",
            v=False,
        )
        if model_tokenizer_dictionary_path is None:
            raise click.UsageError(
                f"Model dictionary file not found at {model_tokenizer_dictionary_path}."
            )

        split_corpus_path = (
            transcorpus_dir / corpus_name / original_language / file_new_name
        )
        preprocess_data(
            corpus_path=split_corpus_path,
            source_lang=original_language,
            target_lang=target,
            model_dict_path=model_tokenizer_dictionary_path,
            dest_dir=dest_dir,
        )
        click.secho(
            f"File sentences processed and binarized.",
            fg="green",
        )
        tokenized_split_path.unlink(missing_ok=True)
        checkpoint_db.update_stage(split_index, 2)

    if split_stage < 3:
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
        generate_translation(
            data_bin_dir=dest_dir,
            model_path=model_translation_path,
            source_lang=original_language,
            target_lang=target,
            results_path=dest_dir,
            fixed_dictionary=model_dictionary_path,
            lang_pairs=language_pairs_path,
            max_tokens=max_tokens,
        )
        for item in dest_dir.iterdir():
            if item.is_file() and item.name != "generate-test.txt":
                item.unlink(missing_ok=True)
        checkpoint_db.update_stage(split_index, 3)

    if split_stage < 4:
        if split_stage != 0:
            with open(documents_sentences_ids_path, "r", encoding="utf-8") as f:
                documents_sentences_ids = f.read().strip().split("_")

        retrieve_translation(
            dest_dir,
            documents_sentences_ids,
        )
        generated_translation_path = dest_dir / "generate-test.txt"
        generated_translation_path.unlink(missing_ok=True)
        checkpoint_db.update_stage(split_index, 4)

    uncompleted_splits = checkpoint_db.get_len_uncompleted_splits()
    if uncompleted_splits == 1:
        merge_splits(
            dest_dir=dest_dir,
            num_splits=num_splits,
        )
        preview_proposal_messsage(
            corpus_name=corpus_name,
            original_language=original_language,
            target=target,
            demo=demo,
        )
        if split_index is not None:
            checkpoint_db.complete_split(split_index)
            click.secho(
                f"Completed split number: {split_index}",
                fg="green",
            )
        uncompleted_splits = checkpoint_db.get_len_uncompleted_splits()
        if uncompleted_splits == 0:
            checkpoint_path.unlink(missing_ok=True)
            shutil.rmtree(dest_dir)
            dest_file = Path(str(dest_dir) + ".txt")
            dest_file.unlink(missing_ok=True)
            documents_sentences_ids_path.unlink(missing_ok=True)
        exit(0)

    else:
        click.secho(
            f"Number of uncompleted splits: {uncompleted_splits}",
            fg="yellow",
        )
        if split_index is not None:
            checkpoint_db.complete_split(split_index)
            click.secho(
                f"Launching the next split.",
                fg="green",
            )
        run_translation(
            corpus_name=corpus_name,
            target=target,
            split_index=None,
            num_splits=num_splits,
            demo=demo,
            max_tokens=max_tokens,
        )


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
@click.option(
    "--max-tokens",
    "-m",
    type=int,
    default=2048,
    help="Maximum number of tokens per batch. Default is 2048 (to avoid crash in small config). For a GPU with 32GB of memory, we recommend about 8192, for 40GB about 19456... However, this is not a strict rule. You can try to increase it to whatever value you want, but you might get an OOM error.",
)
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def translate(
    corpus_name: str,
    target: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    max_tokens: int,
):
    run_translation(
        corpus_name=corpus_name,
        target=target,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
        max_tokens=max_tokens,
    )
