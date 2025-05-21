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
    parse_args_and_arch,
)
from fairseq_cli.generate import main as generate_main
from omegaconf import OmegaConf
from fairseq.dataclass.configs import FairseqConfig
import torch
import argparse


from transcorpus.languages import M2M100_Languages
from transcorpus.preprocess import spm_encode
from transcorpus.retrieval import SuffixModel, download_file
import sys
from pydocstyle.wordlists import stem
from transcorpus.utils import preview_proposal_messsage
from transcorpus.preprocess import process_split
from transcorpus.preprocess import sentence_splitter
from transcorpus.preprocess import preprocess_data
from transcorpus.preprocess import run_preprocess


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


def generate_translation(
    data_bin_dir: Path,  # Binarized data directory (positional arg)
    model_path: Path,  # Model checkpoint path
    source_language: M2M100_Languages,
    target_language: M2M100_Languages,
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
        source_language,
        "-t",
        target_language,
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
            split_path.unlink(missing_ok=True)


def get_model_assets(
    model_path: dict, transcorpus_dir: Path
) -> tuple[Path, Path, Path]:
    path_tuple = []
    for m in ["model_url", "model_dictionary_url", "model_language_pairs_url"]:
        if m == "model_url":
            click.secho(
                "Downloading the model. This may take a while.",
                fg="green",
            )
        path_tuple.append(
            download_file(
                url=HttpUrl(model_path[m]),
                directory=transcorpus_dir / "models",
                v=False,
            )
        )
        if path_tuple[-1] is None:
            raise click.UsageError(
                f"Model dictionary file not found at {path_tuple[-1]}."
            )
    return tuple(path_tuple)


def run_translation(
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    max_tokens: int,
):
    recursive_preprocess = num_splits > 1 and not split_index
    result = run_preprocess(
        corpus_name=corpus_name,
        target_language=target_language,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
        from_translation=True,
    )
    if result is None:
        raise ValueError("run_preprocess returned None unexpectedly")
    (
        split_stage,
        split_index,
        model_path,
        transcorpus_dir,
        dest_dir,
        source_language,
        checkpoint_db,
        checkpoint_path,
        documents_sentences_ids_path,
    ) = result
    if split_stage < 3:
        try:
            (
                model_translation_path,
                model_dictionary_path,
                language_pairs_path,
            ) = get_model_assets(
                model_path=model_path,
                transcorpus_dir=transcorpus_dir,
            )
            generate_translation(
                data_bin_dir=dest_dir,
                model_path=model_translation_path,
                source_language=source_language,
                target_language=target_language,
                results_path=dest_dir,
                fixed_dictionary=model_dictionary_path,
                lang_pairs=language_pairs_path,
                max_tokens=max_tokens,
            )
            for item in dest_dir.iterdir():
                if item.is_file() and item.name != "generate-test.txt":
                    item.unlink(missing_ok=True)
            checkpoint_db.update_stage(split_index, 3)
        except Exception as e:
            click.secho(
                "Error during translation generation. Please check the model path and try again.",
                fg="red",
            )
            checkpoint_db.update_stage(split_index, 0)
            raise e
    if split_stage < 4:
        try:
            with open(documents_sentences_ids_path, "r", encoding="utf-8") as f:
                documents_sentences_ids = f.read().strip().split("_")
            retrieve_translation(
                dest_dir,
                documents_sentences_ids,
            )
            generated_translation_path = dest_dir / "generate-test.txt"
            generated_translation_path.unlink(missing_ok=True)
            dest_dir.rmdir()
            # @new_feature documents_sentences_ids_path can be used to
            # translate the same splits in other languages (same for tokenized
            # files, but their size is much larger), if you want to keep them,
            # comment the next line and the tokenized file deletion too.
            # Modification of the stages handling should also take that into
            # account. (default 1 after first pass).
            documents_sentences_ids_path.unlink(missing_ok=True)
            checkpoint_db.update_stage(split_index, 4)
        except Exception as e:
            click.secho(
                "Error during translation retrieval. Please check the model path and try again.",
                fg="red",
            )
            checkpoint_db.update_stage(split_index, 0)
            raise e
    uncompleted_splits = checkpoint_db.get_len_uncompleted_splits()
    if uncompleted_splits == 1:
        merge_splits(
            dest_dir=dest_dir,
            num_splits=num_splits,
        )
        checkpoint_db.complete_split(split_index)
        uncompleted_splits = checkpoint_db.get_len_uncompleted_splits()
        if uncompleted_splits == 0:
            click.secho(
                "Translation completed successfully.",
                fg="green",
            )
            preview_proposal_messsage(
                corpus_name=corpus_name,
                source_language=source_language,
                target_language=target_language,
                demo=demo,
            )
            checkpoint_path.unlink(missing_ok=True)
            dest_file = Path(str(dest_dir) + ".txt")
            dest_file.unlink(missing_ok=True)
        exit(0)

    if uncompleted_splits > 1 and recursive_preprocess:
        click.secho(
            f"Number of uncompleted splits: {uncompleted_splits}",
            fg="yellow",
        )
        if split_index is not None:
            checkpoint_db.complete_split(split_index)
            click.secho(
                "Launching the next split.",
                fg="green",
            )
        run_translation(
            corpus_name=corpus_name,
            target_language=target_language,
            split_index=None,
            num_splits=num_splits,
            demo=demo,
            max_tokens=max_tokens,
        )


@click.command()
@click.argument("corpus_name")
@click.argument(
    "target-language", type=click.Choice(get_args(M2M100_Languages))
)
@click.option(
    "--split-index",
    "-i",
    type=int,
    default=None,
    help="index of the split to process (1-based). if not provided, the whole file will be processed iteratively.",
)
@click.option(
    "--num-splits",
    "-n",
    type=int,
    default=1,
    help="number of splits to divide the file into. default is 1 (no split).",
)
@click.option(
    "--max-tokens",
    "-m",
    type=int,
    default=2048,
    help="Maximum number of tokens per batch. Default is 2048 (to avoid crash"
    " in small config). For a GPU with 32GB of memory, we recommend about 8192,"
    " for 40GB about 24320... However, this is not a strict rule. You can try"
    " to increase it to whatever value you want, but you might get an OOM"
    " error.",
)
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def translate(
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    max_tokens: int,
):
    run_translation(
        corpus_name=corpus_name,
        target_language=target_language,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
        max_tokens=max_tokens,
    )
