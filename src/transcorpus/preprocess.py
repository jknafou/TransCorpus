from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

import os, sys
from pathlib import Path
from typing import List, Optional, get_args, NoReturn, Literal

import click
import sentencepiece as spm
from fairseq.options import get_preprocessing_parser
from fairseq_cli.preprocess import main as preprocess_main
from pydantic.networks import HttpUrl
from tqdm import tqdm

from transcorpus.languages import M2M100_Languages
from transcorpus.retrieval import SuffixModel, download_file
from transcorpus.translation_checkpointing import TranslationCheckpointing
from transcorpus.utils import (
    abbreviation,
    get_domain_url,
    get_model_translation_url,
    preview_proposal_messsage,
    url_dir2path,
)


def get_corpus_paths(
    corpus_name: str,
    target_language: M2M100_Languages,
    transcorpus_dir: Path,
    domains_dict: dict,
    corpus_url: HttpUrl,
    split_index: Optional[int] = None,
    num_splits: int = 1,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return all relevant paths for corpus processing."""
    source_language = domains_dict[corpus_name].language
    corpus_path = (
        transcorpus_dir
        / corpus_name
        / source_language
        / Path(str(corpus_url)).name
    )
    dest_path = (
        transcorpus_dir
        / corpus_name
        / target_language
        / Path(str(corpus_url)).name
    )
    file_stem = Path(str(corpus_url)).stem
    tokenized_split_file = (
        f"{file_stem}.{split_index}_{num_splits}.{source_language}"
    )
    tokenized_split_path = (
        transcorpus_dir / corpus_name / source_language / tokenized_split_file
    )
    documents_sentences_ids_path = (
        transcorpus_dir
        / corpus_name
        / source_language
        / f"{file_stem}.{split_index}_{num_splits}.ids"
    )
    return (
        corpus_path,
        dest_path,
        tokenized_split_path,
        documents_sentences_ids_path,
        transcorpus_dir,
    )


def validate_languages(
    source_language: M2M100_Languages, target_language: M2M100_Languages
) -> None:
    """Ensure source and target languages are different."""
    if source_language == target_language:
        raise click.UsageError(
            f"Source '{source_language}' and target '{target_language}' languages are the same."
        )


def validate_splits(num_splits: int, split_index: Optional[int]) -> None:
    """Validate split configuration."""
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
            f"Split index not provided. Processing the entire file iteratively by splits.",
            fg="yellow",
        )
    if split_index is not None and (
        split_index < 1 or split_index > num_splits
    ):
        raise click.UsageError(
            f"Split index must be between 1 and {num_splits}, but got {split_index}."
        )


def validate_file_exists(corpus_path: Path) -> None:
    """Check if a critical file exists."""
    if not corpus_path.exists():
        raise click.UsageError(
            f"Corpus file '{corpus_path}' does not exist. Please run the download command first."
        )


def num_split_checkpoint_path(
    transcorpus_dir: Path,
    corpus_name: str,
    target_language: M2M100_Languages,
    num_splits: int,
    file_suffix: SuffixModel,
) -> Path:
    """Get the path for the checkpoint database."""
    checkpoint_path = Path(
        transcorpus_dir
        / corpus_name
        / target_language
        / f"checkpoint_{file_suffix.get_suffix()}_{num_splits}.db"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def split_index_is_none(
    split_index: Optional[int],
    purpose: Literal["translation", "preprocess"],
) -> None:
    if split_index is None:
        click.secho(
            f"No more splits available to {purpose}.",
            fg="green",
        )
        sys.exit(0)


def handle_checkpoints(
    transcorpus_dir: Path,
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    file_suffix: SuffixModel,
    from_translation: bool = False,
) -> tuple[int, TranslationCheckpointing]:
    """Handle checkpointing for translation splits."""
    checkpoint_path = num_split_checkpoint_path(
        transcorpus_dir, corpus_name, target_language, num_splits, file_suffix
    )
    checkpoint_db = TranslationCheckpointing(checkpoint_path, num_splits)
    if from_translation:
        split_index = (
            checkpoint_db.claim_next_split()
            if split_index is None
            else split_index
        )
        split_index_is_none(
            split_index=split_index,
            purpose="translation",
        )
    else:
        split_index = (
            checkpoint_db.claim_next_split(max_stage=2)
            if split_index is None
            else split_index
        )
        split_index_is_none(
            split_index=split_index,
            purpose="preprocess",
        )
    click.secho(
        f"Claimed split number: {split_index}",
        fg="green",
    )
    if split_index is None:
        raise click.UsageError(
            f"No more splits available to process for '{corpus_name}' in '{target_language}'."
        )
    return split_index, checkpoint_db


def load_sentence_tokenizer(
    transcorpus_dir: Path,
    model_url: HttpUrl,
    abbreviation: list,
):
    sentence_tokenizer_path = download_file(
        url=model_url,
        directory=transcorpus_dir / "models",
        v=False,
    )
    sentence_tokenizer = nltk.data.load(sentence_tokenizer_path.as_posix())
    sentence_tokenizer._params.abbrev_types.update(abbreviation)
    return sentence_tokenizer


def get_tokenizer_assets(
    transcorpus_dir: Path,
    model_path: dict,
) -> tuple[nltk.tokenize.PunktSentenceTokenizer, Path]:
    """Download and return paths to tokenizers."""
    sentence_tokenizer = load_sentence_tokenizer(
        transcorpus_dir,
        HttpUrl(model_path["sentence_tokenizer_url"]),
        abbreviation,
    )
    model_tokenizer_path = download_file(
        url=model_path["model_tokenizer_url"],
        directory=transcorpus_dir / "models",
        v=False,
    )
    return sentence_tokenizer, model_tokenizer_path


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


def preprocess_data(
    corpus_path: Path,
    source_language: M2M100_Languages,
    target_language: M2M100_Languages,
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
            source_language,
            "--target-lang",
            target_language,
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


def too_small_sentence_concat(sentences, too_small_sentences_length=10):
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
    return sentences


def sentence_splitter(text, tokenizer):
    text = list(tokenizer.tokenize(text))
    text = too_small_sentence_concat(text)
    abstract = []
    for sentence in text:
        abstract.append(sentence)
    return abstract


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


def check_already_translated(
    dest_path: Path,
    corpus_name: str,
    target_language: M2M100_Languages,
    source_language: M2M100_Languages,
    demo: bool,
) -> None:
    if dest_path.exists():
        click.secho(
            f"Corpus already translated.",
            fg="yellow",
        )
        preview_proposal_messsage(
            corpus_name=corpus_name,
            source_language=source_language,
            target_language=target_language,
            demo=demo,
        )
        exit(0)


def check_corpus_size(corpus_path: Path) -> None:
    file_size_bytes = os.path.getsize(corpus_path)
    size_gb = file_size_bytes / (1024**3)
    size_mb = file_size_bytes / (1024**2)
    if size_gb >= 1:
        size_str = f"{round(size_gb, 2)} GB"
    else:
        size_str = f"{round(size_mb, 2)} MB"
    click.secho(
        f"Size of the entire corpus to be translated: {size_str}",
        fg="yellow",
    )


def process_init(
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    from_translation: bool = False,
) -> tuple[
    Path,
    Path,
    HttpUrl,
    Path,
    Path,
    M2M100_Languages,
    M2M100_Languages,
    int,
    TranslationCheckpointing,
]:
    file_suffix = SuffixModel(flag=demo)
    corpus_url, transcorpus_dir, domains_dict = get_domain_url(
        domain_name=corpus_name,
        data_type="corpus",
        file_suffix=file_suffix.get_suffix(),
    )
    source_language = domains_dict[corpus_name].language
    split_index, checkpoint_db = handle_checkpoints(
        transcorpus_dir=transcorpus_dir,
        corpus_name=corpus_name,
        target_language=target_language,
        split_index=split_index,
        num_splits=num_splits,
        file_suffix=file_suffix,
        from_translation=from_translation,
    )
    (
        corpus_path,
        dest_path,
        tokenized_split_path,
        documents_sentences_ids_path,
        transcorpus_dir,
    ) = get_corpus_paths(
        corpus_name=corpus_name,
        target_language=target_language,
        domains_dict=domains_dict,
        transcorpus_dir=transcorpus_dir,
        corpus_url=corpus_url,
        split_index=split_index,
        num_splits=num_splits,
    )
    validate_languages(
        source_language=source_language,
        target_language=target_language,
    )
    validate_splits(
        num_splits=num_splits,
        split_index=split_index,
    )
    validate_file_exists(
        corpus_path=corpus_path,
    )
    check_already_translated(
        dest_path=dest_path,
        corpus_name=corpus_name,
        target_language=target_language,
        source_language=source_language,
        demo=demo,
    )
    if split_index == 1:
        check_corpus_size(
            corpus_path=corpus_path,
        )
    return (
        corpus_path,
        transcorpus_dir,
        corpus_url,
        tokenized_split_path,
        documents_sentences_ids_path,
        source_language,
        target_language,
        split_index,
        checkpoint_db,
    )


def file_to_tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def handle_tokenization_error(
    temp_tokenized: Path,
    temp_ids: Path,
    checkpoint_db: TranslationCheckpointing,
    split_index: int,
    e: Optional[None | Exception] = None,
) -> None:
    temp_tokenized.unlink(missing_ok=True)
    temp_ids.unlink(missing_ok=True)
    checkpoint_db.update_stage(split_index, 0)
    checkpoint_db.set_pending(split_index)
    if e:
        click.secho(
            f"Error during tokenization: {e}",
            fg="red",
        )
    else:
        click.secho(
            "Keyboard interrupt detected. Exiting.",
            fg="red",
        )
        exit(0)


def load_tokenized_split(
    tokenizer_split_path: Path, documents_sentences_ids_path: Path
) -> List[str]:
    if (
        not tokenizer_split_path.exists()
        and not documents_sentences_ids_path.exists()
    ):
        raise FileNotFoundError(
            f"Tokenized split file '{tokenizer_split_path}' and document sentences IDs file '{documents_sentences_ids_path}' do not exist."
        )
    with open(documents_sentences_ids_path, "r", encoding="utf-8") as f:
        documents_sentences_ids = f.read().strip().split("_")
    return documents_sentences_ids


def tokenize_split_by_sentence(
    corpus_path: Path,
    split_index: int,
    num_splits: int,
    tokenized_split_path: Path,
    documents_sentences_ids_path: Path,
    checkpoint_db: TranslationCheckpointing,
    transcorpus_dir: Path,
    corpus_name: str,
    model_path: dict,
) -> List[str]:
    try:
        click.secho(
            f"Processing split {split_index} of {num_splits} for corpus '{corpus_name}'.",
            fg="green",
        )
        sentence_tokenizer, model_tokenizer_path = get_tokenizer_assets(
            transcorpus_dir=transcorpus_dir,
            model_path=model_path,
        )
        temp_tokenized = file_to_tmp(tokenized_split_path)
        temp_ids = file_to_tmp(documents_sentences_ids_path)
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
                        documents_sentences_ids.extend([f"{i}"] * len(document))
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
                    checkpoint_db.update_stage(split_index, 1)
                else:
                    raise FileNotFoundError(
                        "Temp files missing after processing"
                    )
    except Exception as e:
        handle_tokenization_error(
            temp_tokenized=file_to_tmp(tokenized_split_path),
            temp_ids=file_to_tmp(documents_sentences_ids_path),
            checkpoint_db=checkpoint_db,
            split_index=split_index,
            e=e,
        )
    except KeyboardInterrupt:
        handle_tokenization_error(
            temp_tokenized=file_to_tmp(tokenized_split_path),
            temp_ids=file_to_tmp(documents_sentences_ids_path),
            checkpoint_db=checkpoint_db,
            split_index=split_index,
            e=None,
        )
    return documents_sentences_ids


def split_dest_dir(
    corpus_url: HttpUrl,
    split_index: int,
    num_splits: int,
    transcorpus_dir: Path,
    corpus_name: str,
    language: M2M100_Languages,
) -> Path:
    file_stem = Path(str(corpus_url)).stem
    file_new_name = f"{file_stem}.{split_index}_{num_splits}"
    return transcorpus_dir / corpus_name / language / file_new_name


def sentence_binarize(
    corpus_url: HttpUrl,
    transcorpus_dir: Path,
    corpus_name: str,
    source_language: M2M100_Languages,
    target_language: M2M100_Languages,
    split_index: int,
    num_splits: int,
    checkpoint_db: TranslationCheckpointing,
    model_path: dict,
    tokenized_split_path: Path,
) -> None:
    try:
        dest_dir = split_dest_dir(
            corpus_url=corpus_url,
            split_index=split_index,
            num_splits=num_splits,
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
            language=target_language,
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
        split_corpus_path = split_dest_dir(
            corpus_url=corpus_url,
            split_index=split_index,
            num_splits=num_splits,
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
            language=source_language,
        )
        preprocess_data(
            corpus_path=split_corpus_path,
            source_language=source_language,
            target_language=target_language,
            model_dict_path=model_tokenizer_dictionary_path,
            dest_dir=dest_dir,
        )
        click.secho(
            f"File sentences processed and binarized.",
            fg="green",
        )
        tokenized_split_path.unlink(missing_ok=True)
        checkpoint_db.update_stage(split_index, 2)
    except Exception as e:
        click.secho(
            f"Error during binarization: {e}",
            fg="red",
        )
        checkpoint_db.update_stage(split_index, 0)
        checkpoint_db.set_pending(split_index)


def run_preprocess(
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
    from_translation: bool = False,
) -> Optional[
    tuple[
        int,
        int,
        dict,
        Path,
        Path,
        M2M100_Languages,
        TranslationCheckpointing,
        Path,
        Path,
    ]
]:
    recursive_preprocess = num_splits > 1 and not split_index
    model_path = get_model_translation_url()
    (
        corpus_path,
        transcorpus_dir,
        corpus_url,
        tokenized_split_path,
        documents_sentences_ids_path,
        source_language,
        target_language,
        split_index,
        checkpoint_db,
    ) = process_init(
        corpus_name=corpus_name,
        target_language=target_language,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
        from_translation=from_translation,
    )
    split_stage = checkpoint_db.get_stage(split_index)
    if split_stage == 0:
        documents_sentences_ids = tokenize_split_by_sentence(
            model_path=model_path,
            corpus_path=corpus_path,
            split_index=split_index,
            num_splits=num_splits,
            tokenized_split_path=tokenized_split_path,
            documents_sentences_ids_path=documents_sentences_ids_path,
            checkpoint_db=checkpoint_db,
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
        )
    if split_stage < 2:
        sentence_binarize(
            corpus_url=corpus_url,
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
            source_language=source_language,
            target_language=target_language,
            split_index=split_index,
            num_splits=num_splits,
            checkpoint_db=checkpoint_db,
            model_path=model_path,
            tokenized_split_path=tokenized_split_path,
        )
    click.secho(
        f"Split {split_index} of {num_splits} for corpus '{corpus_name}' preprocessed.",
        fg="green",
    )
    if recursive_preprocess and not from_translation:
        checkpoint_db.set_pending(split_index)
        run_preprocess(
            corpus_name=corpus_name,
            target_language=target_language,
            split_index=None,
            num_splits=num_splits,
            demo=demo,
            from_translation=from_translation,
        )
    if from_translation:
        model_tokenizer_path = url_dir2path(
            url=model_path["model_tokenizer_url"],
            directory=transcorpus_dir / "models",
        )
        dest_dir = split_dest_dir(
            corpus_url=corpus_url,
            split_index=split_index,
            num_splits=num_splits,
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
            language=target_language,
        )
        checkpoint_path = num_split_checkpoint_path(
            transcorpus_dir=transcorpus_dir,
            corpus_name=corpus_name,
            target_language=target_language,
            num_splits=num_splits,
            file_suffix=SuffixModel(flag=demo),
        )
        return (
            split_stage,
            split_index,
            model_path,
            transcorpus_dir,
            dest_dir,
            source_language,
            checkpoint_db,
            checkpoint_path,
            documents_sentences_ids_path,
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
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def preprocess(
    corpus_name: str,
    target_language: M2M100_Languages,
    split_index: Optional[int],
    num_splits: int,
    demo: bool,
):
    run_preprocess(
        corpus_name=corpus_name,
        target_language=target_language,
        split_index=split_index,
        num_splits=num_splits,
        demo=demo,
        from_translation=False,
    )
