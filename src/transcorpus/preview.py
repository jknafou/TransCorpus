"""TransCorpus Preview Module.

This module provides functionality for previewing text files for specific
domains. It includes a command-line interface (CLI) for previewing corpus
files, corpus IDs, if available. If a translation is available, the translation
can be previewed as well for convenience.

Functions:     - preview_txt: Preview a text file for a specific domain with its
translations if available.

Typical usage example:
$ python preview.py preview_corpus bio --demo --count 5 --start-at 0
"""

import shutil
import textwrap
from pathlib import Path
from typing import Optional, get_args

import click
import readchar

from transcorpus.languages import M2M100_Languages
from transcorpus.retrieval import SuffixModel
from transcorpus.utils import get_domain_url

global_domain_name = None


def validate_file(file_path: Path, is_original: bool) -> None:
    """Validate file existence with appropriate error messages."""
    if not file_path.exists():
        msg = (
            "Please download the file before previewing it."
            if is_original
            else "Please translate the file before previewing it."
        )
        raise FileNotFoundError(f"File not found: {file_path}\n{msg}")


def validate_indices(start: int, count: int, total: int) -> None:
    """Validate start and count parameters against file length."""
    if start >= total:
        raise ValueError(f"Start index {start} out of range ({total} lines)")
    if start + count > total:
        raise ValueError(
            f"Count {count} exceeds {total} lines from index {start}"
        )


def read_file_lines(file_path: Path, start: int, end: int) -> list[str]:
    """Read specific lines from a file efficiently."""
    return [
        line.strip()
        for j, line in enumerate(file_path.open(encoding="utf-8"))
        if start <= j < end
    ]


def preview_txt(
    domain_name: str,
    file_suffix: str,
    count: int,
    start_at: int,
    language: tuple[M2M100_Languages],
):
    """
    Preview a text file for a specific domain with translations if available.

    Args:
        domain_name (str): The name of the domain.
        file_suffix (str): The file suffix for the corpus.
        count (int): The number of items to preview.
        start_at (int): The starting index for the preview.
        language (tuple[M2M100_Languages]): The languages to preview.

    Returns:
        tuple: A tuple containing:
            - preview_lines (list): The lines to preview.
            - preview_ids (list): The IDs of the lines.
            - language_list (list): The list of languages.
            - source_language (str): The source language of the corpus.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the start index or count exceeds the file length.

    Example:
        preview_txt(
            domain_name="bio",
            file_suffix="demo",
            count=5,
            start_at=0,
            language=("en", "fr"),
        )
    """
    corpus_url, transcorpus_dir, domains_dict = get_domain_url(
        domain_name=domain_name, data_type="corpus", file_suffix=file_suffix
    )

    domain = domains_dict[domain_name]
    corpus_id_url = (
        getattr(domain.id, file_suffix, None) if hasattr(domain, "id") else None
    )
    source_language = domain.language
    corpus_file_name = Path(str(corpus_url)).name

    language_list = sorted(set(language), key=lambda x: x != source_language)

    if not corpus_id_url:
        preview_ids = [str(i) for i in range(start_at, start_at + count)]

    else:
        id_path = (
            transcorpus_dir
            / domain_name
            / source_language
            / Path(str(corpus_id_url)).name
        )
        preview_ids = read_file_lines(id_path, start_at, start_at + count)

    preview_lines: list[list[str]] = []
    for lang in language_list:
        is_original = lang == source_language
        file_path = transcorpus_dir / domain_name / lang / corpus_file_name

        validate_file(file_path, is_original)

        # File length check
        with file_path.open(encoding="utf-8") as f:
            file_length = sum(1 for _ in f)

        if file_length > 1_000_000:
            click.secho(
                f"Warning: Large file ({file_length} lines). Consider splitting.",
                fg="yellow",
            )

        validate_indices(start_at, count, file_length)

        # Read relevant lines
        lines = read_file_lines(file_path, start_at, start_at + count)
        preview_lines.append(lines)

    return preview_lines, preview_ids, language_list, source_language


def title(
    corpus_id: Optional[str | int] = None,
    language: Optional[str | int] = None,
    source_language: Optional[str] = None,
):
    """
    Generate a title for the preview.

    Args:
        corpus_id (Optional[str | int]): The ID of the corpus.
        language (Optional[str | int]): The language of the corpus.
        source_language (Optional[str]): The source language of the corpus.

    Returns:
        str: The formatted title string.
    """
    if all(
        [
            corpus_id is None,
            language is None,
            source_language is None,
        ]
    ):
        raise ValueError(
            "At least one of id or both language and source_language must be provided."
        )
    if (not language and source_language) or (not source_language and language):
        raise ValueError(
            "If language is provided, both 'language' and 'source_language' must be provided."
        )
    source_or_target = (
        (
            "■ Source language"
            if language == source_language
            else "<-> Target language"
        )
        if language and source_language
        else ""
    )
    source_or_target = (
        f" {source_or_target}: {language} " if source_or_target else ""
    )

    domain_and_id = (
        f" Domain: {global_domain_name} | ID: {corpus_id} " if corpus_id else ""
    )
    return " " + domain_and_id + source_or_target


# # simple preview with no navigation
# def display_preview(
#     preview_lines: list,
#     preview_ids: list,
#     language_list: list,
#     source_language: M2M100_Languages,
# ):
#     for *pl, pi in zip(*preview_lines, preview_ids):
#         if len(pl) == 1:
#             print_one_language(pl[0], pi, language_list[0], source_language)
#         else:
#             print_two_languages(
#                 pl,
#                 pi,
#                 language_list,
#                 source_language,
#             )


def display_preview(
    preview_lines: list,
    preview_ids: list,
    language_list: list,
    source_language: M2M100_Languages,
):
    """
    Display the preview of articles with navigation.

    Args:
        preview_lines (list): The lines to preview.
        preview_ids (list): The IDs of the lines.
        language_list (list): The list of languages.
        source_language (str): The source language of the corpus.

    Example:
        display_preview(
            preview_lines,
            preview_ids,
            language_list,
            source_language,
        )
    """
    articles = []
    for *pl, pi in zip(*preview_lines, preview_ids):
        articles.append((pl, pi))

    current_idx = 0
    total_articles = len(articles)

    while True:
        # Clear the terminal (works on Unix/macOS and Windows Terminal)
        print("\033c", end="")

        # Display the current article
        pl, pi = articles[current_idx]
        if len(pl) == 1:
            print_one_language(pl[0], pi, language_list[0], source_language)
        else:
            print_two_languages(pl, pi, language_list, source_language)

        # Print navigation instructions
        print(f"\nArticle {current_idx + 1}/{total_articles}")
        print("← Previous | → Next | Q Quit")

        # Get user input
        key = readchar.readkey().lower()

        # Handle navigation
        if key == "q":
            print("Exiting preview.")
            break

        if key in ("\x1b[d", key == "\033[d"):
            if current_idx > 0:
                current_idx -= 1
            else:
                print("You are already at the first article.")
                break

        elif key in ("\x1b[c", key == "\033[c"):
            if current_idx < total_articles - 1:
                current_idx += 1
            else:
                print("You are already at the last article.")
                break


def print_one_language(
    preview_line: str,
    preview_id: str | int,
    language: M2M100_Languages,
    source_language: M2M100_Languages,
):
    """
    Print the preview for a single language next to each other taking into account the length of the terminal.

    Args:
        preview_line (str): The line to preview.
        preview_id (str | int): The ID of the line.
        language (str): The language of the line.
        source_language (str): The source language of the corpus.

    Example:
        print_one_language(
            preview_line,
            preview_id,
            language,
            source_language,
        )
    """
    print(shutil.get_terminal_size().columns * "\u2015")
    print(title(preview_id, language, source_language))
    print(shutil.get_terminal_size().columns * "\u2015")
    print(preview_line.strip())
    print(shutil.get_terminal_size().columns * " ")


def print_two_languages(
    preview_line: list,
    preview_id: str | int,
    language_list: list,
    source_language: M2M100_Languages,
):
    """
    Print the preview for two languages.

    Args:
        preview_line (list): The lines to preview.
        preview_id (str | int): The ID of the line.
        language_list (list): The list of languages.
        source_language (str): The source language of the corpus.

    Example:
        print_two_languages(
            preview_line,
            preview_id,
            language_list,
            source_language,
        )
    """
    term_width = shutil.get_terminal_size().columns
    sep_width = 3
    col_width = (term_width - sep_width) // 2

    language_title, preview_lines = [], []
    for i, pl in enumerate(preview_line):
        centered_title = title(
            language=language_list[i],
            source_language=source_language,
        )
        space_left = (col_width - len(centered_title)) // 2
        centered_title = (space_left * " " + centered_title + space_left * " ")[
            :col_width
        ]
        language_title.append(
            [centered_title]
            + [
                col_width * "\u2015",
            ],
        )
        preview_lines.append(textwrap.wrap(pl, width=col_width))

    for i, pl in enumerate(preview_lines):
        preview_lines[i] = language_title[i] + pl

    centered_title = title(corpus_id=preview_id)
    space_left = (term_width - len(centered_title)) // 2
    centered_title = (space_left * " " + centered_title + space_left * " ")[
        :term_width
    ]
    print(shutil.get_terminal_size().columns * "\u2015")
    print(centered_title)
    print(shutil.get_terminal_size().columns * " ")
    for i, (p0, p1) in enumerate(zip(*preview_lines)):
        sep = " " * sep_width if i == 0 else " \u2502 "
        print(f"{p0:<{col_width}}{sep}{p1:<{col_width}}")
    print(shutil.get_terminal_size().columns * " ")


def validate_max_two_languages(ctx, param, value):
    """Validate that the user has provided at most two languages.

    Args:
        ctx: The context object.
        param: The parameter object.
        value: The value provided by the user.

    Raises:
        click.BadParameter: If more than two languages are provided.
    """
    if len(value) > 2:
        raise click.BadParameter("You can specify at most two languages.")
    return value


@click.command()
@click.argument("corpus_name")
@click.option(
    "--language",
    "-l",
    callback=validate_max_two_languages,
    multiple=True,
    required=True,
    type=click.Choice(list(get_args(M2M100_Languages))),
    help="Language(s) of the corpus.",
)
@click.option(
    "--count",
    "-c",
    type=int,
    is_flag=False,
    default=5,
    help="Number of items to preview.",
)
@click.option(
    "--start-at",
    "-s",
    type=int,
    is_flag=False,
    default=0,
    help="Starting index for the preview.",
)
@click.option("--demo", "-d", is_flag=True, help="Run in demo mode.")
def preview(
    corpus_name: str,
    count: int,
    start_at: int,
    language: tuple[M2M100_Languages],
    demo: bool,
):
    """CLI command to preview a corpus for a specific domain.

    Args:     corpus_name (str): The name of the corpus to be previewed.
    count (int): The number of items to preview.     start_at (int): The
    starting index for the preview.     demo (bool): Whether to run in
    demo mode.

    Example:     $ python preview.py preview_corpus bio --demo --count 5
    --start-at 0
    """
    global global_domain_name
    global_domain_name = corpus_name
    file_suffix = SuffixModel(flag=demo)
    preview_lines, preview_ids, language_list, source_language = preview_txt(
        domain_name=corpus_name,
        file_suffix=file_suffix.get_suffix(),
        count=count,
        start_at=start_at,
        language=language,
    )
    display_preview(preview_lines, preview_ids, language_list, source_language)
