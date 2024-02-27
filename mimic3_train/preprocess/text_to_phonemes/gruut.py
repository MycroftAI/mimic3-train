import csv
import logging
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gruut

from vits_train.config import TextCasing
from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_to_phonemes.gruut")


def do_text_to_phonemes_gruut(
    text_csv_path: typing.Union[str, Path], text_language: str, targets
):
    """Convert metadata text to phonemes using gruut"""
    phonemes_csv_path = Path(targets[0])
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_csv_file:
        reader = csv.reader(text_file, delimiter=_CTX.delimiter)
        writer = csv.writer(phonemes_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text
            raw_text = row[-1]
            word_phonemes = []

            for sentence in gruut.sentences(raw_text, lang=text_language):
                for word in sentence:
                    if word.phonemes:
                        word_phonemes.append(word.phonemes)

            if not word_phonemes:
                _LOGGER.warning("No phonemes for %s", row)
                continue

            # Output format:
            # id|[speaker]|text|phonemes
            phonemes_str = _CTX.config.phonemes.join_word_phonemes(word_phonemes)
            writer.writerow((*row, phonemes_str))


def do_text_to_spoken_gruut(
    text_csv_path: typing.Union[str, Path], text_language: str, targets
):
    """Convert metadata text to spoken text using gruut"""
    spoken_csv_path = Path(targets[0])
    spoken_csv_path.parent.mkdir(parents=True, exist_ok=True)

    casing_func: typing.Optional[typing.Callable[[str], str]] = None
    aligner_config = _CTX.config.text_aligner

    if aligner_config:
        if aligner_config.casing == TextCasing.LOWER:
            casing_func = str.lower
        elif aligner_config.casing == TextCasing.UPPER:
            casing_func = str.upper

    def add_spoken(row):
        # Input format:
        # id|[speaker]|text
        raw_text = row[-1]
        spoken_texts = []

        for sentence in gruut.sentences(
            raw_text, lang=text_language, pos=False, phonemes=False
        ):
            spoken_texts.append(sentence.text_spoken)

        spoken_text = " ".join(spoken_texts)

        if casing_func is not None:
            spoken_text = casing_func(spoken_text)

        # Output format:
        # id|[speaker]|text|spoken text
        return (*row, spoken_text)

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        spoken_csv_path, "w", encoding="utf-8"
    ) as spoken_csv_file, ThreadPoolExecutor() as executor:
        reader = csv.reader(text_file, delimiter=_CTX.delimiter)
        writer = csv.writer(spoken_csv_file, delimiter=_CTX.delimiter)

        # Process in parallel
        for new_row in executor.map(add_spoken, reader):
            writer.writerow(new_row)
