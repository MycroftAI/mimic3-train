import csv
import logging
import json
import re
import typing
from collections import Counter
from pathlib import Path

from gruut_ipa import IPA

from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_to_phonemes")


def do_total_phoneme_counts(
    phoneme_counts_paths: typing.List[typing.Union[str, Path]], targets
):
    """Generate JSON file with total phoneme counts from JSON files"""
    total_phoneme_counts_path = Path(targets[0])
    total_phoneme_counts_path.parent.mkdir(parents=True, exist_ok=True)

    total_phoneme_counts = Counter()

    with open(
        total_phoneme_counts_path, "w", encoding="utf-8"
    ) as total_phoneme_counts_file:
        for phoneme_counts_path in phoneme_counts_paths:
            with open(
                phoneme_counts_path, "r", encoding="utf-8"
            ) as phoneme_counts_file:
                # Input format:
                # {
                #     "<phoneme>": <count>,
                #     "<phoneme>": <count>,
                #     ...
                # }
                phoneme_counts = json.load(phoneme_counts_file)
                for phoneme, count in phoneme_counts.items():
                    total_phoneme_counts[phoneme] += count

        # Output format:
        # {
        #     "<phoneme>": <total count>,
        #     "<phoneme>": <total count>,
        #     ...
        # }
        json.dump(
            total_phoneme_counts,
            total_phoneme_counts_file,
            indent=4,
            ensure_ascii=False,
        )


def do_phoneme_counts(phonemes_csv_path: typing.Union[str, Path], targets):
    """Generate JSON file with phoneme counts from a single CSV file"""
    phoneme_counts_path = Path(targets[0])
    phoneme_counts_path.parent.mkdir(parents=True, exist_ok=True)

    phoneme_counts = Counter()

    with open(phonemes_csv_path, "r", encoding="utf-8") as phonemes_file, open(
        phoneme_counts_path, "w", encoding="utf-8"
    ) as phoneme_counts_file:
        reader = csv.reader(phonemes_file, delimiter=_CTX.delimiter)
        for row in reader:
            # Input format:
            # id|[speaker]|text|phonemes
            phonemes_str = row[-1]

            for word_phonemes in _CTX.config.phonemes.split_word_phonemes(phonemes_str):
                for phoneme in word_phonemes:
                    phoneme = IPA.without_stress(phoneme)
                    if phoneme:
                        phoneme_counts[phoneme] += 1

        # Output format:
        # {
        #     "<phoneme>": <count>,
        #     "<phoneme>": <count>,
        #     ...
        # }
        json.dump(phoneme_counts, phoneme_counts_file, indent=4, ensure_ascii=False)
