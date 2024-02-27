import csv
import logging
import re
import typing
from collections import Counter
from pathlib import Path

from gruut_ipa import IPA

from vits_train.config import DatasetConfig
from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_to_phonemes.symbols")


def do_text_to_symbols(text_csv_path: typing.Union[str, Path], targets):
    """Convert metadata text to symbols as 'phonemes'"""
    symbols = _CTX.config.phonemes.symbols
    assert symbols, "No symbols defined in phonemes config"

    symbols_set = set(symbols)
    phoneme_map = _CTX.config.phonemes.phoneme_map or {}
    word_separator = _CTX.config.phonemes.word_separator
    phoneme_separator = _CTX.config.phonemes.phoneme_separator

    phonemes_csv_path = Path(targets[0])
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    missing_graphemes = Counter()
    num_skipped = 0

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_file:
        reader = csv.reader(text_file, delimiter=_CTX.delimiter)
        writer = csv.writer(phonemes_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text
            raw_text = row[-1]

            word_phonemes: typing.List[typing.List[str]] = []
            word_phonemes.append([])

            skip_row = False
            for grapheme in IPA.graphemes(raw_text):
                if grapheme == word_separator:
                    word_phonemes.append([])
                elif grapheme in symbols_set:
                    phoneme = phoneme_map.get(grapheme, grapheme)
                    word_phonemes[-1].append(phoneme)
                else:
                    missing_graphemes[grapheme] += 1
                    skip_row = True

            if skip_row:
                num_skipped += 1
                continue

            # Output format:
            # id|[speaker]|text|phonemes
            phonemes_str = _CTX.config.phonemes.word_separator.join(
                _CTX.config.phonemes.phoneme_separator.join(w_phonemes)
                for w_phonemes in word_phonemes
                if w_phonemes
            )

            if not phonemes_str:
                _LOGGER.warning("No phonemes for %s", row)
                continue

            writer.writerow((*row, phonemes_str))

    if missing_graphemes:
        _LOGGER.warning("Missing symbols: %s", missing_graphemes.most_common())

    if num_skipped > 0:
        _LOGGER.warning("Skipped %s utterance(s) due to missing symbols", num_skipped)
