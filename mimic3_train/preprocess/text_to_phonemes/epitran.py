import csv
import logging
import re
import typing
import unicodedata
from pathlib import Path

import epitran
from gruut_ipa import IPA

from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_to_phonemes.epitran")


def do_text_to_phonemes_epitran(
    text_csv_path: typing.Union[str, Path], text_language: str, targets
):
    """Convert metadata text to phonemes using epitran"""

    phonemes_csv_path = Path(targets[0])
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    epi = epitran.Epitran(text_language)

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_file:
        reader = csv.reader(text_file, delimiter=_CTX.delimiter)
        writer = csv.writer(phonemes_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text
            raw_text = row[-1]
            phonemes_str = epi.transliterate(raw_text).strip()

            if not phonemes_str:
                _LOGGER.warning("No phonemes for %s", row)
                continue

            word_phonemes = []

            # Split into words by whitespace
            for word_phoneme_str in phonemes_str.split():
                word_phoneme_str = word_phoneme_str.strip()
                if not word_phoneme_str:
                    continue

                if _CTX.config.phonemes.break_phonemes_into_codepoints:
                    # Split into codepoints and recombine with phoneme separator
                    codepoints = to_codepoints(word_phoneme_str)
                    codepoints_str = _CTX.config.phonemes.phoneme_separator.join(
                        codepoints
                    )
                    word_phonemes.append(codepoints_str)
                else:
                    # Split graphemes and recombine with phoneme separator
                    graphemes = IPA.graphemes(word_phoneme_str)
                    graphemes_str = _CTX.config.phonemes.phoneme_separator.join(
                        graphemes
                    )
                    word_phonemes.append(graphemes_str)

            # Output format:
            # id|[speaker]|text|phonemes
            final_phonemes_str = _CTX.config.phonemes.word_separator.join(word_phonemes)
            writer.writerow((*row, final_phonemes_str))


def to_codepoints(s: str) -> typing.List[str]:
    """Split string into a list of codepoints"""
    return list(unicodedata.normalize("NFC", s))
