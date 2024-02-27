import csv
import logging
import re
import typing
from pathlib import Path

from espeak_phonemizer import Phonemizer
from gruut_ipa import IPA

from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_to_phonemes.espeak")


def do_text_to_phonemes_espeak(
    text_csv_path: typing.Union[str, Path], text_language: str, targets
):
    """Convert metadata text to phonemes using eSpeak"""

    phonemes_csv_path = Path(targets[0])
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    phonemizer = Phonemizer()

    clause_breakers_str = "".join(phonemizer.clause_breakers)
    clause_pattern = re.compile(f"([{re.escape(clause_breakers_str)}])")

    voice = text_language.lower().replace("_", "-")

    with open(text_csv_path, "r", encoding="utf-8") as text_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_file:
        reader = csv.reader(text_file, delimiter=_CTX.delimiter)
        writer = csv.writer(phonemes_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text
            raw_text = row[-1]
            phonemes_str = phonemizer.phonemize(
                raw_text,
                voice=voice,
                phoneme_separator=_CTX.config.phonemes.phoneme_separator,
                word_separator=_CTX.config.phonemes.word_separator,
                keep_clause_breakers=True,
            ).strip()

            if not phonemes_str:
                _LOGGER.warning("No phonemes for %s", row)
                continue

            word_phonemes = phonemes_str.split(_CTX.config.phonemes.word_separator)
            sub_word_phonemes = []

            for w_phonemes in word_phonemes:
                for clause_phonemes in clause_pattern.split(w_phonemes):
                    if clause_phonemes:
                        sub_word_phonemes.append(clause_phonemes)

            if _CTX.config.phonemes.break_phonemes_into_graphemes:
                broken_word_phonemes = []
                for w_phonemes in sub_word_phonemes:
                    broken_w_phonemes = []
                    for phoneme in w_phonemes.split(
                        _CTX.config.phonemes.phoneme_separator
                    ):
                        for grapheme in IPA.graphemes(phoneme):
                            broken_w_phonemes.append(grapheme)

                    broken_word_phonemes.append(
                        _CTX.config.phonemes.phoneme_separator.join(broken_w_phonemes)
                    )

                sub_word_phonemes = broken_word_phonemes

            # Output format:
            # id|[speaker]|text|phonemes
            final_phonemes_str = _CTX.config.phonemes.word_separator.join(
                sub_word_phonemes
            )
            writer.writerow((*row, final_phonemes_str))
