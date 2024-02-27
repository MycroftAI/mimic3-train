#!/usr/bin/env python3
import csv
import logging
import typing
from pathlib import Path

from vits_train.config import MetadataFormat

from .context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess")


def task_phoneme_map():
    """Learn mapping from phonemes to integer ids, and write phoneme map (phonemes.txt)"""
    if not _CTX.learn_phonemes:
        # Don't try to learn phonemes
        return

    phoneme_map_path = _CTX.phoneme_map_path
    phonemes_paths = []

    for dataset in _CTX.config.datasets:
        if dataset.metadata_format == MetadataFormat.PHONEME_IDS:
            # No phonemes we can learn from
            continue

        for split in _CTX.splits:
            phonemes_paths.append(_CTX.phonemes_path(dataset, split))

    if phonemes_paths:
        yield {
            "name": phoneme_map_path.name,
            "actions": [(do_phoneme_map, [phonemes_paths])],
            "file_dep": phonemes_paths,
            "targets": [phoneme_map_path],
        }


# -----------------------------------------------------------------------------


def do_phonemes_to_ids(
    phonemes_csv_path: typing.Union[str, Path],
    phoneme_map_path: typing.Union[str, Path],
    targets,
):
    """Map phonemes to integer ids according to existing phoneme map (phonemes.txt)"""
    import phonemes2ids

    ids_csv_path = Path(targets[0])
    ids_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(phoneme_map_path, "r", encoding="utf-8") as map_file:
        # Map format:
        # <id> <phoneme>
        # <id> <phoneme>
        # ...
        phoneme_to_id = phonemes2ids.load_phoneme_ids(map_file)

    with open(phonemes_csv_path, "r", encoding="utf-8") as phonemes_file, open(
        ids_csv_path, "w", encoding="utf-8"
    ) as ids_file:
        reader = csv.reader(phonemes_file, delimiter=_CTX.delimiter)
        writer = csv.writer(ids_file, delimiter=_CTX.delimiter)

        num_utterances = 0
        num_phonemes_too_small = 0
        num_phonemes_too_large = 0

        for row in reader:
            # Input format:
            # id|[speaker]|text|phonemes
            num_utterances += 1
            phonemes_str = row[-1]
            word_phonemes = _CTX.config.phonemes.split_word_phonemes(phonemes_str)

            phoneme_ids = phonemes2ids.phonemes2ids(
                word_phonemes=word_phonemes,
                phoneme_to_id=phoneme_to_id,
                pad=_CTX.config.phonemes.pad,
                bos=_CTX.config.phonemes.bos,
                eos=_CTX.config.phonemes.eos,
                auto_bos_eos=_CTX.config.phonemes.auto_bos_eos,
                blank=_CTX.config.phonemes.blank,
                blank_word=_CTX.config.phonemes.blank_word,
                blank_between=_CTX.config.phonemes.blank_between,
                blank_at_start=_CTX.config.phonemes.blank_at_start,
                blank_at_end=_CTX.config.phonemes.blank_at_end,
                simple_punctuation=_CTX.config.phonemes.simple_punctuation,
                punctuation_map=_CTX.config.phonemes.punctuation_map,
                separate=_CTX.config.phonemes.separate,
                separate_graphemes=_CTX.config.phonemes.separate_graphemes,
                separate_tones=_CTX.config.phonemes.separate_tones,
                tone_before=_CTX.config.phonemes.tone_before,
                phoneme_map=_CTX.config.phonemes.phoneme_map,
                fail_on_missing=True,
            )

            # Drop utterances that are too small/large
            if (_CTX.config.min_seq_length is not None) and (
                len(phoneme_ids) < _CTX.config.min_seq_length
            ):
                num_phonemes_too_small += 1
                continue

            if (_CTX.config.max_seq_length is not None) and (
                len(phoneme_ids) > _CTX.config.max_seq_length
            ):
                num_phonemes_too_large += 1
                continue

            # Output format:
            # id|[speaker]|text|phonemes|phoneme ids
            phoneme_ids_str = " ".join(str(p_id) for p_id in phoneme_ids)
            writer.writerow((*row, phoneme_ids_str))

        if num_phonemes_too_small > 0:
            _LOGGER.warning(
                "%s/%s utterance(s) dropped whose phoneme length was smaller than %s (%s)",
                num_phonemes_too_small,
                num_utterances,
                _CTX.config.min_seq_length,
                ids_csv_path,
            )

        if num_phonemes_too_large > 0:
            _LOGGER.warning(
                "%s/%s utterance(s) dropped whose phoneme length was larger than %s (%s)",
                num_phonemes_too_large,
                num_utterances,
                _CTX.config.max_seq_length,
                ids_csv_path,
            )


# -----------------------------------------------------------------------------


def do_phoneme_map(
    phoneme_csv_paths: typing.Iterable[typing.Union[str, Path]], targets
):
    """Learn mapping from phonemes to integer ids, and write phoneme map (phonemes.txt)"""
    import phonemes2ids

    phoneme_map_path = Path(targets[0])
    phoneme_map_path.parent.mkdir(parents=True, exist_ok=True)

    word_phonemes: typing.List[typing.List[str]] = []

    for phoneme_csv_path in phoneme_csv_paths:
        with open(phoneme_csv_path, "r", encoding="utf-8") as phonemes_file:
            reader = csv.reader(phonemes_file, delimiter=_CTX.delimiter)
            for row in reader:
                # Output format:
                # id|[speaker]|text|phonemes
                phonemes_str = row[-1]
                word_phonemes.extend(
                    _CTX.config.phonemes.split_word_phonemes(phonemes_str)
                )

    all_phonemes: typing.Set[str] = set()
    phoneme_to_id: typing.Dict[str, int] = dict(
        _CTX.config.phonemes.phoneme_to_id or {}
    )

    def add_phoneme(phoneme: str):
        """Add phoneme with new id"""
        if phoneme and (phoneme not in phoneme_to_id):
            phoneme_to_id[phoneme] = len(phoneme_to_id)

    add_phoneme(_CTX.config.phonemes.pad)
    add_phoneme(_CTX.config.phonemes.bos)
    add_phoneme(_CTX.config.phonemes.eos)
    add_phoneme(_CTX.config.phonemes.minor_break)
    add_phoneme(_CTX.config.phonemes.major_break)
    add_phoneme(_CTX.config.phonemes.blank)
    add_phoneme(_CTX.config.phonemes.blank_word)

    if _CTX.config.phonemes.separate:
        # Add stress symbols
        for stress in sorted(_CTX.config.phonemes.separate):
            add_phoneme(stress)

    phonemes2ids.learn_phoneme_ids(
        word_phonemes=word_phonemes,
        all_phonemes=all_phonemes,
        simple_punctuation=_CTX.config.phonemes.simple_punctuation,
        punctuation_map=_CTX.config.phonemes.punctuation_map,
        separate=_CTX.config.phonemes.separate,
        separate_graphemes=_CTX.config.phonemes.separate_graphemes,
        separate_tones=_CTX.config.phonemes.separate_tones,
        phoneme_map=_CTX.config.phonemes.phoneme_map,
    )

    for phoneme in sorted(all_phonemes):
        add_phoneme(phoneme)

    # Write phoneme map
    with open(phoneme_map_path, "w", encoding="utf-8") as map_file:
        # Map format:
        # <id> <phoneme>
        # <id> <phoneme>
        # ...
        for phoneme, phoneme_idx in phoneme_to_id.items():
            print(phoneme_idx, phoneme, file=map_file)
