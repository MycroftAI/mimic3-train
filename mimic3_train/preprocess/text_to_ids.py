#!/usr/bin/env python3
import csv
import logging
import typing
from pathlib import Path

from vits_train.config import DatasetConfig, MetadataFormat, Phonemizer, Aligner

from .context import Context
from .text_to_phonemes import do_phoneme_counts, do_total_phoneme_counts
from .phonemes_to_ids import task_phoneme_map, do_phonemes_to_ids

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess")


def task_text_to_ids():
    """Convert dataset text/phonemes to phoneme ids"""
    phoneme_map_path = _CTX.phoneme_map_path
    phoneme_counts_paths = []

    for dataset in _CTX.config.datasets:
        dataset_dir = _CTX.dataset_dir(dataset)

        for split in _CTX.splits:
            ids_path = _CTX.ids_path(dataset, split)
            phonemes_path = _CTX.phonemes_path(dataset, split)

            if dataset.metadata_format == MetadataFormat.PHONEME_IDS:
                # Ids should already exist
                assert _exists_and_not_empty(ids_path)

                # Make audio list that uses entire audio file
                audio_csv_path = _CTX.audio_list_path(dataset, split)
                yield {
                    "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
                    "actions": [(do_audio_list, [ids_path, dataset])],
                    "file_dep": [ids_path],
                    "targets": [audio_csv_path],
                }
                continue

            if dataset.metadata_format == MetadataFormat.PHONEMES:
                # Phonemnes should already exist
                assert _exists_and_not_empty(phonemes_path)

                # Make audio list that uses entire audio file
                audio_csv_path = _CTX.audio_list_path(dataset, split)
                yield {
                    "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
                    "actions": [(do_audio_list, [phonemes_path, dataset])],
                    "file_dep": [phonemes_path],
                    "targets": [audio_csv_path],
                }

            if dataset.metadata_format == MetadataFormat.TEXT:
                # text -> phonemes
                phonemizer = _CTX.config.phonemizer
                assert phonemizer is not None, "config.phonemizer is required"
                text_aligner = _CTX.config.text_aligner

                if phonemizer == Phonemizer.GRUUT:
                    if text_aligner and (text_aligner.aligner == Aligner.KALDI_ALIGN):
                        yield from _gruut_align(dataset, split)
                    else:
                        yield from _gruut_no_align(dataset, split)

                elif phonemizer == Phonemizer.ESPEAK:
                    yield from _espeak_no_align(dataset, split)
                elif phonemizer == Phonemizer.EPITRAN:
                    yield from _epitran_no_align(dataset, split)
                elif phonemizer == Phonemizer.SYMBOLS:
                    yield from _symbols_no_align(dataset, split)
                else:
                    raise ValueError(f"Unknown phonemizer: {phonemizer}")

                # Generate phoneme counts for this dataset
                phoneme_counts_path = dataset_dir / "phoneme_counts.json"
                yield {
                    "name": str(phoneme_counts_path.relative_to(_CTX.output_dir)),
                    "actions": [(do_phoneme_counts, [phonemes_path])],
                    "file_dep": [phonemes_path],
                    "targets": [phoneme_counts_path],
                }

                phoneme_counts_paths.append(phoneme_counts_path)

            # phonemes -> ids
            yield {
                "name": str(ids_path.relative_to(_CTX.output_dir)),
                "actions": [(do_phonemes_to_ids, [phonemes_path, phoneme_map_path])],
                "file_dep": [phonemes_path, phoneme_map_path],
                "targets": [ids_path],
            }

    if phoneme_counts_paths:
        # Generate phoneme counts for all datasets
        total_phoneme_counts_path = _CTX.output_dir / "phoneme_counts.json"
        yield {
            "name": str(total_phoneme_counts_path.relative_to(_CTX.output_dir)),
            "actions": [(do_total_phoneme_counts, [phoneme_counts_paths])],
            "file_dep": phoneme_counts_paths,
            "targets": [total_phoneme_counts_path],
        }


# -----------------------------------------------------------------------------


def _symbols_no_align(dataset: DatasetConfig, split: str):
    # Use symbols directly
    from .text_to_phonemes.symbols import do_text_to_symbols

    text_path = _CTX.text_path(dataset, split)
    phonemes_path = _CTX.phonemes_path(dataset, split)

    yield {
        "name": str(phonemes_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_symbols, [text_path])],
        "file_dep": [text_path],
        "targets": [phonemes_path],
    }

    # Make audio list that uses entire audio file
    audio_csv_path = _CTX.audio_list_path(dataset, split)
    yield {
        "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
        "actions": [(do_audio_list, [text_path, dataset])],
        "file_dep": [text_path],
        "targets": [audio_csv_path],
    }


def _espeak_no_align(dataset: DatasetConfig, split: str):
    # Use eSpeak for phonemization
    from .text_to_phonemes.espeak import do_text_to_phonemes_espeak

    text_path = _CTX.text_path(dataset, split)
    phonemes_path = _CTX.phonemes_path(dataset, split)

    dataset_language = _CTX.dataset_language(dataset)
    yield {
        "name": str(phonemes_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_phonemes_espeak, [text_path, dataset_language])],
        "file_dep": [text_path],
        "targets": [phonemes_path],
    }

    # Make audio list that uses entire audio file
    audio_csv_path = _CTX.audio_list_path(dataset, split)
    yield {
        "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
        "actions": [(do_audio_list, [text_path, dataset])],
        "file_dep": [text_path],
        "targets": [audio_csv_path],
    }


def _epitran_no_align(dataset: DatasetConfig, split: str):
    # Use epitran for phonemization
    from .text_to_phonemes.epitran import do_text_to_phonemes_epitran

    text_path = _CTX.text_path(dataset, split)
    phonemes_path = _CTX.phonemes_path(dataset, split)

    dataset_language = _CTX.dataset_language(dataset)
    yield {
        "name": str(phonemes_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_phonemes_epitran, [text_path, dataset_language])],
        "file_dep": [text_path],
        "targets": [phonemes_path],
    }

    # Make audio list that uses entire audio file
    audio_csv_path = _CTX.audio_list_path(dataset, split)
    yield {
        "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
        "actions": [(do_audio_list, [text_path, dataset])],
        "file_dep": [text_path],
        "targets": [audio_csv_path],
    }


def _gruut_align(dataset: DatasetConfig, split: str):
    from .text_to_phonemes.gruut import do_text_to_spoken_gruut
    from .text_align import (
        do_text_to_words,
        do_words_to_lexicon,
        do_alignment_model,
        do_forced_alignment,
        do_align_to_csv,
        do_phonemes_from_align,
    )

    text_path = _CTX.text_path(dataset, split)
    phonemes_path = _CTX.phonemes_path(dataset, split)

    # text -> spoken text only
    dataset_language = _CTX.dataset_language(dataset)
    text_spoken_path = text_path.parent / f"{text_path.stem}_spoken{text_path.suffix}"
    yield {
        "name": str(text_spoken_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_spoken_gruut, [text_path, dataset_language])],
        "file_dep": [text_path],
        "targets": [text_spoken_path],
    }

    # spoken text -> unique words
    words_path = text_spoken_path.parent / "words.txt"
    yield {
        "name": str(words_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_words, [text_spoken_path])],
        "file_dep": [text_spoken_path],
        "targets": [words_path],
    }

    # unique words -> pronunciation dictionary (lexicon)
    lexicon_path = words_path.parent / "lexiconp.txt"
    yield {
        "name": str(lexicon_path.relative_to(_CTX.output_dir)),
        "actions": [(do_words_to_lexicon, [words_path, dataset])],
        "file_dep": [words_path],
        "targets": [lexicon_path],
    }

    # pronunciation dictionary (lexicon) -> data/lang/L.fst
    align_dir = lexicon_path.parent / "align"
    align_model_dir = align_dir / "custom_model"
    lang_fst = align_model_dir / "data" / "lang" / "L.fst"
    yield {
        "name": str(align_model_dir.relative_to(_CTX.output_dir)),
        "actions": [(do_alignment_model, [lexicon_path, align_model_dir, dataset])],
        "file_dep": [lexicon_path],
        "targets": [lang_fst],
    }

    # align text/audio
    json_align_path = text_path.parent / f"{text_path.stem}_align.jsonl"
    yield {
        "name": str(json_align_path.relative_to(_CTX.output_dir)),
        "actions": [
            (do_forced_alignment, [text_spoken_path, align_model_dir, dataset])
        ],
        "file_dep": [text_path, lang_fst],
        "targets": [json_align_path],
    }

    # get begin/end of trimmed audio and phonemes
    audio_list_path = _CTX.audio_list_path(dataset, split)
    yield {
        "name": str(audio_list_path.relative_to(_CTX.output_dir)),
        "actions": [
            (
                do_align_to_csv,
                [json_align_path, text_spoken_path, dataset, dataset_language],
            )
        ],
        "file_dep": [json_align_path, text_spoken_path],
        "targets": [audio_list_path],
    }

    # write phonemes file
    yield {
        "name": str(phonemes_path.relative_to(_CTX.output_dir)),
        "actions": [(do_phonemes_from_align, [audio_list_path, text_spoken_path])],
        "file_dep": [audio_list_path, text_spoken_path],
        "targets": [phonemes_path],
    }


def _gruut_no_align(dataset: DatasetConfig, split: str):
    from .text_to_phonemes.gruut import do_text_to_phonemes_gruut

    # Use gruut directly on text
    text_path = _CTX.text_path(dataset, split)
    phonemes_path = _CTX.phonemes_path(dataset, split)

    dataset_language = _CTX.dataset_language(dataset)
    yield {
        "name": str(phonemes_path.relative_to(_CTX.output_dir)),
        "actions": [(do_text_to_phonemes_gruut, [text_path, dataset_language])],
        "file_dep": [text_path],
        "targets": [phonemes_path],
    }

    # Make audio list that uses entire audio file
    audio_csv_path = _CTX.audio_list_path(dataset, split)
    yield {
        "name": str(audio_csv_path.relative_to(_CTX.output_dir)),
        "actions": [(do_audio_list, [text_path, dataset])],
        "file_dep": [text_path],
        "targets": [audio_csv_path],
    }


# -----------------------------------------------------------------------------


def do_audio_list(
    text_csv_path: typing.Union[str, Path], dataset_config: DatasetConfig, targets
):
    assert (
        dataset_config.audio_dir is not None
    ), f"Audio directory is required: {dataset_config}"
    audio_dir = Path(dataset_config.audio_dir)

    if not audio_dir.is_absolute():
        audio_dir = _CTX.output_dir / str(audio_dir)

    audio_csv_path = Path(targets[0])
    audio_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text_csv_path, "r", encoding="utf-8") as text_csv_file, open(
        audio_csv_path, "w", encoding="utf-8"
    ) as audio_csv_file:
        reader = csv.reader(text_csv_file, delimiter=_CTX.delimiter)
        writer = csv.writer(audio_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            utt_id, phonemes_str = row[0], row[-1]

            # end_ms = 0 indicates entire audio file should be used
            start_ms, end_ms = 0, 0

            audio_path = _CTX.find_audio_file(audio_dir, utt_id)
            if audio_path is None:
                _LOGGER.warning(
                    "Missing audio file for dataset '%s': %s",
                    dataset_config.name,
                    utt_id,
                )
                continue

            writer.writerow(
                (*row[:-1], start_ms, end_ms, str(audio_path.absolute()), phonemes_str)
            )


# -----------------------------------------------------------------------------


def _exists_and_not_empty(file_path: typing.Union[str, Path]) -> bool:
    """True if file exists and is not empty"""
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    return file_path.stat().st_size > 0
