#!/usr/bin/env python3
import csv
import math
import json
import logging
import shutil
import typing
from pathlib import Path

import gruut
import phonetisaurus
from gruut_ipa import IPA
from kaldi_align import KaldiAligner, Utterance

from vits_train.config import DatasetConfig
from vits_train.preprocess.context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.text_align")


def do_text_to_words(text_csv_path: typing.Union[str, Path], targets):
    words_path = Path(targets[0])
    words_path.parent.mkdir(parents=True, exist_ok=True)

    all_words: typing.Set[str] = set()

    with open(text_csv_path, "r", encoding="utf-8") as text_csv_file:
        reader = csv.reader(text_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text|spoken text
            text = row[-1]
            text_words = text.split()
            all_words.update(text_words)

    with open(words_path, "w", encoding="utf-8") as words_file:
        # Output format:
        # <word>
        # <word>
        # ...
        for word in sorted(all_words):
            print(word, file=words_file)


def do_words_to_lexicon(
    words_path: typing.Union[str, Path], dataset_config: DatasetConfig, targets
):
    lexicon_out_path = Path(targets[0])
    lexicon_out_path.parent.mkdir(parents=True, exist_ok=True)

    aligner = _CTX.dataset_aligner(dataset_config)
    assert aligner is not None, f"No aligner for dataset: {dataset_config.name}"
    assert isinstance(aligner, KaldiAligner)

    drop_stress = _CTX.config.phonemes.drop_stress

    # <unk>
    oov_path = aligner.model_dir / "data" / "lang" / "oov.txt"
    oov = oov_path.read_text().strip()

    lexicon_in_path = aligner.model_dir / "data" / "local" / "dict" / "lexiconp.txt"
    g2p_model_path = aligner.model_dir / "g2p.fst"

    with open(words_path, "r", encoding="utf-8") as words_file, open(
        lexicon_in_path, "r", encoding="utf-8"
    ) as lexicon_in_file, open(
        lexicon_out_path, "w", encoding="utf-8"
    ) as lexicon_out_file:
        # Gather words used in utterances
        words_needed: typing.Set[str] = set()
        words_needed.add(oov)

        for line in words_file:
            # Input format:
            # <word>
            # <word>
            # ...
            word = line.strip()
            if word:
                words_needed.add(word)

        # Lexicon format:
        # <word> <probability> <phoneme> <phoneme> ...
        for line in lexicon_in_file:
            line = line.strip()
            if not line:
                continue

            word = line.split(maxsplit=1)[0]
            if word in words_needed:
                # Already in the lexicon
                words_needed.remove(word)
                print(line, file=lexicon_out_file)

        if words_needed:
            # Guess pronunciations
            word_phoneme_strs = set()

            for word, phonemes in phonetisaurus.predict(
                words_needed, model_path=g2p_model_path, nbest=5
            ):
                if drop_stress:
                    phonemes = list(
                        filter(None, (IPA.without_stress(p) for p in phonemes))
                    )

                phoneme_str = " ".join(phonemes)
                if phoneme_str in word_phoneme_strs:
                    continue

                print(
                    word, " ", 1.0, "\t", phoneme_str, file=lexicon_out_file, sep="",
                )

                word_phoneme_strs.add(phoneme_str)


def do_alignment_model(
    lexicon_path: typing.Union[str, Path],
    model_dir: typing.Union[str, Path],
    dataset_config: DatasetConfig,
    targets,
):
    """Generate aligner model with custom lexicon"""
    aligner = _CTX.dataset_aligner(dataset_config)
    assert aligner is not None, f"No aligner for dataset: {dataset_config.name}"
    assert isinstance(aligner, KaldiAligner)

    aligner._prepare_output_dir()
    path_sh = aligner.output_dir / "path.sh"
    if not path_sh.exists():
        path_sh.write_text("")

    # <unk>
    oov_path = aligner.model_dir / "data" / "lang" / "oov.txt"
    oov = oov_path.read_text().strip()

    # Copy model to dataset directory
    model_dir = Path(model_dir)
    if model_dir.is_dir():
        shutil.rmtree(model_dir)

    shutil.copytree(aligner.model_dir, model_dir)

    # Re-create data/lang from local lexicon
    data_local_dict = model_dir / "data" / "local" / "dict"
    shutil.copy(lexicon_path, data_local_dict)

    data_local_lang = model_dir / "data" / "local" / "lang"
    data_lang = model_dir / "data" / "lang"
    aligner._run_command(
        "utils/prepare_lang.sh",
        str(data_local_dict.absolute()),
        oov,
        str(data_local_lang.absolute()),
        str(data_lang.absolute()),
    )


def do_forced_alignment(
    text_csv_path: typing.Union[str, Path],
    model_dir: typing.Union[str, Path],
    dataset_config: DatasetConfig,
    targets,
):
    """Generate forced alignment between spoken text and audio"""
    assert (
        dataset_config.audio_dir is not None
    ), f"Audio directory is required for alignment: {dataset_config}"

    audio_dir = Path(dataset_config.audio_dir)
    if not audio_dir.is_absolute():
        audio_dir = _CTX.output_dir / str(audio_dir)

    aligner = _CTX.dataset_aligner(dataset_config)
    assert aligner is not None, f"No aligner for dataset: {dataset_config.name}"

    aligner.model_path = model_dir

    json_align_path = Path(targets[0])
    json_align_path.parent.mkdir(parents=True, exist_ok=True)

    utterances = []

    with open(text_csv_path, "r", encoding="utf-8") as text_csv_file:
        reader = csv.reader(text_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text|spoken text
            utt_id, spoken_text = row[0], row[-1]

            audio_path = _CTX.find_audio_file(audio_dir, utt_id)
            if audio_path is None:
                _LOGGER.warning(
                    "Missing audio file for dataset '%s': %s",
                    dataset_config.name,
                    utt_id,
                )
                continue

            if dataset_config.multispeaker:
                speaker = row[1]
            else:
                speaker = dataset_config.name

            utterances.append(
                Utterance(
                    id=utt_id, speaker=speaker, text=spoken_text, audio_path=audio_path
                )
            )

    assert utterances, f"No utterances for dataset: {dataset_config.name}"
    aligned_utterances = aligner.align(utterances)

    with open(json_align_path, "w", encoding="utf-8") as json_align_file:
        # Output format:
        # { "key": value, ... }
        # ...
        for aligned_utt in aligned_utterances:
            json_line = json.dumps(aligned_utt.to_dict(), ensure_ascii=False)
            print(json_line, file=json_align_file)


def do_align_to_csv(
    json_align_path: typing.Union[str, Path],
    text_csv_path: typing.Union[str, Path],
    dataset_config: DatasetConfig,
    dataset_language: str,
    targets,
):
    """Create CSV file from alignments with begin/end timestamps for audio and aligned phonemes"""
    audio_dir = Path(dataset_config.audio_dir)
    if not audio_dir.is_absolute():
        audio_dir = _CTX.output_dir / str(audio_dir)

    align_csv_path = Path(targets[0])
    align_csv_path.parent.mkdir(parents=True, exist_ok=True)

    min_sec = 0.5
    buffer_sec = 0.15
    skip_phones = {"SIL", "SPN", "NSN"}

    # id -> (start_ms, end_ms, audio_path, phonemes_str)
    utt_alignments = {}

    with open(json_align_path, "r", encoding="utf-8") as json_align_file:
        for line in json_align_file:
            line = line.strip()
            if not line:
                continue

            has_unknown_words = False
            align_obj = json.loads(line)
            utt_id = align_obj["id"]

            audio_path = _CTX.find_audio_file(audio_dir, utt_id)
            if audio_path is None:
                _LOGGER.warning(
                    "Missing audio file for dataset '%s': %s",
                    dataset_config.name,
                    utt_id,
                )
                continue

            # Find sentence boundaries (exclude <eps> before and after)
            start_sec = -1.0
            end_sec = -1.0

            all_word_phonemes: typing.List[typing.List[str]] = []

            for word in align_obj["words"]:
                if word["text"] == "<unk>":
                    has_unknown_words = True
                    break

                word_phonemes = [
                    phone["text"]
                    for phone in word["phones"]
                    if phone["text"] not in skip_phones
                ]
                if word_phonemes:
                    all_word_phonemes.append(word_phonemes)

                if word["text"] != "<eps>":
                    if start_sec < 0:
                        start_sec = word["phones"][0]["start_sec"]
                    else:
                        end_sec = (
                            word["phones"][-1]["start_sec"]
                            + word["phones"][-1]["duration_sec"]
                        )
                elif start_sec >= 0:
                    silence_sec = sum(p["duration_sec"] for p in word["phones"])

                    # Insert a "minor break" for each 1/10 sec of silence
                    num_minor_breaks = int(math.floor(silence_sec / 0.1))
                    break_phonemes = []
                    if num_minor_breaks > 0:
                        # Collect breaks into a single "word"
                        for _ in range(0, num_minor_breaks):
                            break_phonemes.append(_CTX.config.phonemes.minor_break)

                        all_word_phonemes.append(break_phonemes)

                        # Extend utterance to include silence
                        end_sec = (
                            word["phones"][-1]["start_sec"]
                            + word["phones"][-1]["duration_sec"]
                        )

            if has_unknown_words:
                _LOGGER.warning("Unknown word(s) in %s", utt_id)
                # utt_alignments[utt_id] = (
                #     0,
                #     0,
                #     str(audio_path.absolute()),
                #     None,  # phonemes will be generated with gruut instead
                # )
                continue

            # Determine sentence audio duration
            start_sec = max(0, start_sec - buffer_sec)
            end_sec = end_sec + buffer_sec
            if start_sec > end_sec:
                _LOGGER.warning("start > end: %s", align_obj)
                # utt_alignments[utt_id] = (
                #     0,
                #     0,
                #     str(audio_path.absolute()),
                #     None,  # phonemes will be generated with gruut instead
                # )
                continue

            if (end_sec - start_sec) < min_sec:
                _LOGGER.warning("Trimmed audio < %s: %s", min_sec, align_obj)
                # utt_alignments[utt_id] = (
                #     0,
                #     0,
                #     str(audio_path.absolute()),
                #     None,  # phonemes will be generated with gruut instead
                # )
                continue

            # begin/end of trimmed audio
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            # aligned phonemes
            phonemes_str = _CTX.config.phonemes.join_word_phonemes(all_word_phonemes)

            # Appended to each row of text CSV by id
            utt_alignments[utt_id] = (
                start_ms,
                end_ms,
                str(audio_path.absolute()),
                phonemes_str,
            )

    num_skipped = 0

    with open(text_csv_path, "r", encoding="utf-8") as text_csv_file, open(
        align_csv_path, "w", encoding="utf-8"
    ) as align_csv_file:
        reader = csv.reader(text_csv_file, delimiter=_CTX.delimiter)
        writer = csv.writer(align_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            # Input format:
            # id|[speaker]|text|spoken text
            utt_id = row[0]
            alignment = utt_alignments.get(utt_id)
            if alignment is None:
                num_skipped += 1
                continue

            start_ms, end_ms, audio_path, phonemes_str = alignment

            if not phonemes_str:
                # Use entire utterance with gruut phonemes
                _LOGGER.debug("Phonemizing %s", utt_id)
                text = row[-2]
                word_phonemes = []

                for sentence in gruut.sentences(text, lang=dataset_language):
                    for word in sentence:
                        if word.phonemes:
                            word_phonemes.append(word.phonemes)

                if not word_phonemes:
                    _LOGGER.warning("No phonemes for %s", utt_id)
                    continue

                phonemes_str = _CTX.config.phonemes.join_word_phonemes(word_phonemes)

            # Output format:
            # id|[speaker]|text|spoken text|start_ms|end_ms|audio_path|phonemes
            writer.writerow((*row, start_ms, end_ms, audio_path, phonemes_str))

    if num_skipped > 0:
        _LOGGER.warning("Skipped %s utterance(s)", num_skipped)


def do_phonemes_from_align(
    align_csv_path: typing.Union[str, Path],
    text_csv_path: typing.Union[str, Path],
    targets,
):
    """Create metadata files with phonemes from alignment (instead of directly from gruut)"""
    phonemes_csv_path = Path(targets[0])
    phonemes_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # id -> phonemes
    utt_phonemes: typing.Dict[str, str] = {}

    with open(align_csv_path, "r", encoding="utf-8") as align_file:
        reader = csv.reader(align_file, delimiter=_CTX.delimiter)
        for row in reader:
            utt_id = row[0]
            phonemes_str = row[-1]
            utt_phonemes[utt_id] = phonemes_str

    with open(text_csv_path, "r", encoding="utf-8") as text_csv_file, open(
        phonemes_csv_path, "w", encoding="utf-8"
    ) as phonemes_csv_file:
        reader = csv.reader(text_csv_file, delimiter=_CTX.delimiter)
        writer = csv.writer(phonemes_csv_file, delimiter=_CTX.delimiter)

        for row in reader:
            utt_id = row[0]
            phonemes_str = utt_phonemes.get(utt_id)
            if phonemes_str is not None:
                writer.writerow((*row, phonemes_str))
