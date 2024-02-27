#!/usr/bin/env python3
import csv
import logging
import json
import typing
from collections import Counter
from pathlib import Path

from vits_train.config import MetadataFormat

from .context import Context

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess")


def task_speaker_map():
    if not _CTX.config.is_multispeaker:
        return

    speaker_map_path = _CTX.output_dir / "speaker_map.csv"
    speakers_paths = {}

    for dataset in _CTX.config.datasets:
        dataset_dir = _CTX.dataset_dir(dataset)
        speakers_path = dataset_dir / "speakers.txt"
        speakers_paths[dataset.name] = speakers_path

        speaker_counts_path = dataset_dir / "speaker_counts.json"

        if dataset.multispeaker:
            for split in _CTX.splits:
                ids_path = _CTX.ids_path(dataset, split)

                # Get utterance counts by speaker
                yield {
                    "name": str(speaker_counts_path.relative_to(_CTX.output_dir)),
                    "actions": [(do_speaker_counts, [ids_path])],
                    "file_dep": [ids_path],
                    "targets": [speaker_counts_path],
                }

            # Generate list of unique speakers in dataset
            yield {
                "name": str(speakers_path.relative_to(_CTX.output_dir)),
                "actions": [(do_multiple_speakers, [speaker_counts_path])],
                "file_dep": [speaker_counts_path],
                "targets": [speakers_path],
            }
        else:
            # Use dataset name as "speaker"
            yield {
                "name": str(speakers_path.relative_to(_CTX.output_dir)),
                "actions": [(do_single_speakers, [dataset.name])],
                "targets": [speakers_path],
                "uptodate": [True],
            }

    # Get mapping from speaker names to ids
    yield {
        "name": str(speaker_map_path.relative_to(_CTX.output_dir)),
        "actions": [(do_speaker_map, [speakers_paths])],
        "file_dep": list(speakers_paths.values()),
        "targets": [speaker_map_path],
    }


# -----------------------------------------------------------------------------


def do_speaker_map(speakers_paths: typing.Dict[str, typing.Union[str, Path]], targets):
    speaker_map_path = Path(targets[0])
    speaker_map_path.parent.mkdir(parents=True, exist_ok=True)

    speaker_idx = 0

    with open(speaker_map_path, "w", encoding="utf-8") as speaker_map_file:
        writer = csv.writer(speaker_map_file, delimiter=_CTX.delimiter)
        for dataset_name, speakers_path in speakers_paths.items():
            with open(speakers_path, "r", encoding="utf-8") as speakers_file:
                for line in speakers_file:
                    line = line.strip()
                    if not line:
                        continue

                    speaker = line
                    writer.writerow((str(speaker_idx), dataset_name, speaker))

                    speaker_idx += 1


def do_speaker_counts(ids_csv_path: typing.Union[str, Path], targets):
    speaker_counts_path = Path(targets[0])
    speaker_counts_path.parent.mkdir(parents=True, exist_ok=True)

    speaker_counts: Counter[str] = Counter()

    with open(ids_csv_path, "r", encoding="utf-8") as ids_file:
        reader = csv.reader(ids_file, delimiter=_CTX.delimiter)
        for row in reader:
            speaker = row[1]
            speaker_counts[speaker] += 1

    with open(speaker_counts_path, "w", encoding="utf-8") as speaker_counts_file:
        json.dump(speaker_counts, speaker_counts_file, indent=4, ensure_ascii=False)


def do_single_speakers(dataset_name: str, targets):
    speakers_path = Path(targets[0])
    speakers_path.parent.mkdir(parents=True, exist_ok=True)
    speakers_path.write_text(dataset_name)


def do_multiple_speakers(speaker_counts_path: typing.Union[str, Path], targets):
    speakers_path = Path(targets[0])
    speakers_path.parent.mkdir(parents=True, exist_ok=True)

    with open(speaker_counts_path, "r", encoding="utf-8") as speaker_counts_file:
        speaker_counts = json.load(speaker_counts_file)

    with open(speakers_path, "w", encoding="utf-8") as speakers_file:
        for speaker, _count in sorted(
            speaker_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            num_utterances = speaker_counts[speaker]

            if (_CTX.config.min_speaker_utterances is not None) and (
                num_utterances < _CTX.config.min_speaker_utterances
            ):
                _LOGGER.warning(
                    "Skipping speaker %s with %s < %s utterances",
                    speaker,
                    num_utterances,
                    _CTX.config.min_speaker_utterances,
                )
                continue

            print(speaker, file=speakers_file)
