#!/usr/bin/env python3
import csv
import math
import json
import logging
import shutil
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

from vits_train.config import DatasetConfig
from vits_train.preprocess.context import Context
from vits_train.mel_processing import spectrogram_torch

_CTX = Context.get()
_LOGGER = logging.getLogger("preprocess.audio")


def task_audio_norm():
    """Generate normalized audio cache"""
    for dataset in _CTX.config.datasets:
        for split in _CTX.splits:
            audio_csv_path = _CTX.audio_list_path(dataset, split)
            cache_csv_path = _CTX.audio_cache_path(dataset, split)

            yield {
                "name": str(cache_csv_path.relative_to(_CTX.output_dir)),
                "actions": [(do_audio_norm, [audio_csv_path, dataset])],
                "file_dep": [audio_csv_path],
                "targets": [cache_csv_path],
            }


def task_mels():
    """Generate mel spectrograms from audio"""
    for dataset in _CTX.config.datasets:
        for split in _CTX.splits:
            audio_csv_path = _CTX.audio_list_path(dataset, split)
            cache_csv_path = _CTX.mel_cache_path(dataset, split)

            yield {
                "name": str(cache_csv_path.relative_to(_CTX.output_dir)),
                "actions": [(do_mels, [audio_csv_path, dataset])],
                "file_dep": [audio_csv_path],
                "targets": [cache_csv_path],
            }


def task_mel_stats():
    """Create mel scale stats for dataset"""
    if not _CTX.config.audio.scale_mels:
        return

    metadata_format = _CTX.config.dataset_format.value

    stats_path = _CTX.output_dir / "scale_stats.npy"
    cache_csv_paths = []

    for dataset in _CTX.config.datasets:
        dataset_dir = _CTX.output_dir / dataset.name

        for split in _CTX.splits:
            cache_csv_path = _CTX.mel_cache_path(dataset, split)
            cache_csv_paths.append(cache_csv_path)

    yield {
        "name": str(stats_path.relative_to(_CTX.output_dir)),
        "actions": [(do_mel_stats, [cache_csv_paths])],
        "file_dep": cache_csv_paths,
        "targets": [stats_path],
    }


# -----------------------------------------------------------------------------


def do_audio_norm(
    audio_csv_path: typing.Union[str, Path], dataset_config: DatasetConfig, targets
):
    """Generate normalized audio cache"""
    cache_dir = dataset_config.get_cache_dir(_CTX.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_csv_path = Path(targets[0])
    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def cache_audio(row):
        utt_id, start_ms, end_ms, audio_path_str = (
            row[0],
            int(row[-4]),
            int(row[-3]),
            row[-2],
            # phonemes
        )
        cache_path = cache_dir / f"{utt_id}.audio.pt"
        if cache_path.is_file():
            return (utt_id, cache_path)

        audio_path = Path(audio_path_str)

        load_args = {"path": audio_path, "sr": _CTX.config.audio.sample_rate}

        if (start_ms > 0) or (end_ms > 0):
            load_args["offset"] = start_ms / 1000.0
            load_args["duration"] = (end_ms - start_ms) / 1000.0

        try:
            audio, _sample_rate = librosa.load(**load_args)
        except Exception as e:
            _LOGGER.exception(audio_path)
            raise e

        # NOTE: audio is already in [-1, 1] coming from librosa
        audio_norm = torch.FloatTensor(audio).unsqueeze(0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(audio_norm, cache_path)

        return (utt_id, cache_path)

    with open(audio_csv_path, "r", encoding="utf-8") as audio_csv_file, open(
        cache_csv_path, "w", encoding="utf-8"
    ) as cache_csv_file, ThreadPoolExecutor(max_workers=8) as executor:
        reader = csv.reader(audio_csv_file, delimiter=_CTX.delimiter)
        writer = csv.writer(cache_csv_file, delimiter=_CTX.delimiter)

        for row in executor.map(cache_audio, reader):
            writer.writerow(row)


def do_mels(
    audio_csv_path: typing.Union[str, Path], dataset_config: DatasetConfig, targets
):
    """Generate mel spectrograms from audio"""
    cache_dir = dataset_config.get_cache_dir(_CTX.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_csv_path = Path(targets[0])
    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def cache_mel(row):
        utt_id, start_ms, end_ms, audio_path_str = (
            row[0],
            int(row[-4]),
            int(row[-3]),
            row[-2],
            # phonemes
        )
        cache_path = cache_dir / f"{utt_id}.spec.pt"
        if cache_path.is_file():
            return (utt_id, cache_path)

        audio_path = Path(audio_path_str)

        load_args = {"path": audio_path, "sr": _CTX.config.audio.sample_rate}

        if (start_ms > 0) or (end_ms > 0):
            load_args["offset"] = start_ms / 1000.0
            load_args["duration"] = (end_ms - start_ms) / 1000.0

        audio, _sample_rate = librosa.load(**load_args)

        # NOTE: audio is already in [-1, 1] coming from librosa
        audio_norm = torch.FloatTensor(audio).unsqueeze(0)

        mel = spectrogram_torch(
            y=audio_norm,
            n_fft=_CTX.config.audio.filter_length,
            sampling_rate=_CTX.config.audio.sample_rate,
            hop_size=_CTX.config.audio.hop_length,
            win_size=_CTX.config.audio.win_length,
            center=False,
        ).squeeze(0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, cache_path)

        return (utt_id, cache_path)

    with open(audio_csv_path, "r", encoding="utf-8") as audio_csv_file, open(
        cache_csv_path, "w", encoding="utf-8"
    ) as cache_csv_file, ThreadPoolExecutor(max_workers=8) as executor:
        reader = csv.reader(audio_csv_file, delimiter=_CTX.delimiter)
        writer = csv.writer(cache_csv_file, delimiter=_CTX.delimiter)

        for row in executor.map(cache_mel, reader):
            writer.writerow(row)


def do_mel_stats(cache_csv_paths: typing.Iterable[typing.Union[str, Path]], targets):
    """Create mel scale stats for dataset"""
    target_path = Path(targets[0])
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def mel_sums(row):
        cache_path = Path(row[-1])
        mel = torch.load(cache_path)

        mel_n = mel.shape[1]
        mel_sum = mel.sum(1)
        mel_square_sum = (mel**2).sum(axis=1)

        return (mel_n, mel_sum, mel_square_sum)

    total_mel_n: typing.Optional[torch.Tensor] = None
    total_mel_sum: typing.Optional[torch.Tensor] = None
    total_mel_square_sum: typing.Optional[torch.Tensor] = None

    for cache_csv_path in cache_csv_paths:
        with open(
            cache_csv_path, "r", encoding="utf-8"
        ) as input_file, ThreadPoolExecutor() as executor:
            reader = csv.reader(input_file, delimiter=_CTX.delimiter)

            for mel_n, mel_sum, mel_square_sum in executor.map(mel_sums, reader):
                if total_mel_n is None:
                    total_mel_n = mel_n
                else:
                    total_mel_n += mel_n

                if total_mel_sum is None:
                    total_mel_sum = mel_sum
                else:
                    total_mel_sum += mel_sum

                if total_mel_square_sum is None:
                    total_mel_square_sum = mel_square_sum
                else:
                    total_mel_square_sum += mel_square_sum

    assert (
        (total_mel_n is not None)
        and (total_mel_sum is not None)
        and (total_mel_square_sum is not None)
    )
    mel_mean = total_mel_sum / total_mel_n
    mel_scale = np.sqrt(total_mel_square_sum / total_mel_n - mel_mean**2)
    stats = {"mel_mean": mel_mean.numpy(), "mel_scale": mel_scale.numpy()}

    np.save(target_path, stats, allow_pickle=True)
