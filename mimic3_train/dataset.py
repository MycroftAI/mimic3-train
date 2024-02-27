import csv
import logging
import math
import typing
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from phonemes2ids import phonemes2ids
from torch.utils.data import Dataset

from vits_train.config import DatasetConfig, MetadataFormat, TrainingConfig
from vits_train.mel_processing import spectrogram_torch

_LOGGER = logging.getLogger("vits_train.dataset")

_MIN_UTTERANCE_COUNT = 100


@dataclass
class Utterance:
    id: str
    phoneme_ids: typing.List[int]
    audio_path: Path
    spec_path: Path
    speaker_id: typing.Optional[int] = None


@dataclass
class UtteranceTensors:
    id: str
    phoneme_ids: torch.LongTensor
    spectrogram: torch.FloatTensor
    audio_norm: torch.FloatTensor
    spec_length: int
    speaker_id: typing.Optional[torch.LongTensor] = None


@dataclass
class Batch:
    phoneme_ids: torch.LongTensor
    phoneme_lengths: torch.LongTensor
    spectrograms: torch.FloatTensor
    spectrogram_lengths: torch.LongTensor
    audios: torch.FloatTensor
    audio_lengths: torch.LongTensor
    speaker_ids: typing.Optional[torch.LongTensor] = None


UTTERANCE_PHONEME_IDS = typing.Dict[str, typing.List[int]]
UTTERANCE_SPEAKER_IDS = typing.Dict[str, int]
UTTERANCE_IDS = typing.Collection[str]


@dataclass
class DatasetInfo:
    name: str
    cache_dir: Path
    utt_phoneme_ids: UTTERANCE_PHONEME_IDS
    utt_speaker_ids: UTTERANCE_SPEAKER_IDS
    split_ids: typing.Mapping[str, UTTERANCE_IDS]


class MissingDataError(Exception):
    pass


# -----------------------------------------------------------------------------


class PhonemeIdsAndMelsDataset(Dataset):
    def __init__(
        self, config: TrainingConfig, datasets: typing.Sequence[DatasetInfo], split: str
    ):
        super().__init__()

        self.config = config
        self.utterances = []
        self.split = split

        # Check utterances
        speakers_with_data: typing.Dict[str, int] = Counter()

        for dataset in datasets:
            for utt_id in dataset.split_ids.get(split, []):
                audio_path = dataset.cache_dir / f"{utt_id}.audio.pt"
                spec_path = dataset.cache_dir / f"{utt_id}.spec.pt"
                speaker_id = dataset.utt_speaker_ids.get(utt_id)

                if audio_path.is_file() and spec_path.is_file():
                    self.utterances.append(
                        Utterance(
                            id=utt_id,
                            phoneme_ids=dataset.utt_phoneme_ids[utt_id],
                            audio_path=audio_path,
                            spec_path=spec_path,
                            speaker_id=speaker_id,
                        )
                    )

                    if speaker_id is not None:
                        speakers_with_data[speaker_id] += 1
                else:
                    _LOGGER.warning(
                        "Missing audio or spec file: %s %s", audio_path, spec_path
                    )

        # if config.model.is_multispeaker and (
        #     len(speakers_with_data) < config.model.n_speakers
        # ):
        #     # Possibly missing data
        #     raise MissingDataError(
        #         f"Data was found for only {len(speakers_with_data)}/{config.model.n_speakers} speakers",
        #     )

        for speaker_id, utt_count in speakers_with_data.items():
            if utt_count < _MIN_UTTERANCE_COUNT:
                _LOGGER.warning(
                    "Less than %s utterance(s) for speaker %s (found %s)",
                    _MIN_UTTERANCE_COUNT,
                    speaker_id,
                    utt_count,
                )

    def __getitem__(self, index):
        utterance = self.utterances[index]

        # Normalized audio
        audio_norm_path = utterance.audio_path
        audio_norm = torch.load(str(audio_norm_path))

        # Mel spectrogram
        spectrogram_path = utterance.spec_path
        spectrogram = torch.load(str(spectrogram_path))

        speaker_id = None
        if utterance.speaker_id is not None:
            speaker_id = torch.LongTensor([utterance.speaker_id])

        return UtteranceTensors(
            id=utterance.id,
            phoneme_ids=torch.LongTensor(utterance.phoneme_ids),
            audio_norm=audio_norm,
            spectrogram=spectrogram,
            spec_length=spectrogram.size(1),
            speaker_id=speaker_id,
        )

    def __len__(self):
        return len(self.utterances)


class UtteranceCollate:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def __call__(self, utterances: typing.Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0
        multispeaker = False

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if utt.speaker_id is not None:
                multispeaker = True

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.config.segment_size)

        # Create padded tensors
        phonemes_padded = torch.LongTensor(num_utterances, max_phonemes_length)
        spec_padded = torch.FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = torch.FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = torch.LongTensor(num_utterances)
        spec_lengths = torch.LongTensor(num_utterances)
        audio_lengths = torch.LongTensor(num_utterances)

        speaker_ids: typing.Optional[torch.LongTensor] = None
        if multispeaker:
            speaker_ids = torch.LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = [utt.spec_length for utt in dataset]
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


# -----------------------------------------------------------------------------


def load_dataset(
    config: TrainingConfig,
    dataset_name: str,
    dataset_config: DatasetConfig,
    metadata_dir: typing.Union[str, Path],
    cache_dir: typing.Union[str, Path],
    splits=("train",),
    speaker_id_map: typing.Optional[typing.Dict[str, int]] = None,
) -> DatasetInfo:
    metadata_dir = Path(metadata_dir)
    cache_dir = Path(cache_dir)

    multispeaker = config.model.n_speakers > 1
    if multispeaker:
        assert speaker_id_map, "Speaker id map required for multispeaker models"

    ids_format = MetadataFormat.PHONEME_IDS.value

    # Determine data paths
    data_paths: typing.Dict[str, typing.Dict[str, typing.Any]] = defaultdict(dict)
    for split in splits:
        csv_path = metadata_dir / f"{split}_ids.csv"

        data_paths[split]["csv_path"] = csv_path
        data_paths[split]["utt_ids"] = []

    # train set is required
    for split in splits:
        assert data_paths[split][
            "csv_path"
        ].is_file(), f"Missing {split}_ids.csv in {metadata_dir}"

    # Load utterances
    phoneme_to_id = config.phonemes.phoneme_to_id

    utt_phoneme_ids: typing.Dict[str, str] = {}
    utt_speaker_ids: typing.Dict[str, int] = {}

    for split in splits:
        csv_path = data_paths[split]["csv_path"]
        if not csv_path.is_file():
            _LOGGER.debug("Skipping data for %s", split)
            continue

        utt_ids = data_paths[split]["utt_ids"]
        missing_speakers: typing.Set[str] = set()

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_idx, row in enumerate(reader):
                assert len(row) > 1, f"{row} in {csv_path}:{row_idx+1}"
                utt_id, phonemes_ids_strs = row[0], row[-1]

                if multispeaker:
                    assert speaker_id_map is not None

                    if dataset_config.multispeaker:
                        # Speaker id in row
                        speaker_id = row[1]
                        if speaker_id not in speaker_id_map:
                            missing_speakers.add(speaker_id)
                            continue

                        utt_speaker_ids[utt_id] = speaker_id_map[speaker_id]
                    else:
                        # Single speaker for dataset
                        utt_speaker_ids[utt_id] = speaker_id_map[dataset_name]

                phoneme_ids = [int(p_id) for p_id in phonemes_ids_strs.split()]
                phoneme_ids = [
                    p_id for p_id in phoneme_ids if 0 <= p_id < config.model.num_symbols
                ]

                if phoneme_ids:
                    utt_phoneme_ids[utt_id] = phoneme_ids
                    utt_ids.append(utt_id)
                else:
                    _LOGGER.warning("No phoneme ids for %s (%s)", utt_id, csv_path)

        if missing_speakers:
            _LOGGER.warning(
                "%s speaker(s) were unknown: %s",
                len(missing_speakers),
                sorted(missing_speakers),
            )

        _LOGGER.debug(
            "Loaded %s utterance(s) for %s from %s", len(utt_ids), split, csv_path
        )

    # Filter utterances based on min/max settings in config
    _LOGGER.debug("Filtering data")
    drop_utt_ids: typing.Set[str] = set()

    num_phonemes_too_small = 0
    num_phonemes_too_large = 0
    num_audio_missing = 0
    num_spec_too_small = 0
    num_spec_too_large = 0

    for utt_id, phoneme_ids in utt_phoneme_ids.items():
        # Check phonemes length
        if (config.min_seq_length is not None) and (
            len(phoneme_ids) < config.min_seq_length
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_small += 1
            continue

        if (config.max_seq_length is not None) and (
            len(phoneme_ids) > config.max_seq_length
        ):
            drop_utt_ids.add(utt_id)
            num_phonemes_too_large += 1
            continue

    # Filter out dropped utterances
    if drop_utt_ids:
        _LOGGER.info("Dropped %s utterance(s)", len(drop_utt_ids))

        if num_phonemes_too_small > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was smaller than %s",
                num_phonemes_too_small,
                config.min_seq_length,
            )

        if num_phonemes_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose phoneme length was larger than %s",
                num_phonemes_too_large,
                config.max_seq_length,
            )

        if num_audio_missing > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose audio file was missing",
                num_audio_missing,
            )

        if num_spec_too_small > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spectrogram length was smaller than %s",
                num_spec_too_small,
                config.min_spec_length,
            )

        if num_spec_too_large > 0:
            _LOGGER.debug(
                "%s utterance(s) dropped whose spectrogram length was larger than %s",
                num_spec_too_large,
                config.max_spec_length,
            )

        utt_phoneme_ids = {
            utt_id: phoneme_ids
            for utt_id, phoneme_ids in utt_phoneme_ids.items()
            if utt_id not in drop_utt_ids
        }
    else:
        _LOGGER.info("Kept all %s utterances", len(utt_phoneme_ids))

    if not utt_phoneme_ids:
        _LOGGER.warning("No utterances after filtering")

    return DatasetInfo(
        name=dataset_name,
        cache_dir=cache_dir,
        utt_phoneme_ids=utt_phoneme_ids,
        utt_speaker_ids=utt_speaker_ids,
        split_ids={
            split: set(data_paths[split]["utt_ids"]) - drop_utt_ids for split in splits
        },
    )
