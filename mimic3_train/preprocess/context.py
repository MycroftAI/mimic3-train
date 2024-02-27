#!/usr/bin/env python3
import typing
from dataclasses import dataclass, field
from pathlib import Path

from vits_train.config import TrainingConfig, DatasetConfig, MetadataFormat, Aligner


@dataclass
class Context:
    """Shared context for preprocessing"""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path.cwd()
    delimiter: str = "|"
    splits: typing.Sequence[str] = field(default=("train",))
    learn_phonemes: bool = True

    @classmethod
    def get(cls) -> "Context":
        """Get or create shared context object"""
        global_context = getattr(cls, "_context", None)
        if global_context is None:
            global_context = Context()
            setattr(cls, "_context", global_context)

        return global_context

    @property
    def phoneme_map_path(self) -> Path:
        return self.output_dir / "phonemes.txt"

    def dataset_dir(self, dataset: DatasetConfig) -> Path:
        return self.output_dir / dataset.name

    def dataset_language(self, dataset: DatasetConfig) -> str:
        return dataset.text_language or self.config.text_language or "en_US"

    def ids_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        ids_format = MetadataFormat.PHONEME_IDS.value

        return self.dataset_dir(dataset) / f"{split}_{ids_format}.csv"

    def text_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        text_format = MetadataFormat.TEXT.value

        return self.dataset_dir(dataset) / f"{split}_{text_format}.csv"

    def phonemes_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        phonemes_format = MetadataFormat.PHONEMES.value

        return self.dataset_dir(dataset) / f"{split}_{phonemes_format}.csv"

    def audio_list_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        metadata_format = dataset.metadata_format.value

        return self.dataset_dir(dataset) / f"{split}_{metadata_format}_audio.csv"

    def audio_cache_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        metadata_format = dataset.metadata_format.value

        return self.dataset_dir(dataset) / f"{split}_{metadata_format}_audio_cache.csv"

    def mel_cache_path(self, dataset: DatasetConfig, split: str = "train") -> Path:
        metadata_format = dataset.metadata_format.value

        return self.dataset_dir(dataset) / f"{split}_{metadata_format}_mels_cache.csv"

    def dataset_aligner(self, dataset: DatasetConfig) -> typing.Optional[typing.Any]:
        aligner_config = self.config.text_aligner
        if aligner_config:
            if aligner_config.aligner == Aligner.KALDI_ALIGN:
                from kaldi_align import KaldiAligner

                dataset_lang = self.dataset_language(dataset)
                dataset_dir = self.dataset_dir(dataset)
                aligner = KaldiAligner(
                    language=dataset_lang, output_dir=dataset_dir / "align"
                )

        return aligner

    def find_audio_file(
        self, audio_dir: Path, utterance_id: str
    ) -> typing.Optional[Path]:
        audio_path = audio_dir / utterance_id
        exts = ["ogg", "mp3", "flac", "wav"]
        while exts and (not audio_path.is_file()):
            ext = exts.pop()
            audio_path = audio_dir / f"{utterance_id}.{ext}"

        if audio_path.is_file():
            return audio_path

        return None
