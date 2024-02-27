#!/usr/bin/env python3
import argparse
import logging
import random
import sys
from pathlib import Path

import doit

from ..config import TrainingConfig

from . import (
    Context,
    task_text_to_ids,
    task_phoneme_map,
    task_speaker_map,
    task_audio_norm,
    task_mels,
    task_mel_stats,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to JSON configuration file"
    )
    parser.add_argument("--output-dir", help="Path to output directory")
    parser.add_argument(
        "--no-learn-phonemes",
        action="store_true",
        help="Don't learn phonemes (requires phonemes.txt)",
    )
    args, rest_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    context = Context.get()

    if args.config:
        args.config = Path(args.config)

        with open(args.config, "r", encoding="utf-8") as config_file:
            context.config = TrainingConfig.load(config_file)

        # Default to directory of config file
        context.output_dir = args.config.parent

    if args.output_dir:
        context.output_dir = Path(args.output_dir)

    context.output_dir.mkdir(parents=True, exist_ok=True)

    context.learn_phonemes = not args.no_learn_phonemes

    random.seed(context.config.seed)

    sys.argv[1:] = rest_args

    doit.run(globals())
