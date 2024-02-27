# Copyright 2022 Mycroft AI Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import logging
import time
import typing
from collections import Counter
from pathlib import Path

import torch
import torch.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, save_checkpoint
from .commons import clip_grad_value_, slice_segments
from .config import TrainingConfig
from .dataset import Batch
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    setup_discriminator,
    setup_model,
    setup_optimizer,
    setup_scheduler,
)
from .utils import to_gpu

_LOGGER = logging.getLogger("mimic3_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    model_g: typing.Optional[SynthesizerTrn] = None,
    model_d: typing.Optional[MultiPeriodDiscriminator] = None,
    optimizer_g: typing.Optional[torch.optim.AdamW] = None,
    optimizer_d: typing.Optional[torch.optim.AdamW] = None,
    scheduler_g: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_d: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_epochs: int = 100,
    val_epochs: int = 1,
    rank: int = 0,
    save_best: bool = True,
):
    """Run training for the specified number of epochs"""
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)

    if model_g is None:
        model_g = setup_model(config)

    assert model_g is not None

    if model_d is None:
        model_d = setup_discriminator(config)

    assert model_d is not None

    if optimizer_g is None:
        optimizer_g = setup_optimizer(config, model_g)

    assert optimizer_g is not None

    if optimizer_d is None:
        optimizer_d = setup_optimizer(config, model_d)

    assert optimizer_d is not None

    if scheduler_g is None:
        scheduler_g = setup_scheduler(config, optimizer_g)

    assert scheduler_g is not None

    if scheduler_d is None:
        scheduler_d = setup_scheduler(config, optimizer_d)

    assert scheduler_d is not None

    # Gradient scaler
    scaler = GradScaler(enabled=config.fp16_run)
    if config.fp16_run:
        _LOGGER.info("Using fp16 scaler")

    # Begin training
    best_val_loss = config.best_loss
    global_step = config.global_step

    bad_utterance_counts = Counter()
    bad_utterances_path = model_dir / "bad_utterances.txt"

    if bad_utterances_path.is_file():
        # Load bad counts
        with open(bad_utterances_path, "r", encoding="utf-8") as bad_utterances_file:
            for line in bad_utterances_file:
                line = line.strip()
                if not line:
                    continue

                utt_id, count_str = line.split(maxsplit=1)
                bad_utterance_counts[utt_id] = int(count_str)

    for epoch in range(config.last_epoch, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s, learning_rates=%s)",
            epoch,
            config.epochs,
            global_step,
            [optimizer_g.param_groups[0]["lr"], optimizer_d.param_groups[0]["lr"]],
        )
        epoch_start_time = time.perf_counter()
        last_global_step = global_step
        global_step = train_step(
            rank=rank,
            global_step=global_step,
            epoch=epoch,
            model_g=model_g,
            model_d=model_d,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            config=config,
            train_loader=train_loader,
            scaler=scaler,
            bad_utterance_counts=bad_utterance_counts,
        )

        assert global_step > last_global_step

        scheduler_g.step()
        scheduler_d.step()

        config.global_step = global_step
        config.last_epoch = epoch

        if rank == 0:
            if save_best:
                save_names = ["best_model"]
            else:
                save_names = []

            if (epoch % checkpoint_epochs) == 0:
                save_names.append(f"checkpoint_{global_step}")

            for name in save_names:
                model_path = model_dir / f"{name}.pth"
                _LOGGER.debug("Saving model to %s", model_path)
                save_checkpoint(
                    Checkpoint(
                        model_g=model_g,
                        model_d=model_d,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        scheduler_g=scheduler_g,
                        scheduler_d=scheduler_d,
                        global_step=global_step,
                        epoch=epoch,
                        version=config.version,
                        best_loss=best_val_loss,
                    ),
                    model_path,
                )

                # Save config too
                model_config_path = model_path.with_suffix(".config.json")
                with open(
                    model_config_path, "w", encoding="utf-8"
                ) as model_config_file:
                    config.save(model_config_file)

            if not save_best:
                checkpoints = list(model_dir.glob("checkpoint_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(
                        checkpoints, key=lambda p: p.stat().st_mtime
                    )

                    # Symlink best
                    best_model_path = model_dir / "best_model.pth"
                    best_model_path.unlink(missing_ok=True)
                    best_model_path.symlink_to(latest_checkpoint)

                    latest_config = latest_checkpoint.with_suffix(".config.json")
                    best_config_path = model_dir / "best_model.config.json"
                    best_config_path.unlink(missing_ok=True)
                    best_config_path.symlink_to(latest_config)

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "[%s] epoch %s complete in %s second(s) (global step=%s)",
            rank,
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    rank: int,
    global_step: int,
    epoch: int,
    model_g: SynthesizerTrn,
    model_d: MultiPeriodDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: TrainingConfig,
    train_loader: DataLoader,
    scaler: GradScaler,
    bad_utterance_counts: typing.Optional[typing.Counter[str]] = None,
):
    steps_per_epoch = len(train_loader)

    model_g.train()
    model_d.train()

    for batch_idx, batch in enumerate(train_loader):
        try:
            batch = typing.cast(Batch, batch)
            x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
                to_gpu(batch.phoneme_ids),
                to_gpu(batch.phoneme_lengths),
                to_gpu(batch.audios),
                to_gpu(batch.audio_lengths),
                to_gpu(batch.spectrograms),
                to_gpu(batch.spectrogram_lengths),
                to_gpu(batch.speaker_ids) if batch.speaker_ids is not None else None,
            )

            # Train model
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with autocast(enabled=config.fp16_run):
                (
                    y_hat,
                    l_length,
                    _attn,
                    ids_slice,
                    _x_mask,
                    z_mask,
                    (_z, z_p, m_p, logs_p, _m_q, logs_q),
                ) = model_g(x, x_lengths, spec, spec_lengths, speaker_ids)

                mel = spec_to_mel_torch(
                    spec,
                    config.audio.filter_length,
                    config.audio.mel_channels,
                    config.audio.sample_rate,
                    config.audio.mel_fmin,
                    config.audio.mel_fmax,
                )
                y_mel = slice_segments(
                    mel, ids_slice, config.segment_size // config.audio.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    config.audio.filter_length,
                    config.audio.mel_channels,
                    config.audio.sample_rate,
                    config.audio.hop_length,
                    config.audio.win_length,
                    config.audio.mel_fmin,
                    config.audio.mel_fmax,
                )

                y = slice_segments(
                    y, ids_slice * config.audio.hop_length, config.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = model_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
        except Exception:
            _LOGGER.exception("slice_segments")
            continue

        # Discriminator loss
        # Run here because loss variable is modified
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optimizer_d)
        clip_grad_value_(model_d.parameters(), config.grad_clip)
        scaler.step(optimizer_d)

        with autocast(enabled=config.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = model_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        # Generator loss
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optimizer_g)
        clip_grad_value_(model_g.parameters(), config.grad_clip)
        scaler.step(optimizer_g)

        scaler.update()

        _LOGGER.debug(
            "[%s] loss: gen=%s, disc=%s (epoch=%s, step=%s/%s)",
            rank,
            loss_gen_all.item(),
            loss_disc_all.item(),
            epoch,
            batch_idx + 1,
            steps_per_epoch,
        )
        global_step += 1

    return global_step
