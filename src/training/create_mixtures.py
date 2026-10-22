from audiomentations import AddBackgroundNoise

import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as signal

import random

class AudioProcessor:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Recording not found: {self.filepath}")

        self.original_audio, self.sample_rate = sf.read(
                            self.filepath, always_2d=True
                        )

        self.channels = self.original_audio.shape[1]
        self.sample_count = self.original_audio.shape[0]

        self.duration = self.sample_count / self.sample_rate

        # Create seperate mono channel
        if len(self.original_audio.shape) == 2:
            self.audio = np.mean(self.original_audio, axis=1, dtype=self.original_audio.dtype)

    def get_time_axis(self):
            return np.arange(len(self.audio)) / self.sample_rate


def split_into_clips(audio, sample_rate, clip_seconds=5):
    samples_per_clip = int(clip_seconds * sample_rate)

    for start in range(0, len(audio), samples_per_clip):
        end = start + samples_per_clip
        section = audio[start:end].copy()

        if len(section) < samples_per_clip:
            section = np.pad(
                section,
                (0, samples_per_clip - len(section)),
            )

        yield section


def give_noise(filepath, background_files, target_sample_rate=8000, training=False):
    print("Selected:", filepath)

    add_noise = AddBackgroundNoise(
        sounds_path=background_files,
        min_snr_db=-8,
        max_snr_db=8,
        p=1.0,
    )

    if training:
        background_only = AddBackgroundNoise(
            sounds_path=background_files,
            noise_rms="absolute",
            min_absolute_rms_db=-35,
            max_absolute_rms_db=-15,
            p=1.0,
        )

    data = AudioProcessor(filepath)
    clean = data.audio.astype(np.float32).copy()

    if data.sample_rate != target_sample_rate:
        clean = signal.resample_poly(
            clean,
            up=target_sample_rate,
            down=data.sample_rate,
        ).astype(np.float32)

    for section in split_into_clips(clean, target_sample_rate):
        situation = "noisy"

        if training:
            situation = random.choices(
                ["noisy", "unchanged", "background"],
                weights=[80, 10, 10],
            )[0]

        target = section

        if situation == "noisy":
            mixture = add_noise(
                samples=section.copy(),
                sample_rate=target_sample_rate,
            )

        elif situation == "unchanged":
            mixture = section.copy()

        else:
            target = np.zeros_like(section)
            mixture = background_only(
                samples=target.copy(),
                sample_rate=target_sample_rate,
            )

        yield mixture, target, target_sample_rate

