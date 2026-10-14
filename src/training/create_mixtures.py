from audiomentations import AddBackgroundNoise

import numpy as np
import soundfile as sf
from pathlib import Path
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


# Select from the clean-call folder, rather than background recordings.
audio_files = list(Path("data/clean_calls").glob("*.wav"))

if not audio_files:
    raise FileNotFoundError("No WAV files found in data/clean_calls")

filepath = random.choice(audio_files)
print("Selected:", filepath)

add_noise = AddBackgroundNoise(
    sounds_path="data/backgrounds",
    min_snr_db=-10,
    max_snr_db=-10,
    p=1.0,
)

data = AudioProcessor(filepath)

clean = data.audio.astype(np.float32).copy()
mixture = add_noise(clean.copy(), sample_rate=data.sample_rate)

sf.write("mixture_test.wav", mixture, data.sample_rate)