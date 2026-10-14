import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
import scipy.signal as signal
from pathlib import Path

import matplotlib.pyplot as plt


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



def main():
    test = AudioProcessor("data/Meno-song-NOAA-PAGroup-13-humpback-clip.wav")

    clip = test.audio
    win = signal.windows.hann(1024, sym=False)
    
    analyser = signal.ShortTimeFFT(
        win=win,
        hop=256,
        fs=test.sample_rate,
        scale_to="psd",
        fft_mode="onesided2X"
    )

    power = analyser.spectrogram(clip)
    frequencies = analyser.f
    times = analyser.t(len(clip)) + start / test.sample_rate

    floor = np.finfo(float).tiny
    reference = max(float(power.max()), floor)
    power_db = 10 * np.log10(np.maximum(power, floor) / reference)

    plt.figure(figsize=(12, 4))
    image = plt.pcolormesh(
                times, frequencies, power_db, shading="auto", cmap="magma",
                vmin=-80, vmax=0,
            )

    plt.xlabel("Time in recording (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram: seconds 2-6")
    plt.xlim(2, 6)
    plt.colorbar(image, label="Power (dB relative to clip peak)")
    plt.tight_layout()
    plt.show()

    print("Window shape:", win.shape)
    print("Frequency count:", len(frequencies))
    print("Time count:", len(times))
    print("Power shape:", power.shape)
    
main()

