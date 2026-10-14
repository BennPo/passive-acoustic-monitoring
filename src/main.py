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
    # ShortTimeFFT already returns times in seconds using its fs setting.
    times = analyser.t(len(clip))

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
    plt.title("Spectrogram:")
    plt.xlim(0, test.duration)
    plt.colorbar(image, label="Power (dB relative to clip peak)")
    plt.tight_layout()


    band_mask = (frequencies >= 150) & (frequencies <= 1000)
    band_power = power[band_mask, :]
    band_energy = np.sum(band_power, axis=0) * analyser.delta_f
    threshold = 0.01
    above_threshold = band_energy > threshold

    plt.figure(figsize=(12, 4))
    plt.plot(times, band_energy)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Band power (digital amplitude²)")
    plt.title("Power between 150 and 1,000 Hz")
    plt.xlim(0, test.duration)
    plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label="Candidate threshold",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    
main()

