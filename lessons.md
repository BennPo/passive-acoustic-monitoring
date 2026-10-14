# Passive Acoustic Ocean Monitor

A learning project working towards call detection, species classification, and
eventually experiments in individual animal recognition and live monitoring.

## Lesson 1: Load and understand a recording

Run from the project folder with the existing environment:

```powershell
uv run main.py
uv run main.py "path/to/another-recording.wav"
```

The default recording is the included humpback song clip. The program prints its
sample rate, channel count, samples per channel, duration, and time-axis details.
MP3 support depends on the installed audio decoder; use a WAV copy if necessary.

`AudioProcessor.original_audio` preserves all decoded channels with shape
`(samples, channels)`. `audio` is a separate one-dimensional mono average for
learning. Keep the original channels for any future hydrophone localisation.
Importing `main` does not load a recording or run the command-line program.

### Concepts

- A **sample** is one amplitude measurement at a particular instant.
- **Sample rate** is the number of measurements per second, per channel.
- **Channels** are separate signals recorded together.
- Duration in seconds is `sample_count / sample_rate`.
- Sample number `i` occurs at `i / sample_rate` seconds, starting at zero.
  The final sample therefore occurs one sample interval before the duration.
- `get_time_axis()` calls the method; `get_time_axis` alone refers to the method.

### Your exercise

1. Run the default recording and note its sample rate, channels, and sample count.
2. Calculate the duration yourself and compare it with the printed value.
3. Calculate the interval between samples and explain why the last sample time
   is slightly smaller than the duration.
4. In a Python session, load the recording and compare `original_audio.shape`,
   `audio.shape`, and `get_time_axis().shape`.

## Lesson 2: Plot the waveform

Running `uv run main.py` now also opens a Matplotlib waveform window. The plot
pairs `get_time_axis()` with the mono amplitudes in `recording.audio`.
The horizontal axis shows time in seconds; the vertical axis shows decoded
digital sample values, rather than calibrated underwater sound pressure.

Try zooming into a short section using the plot toolbar. Find a quiet section
and a section with larger amplitudes. A waveform shows how amplitude changes
over time; the next exercise will use a spectrogram to investigate frequencies.

## Lesson 3: See frequencies with a spectrogram

Run `uv run main.py`. The window now shows a waveform above a spectrogram,
with a shared time axis so you can compare the same moment in both plots.

- Horizontal axis: time in seconds (each spectrogram column represents a window).
- Vertical axis: frequency in hertz, or cycles per second.
- Colour: power in decibels relative to the strongest bin in this recording.
  Brighter colours mean stronger frequency components. This is not calibrated
  underwater sound pressure, and colours cannot establish absolute loudness
  across different recordings. Only the strongest 80 dB range is displayed.

`signal.spectrogram()` splits the audio into overlapping windows, applies a
Hann taper to reduce window-edge artefacts, and uses Fourier transforms to
estimate the power at different frequencies within each window.

We start with 1,024 samples per window and 75% overlap. At this clip's 8,000 Hz
sample rate, each window spans 0.128 seconds, windows advance by 0.032 seconds,
and frequency bins are spaced 7.8125 Hz apart. Larger windows distinguish nearby
frequencies better but blur fast changes in time; smaller windows do the reverse.
For shorter recordings, the window shrinks to fit the available samples.

The plot ends at half the sample rate: 4,000 Hz for this clip. This is the
Nyquist frequency, the upper frequency limit for sampled audio. A recording
at this sample rate cannot represent higher-frequency animal sounds faithfully.

### Your exercise

1. Find one bright feature and note its approximate time and frequency.
2. Describe whether it stays level, rises, falls, or repeats. Compare that time
   with the waveform above; a visible feature alone does not identify an animal.
3. Change the window limit from `1024` to `512`, then to `2048`, and run again.
   Compare how sharp the feature looks in time and frequency, then restore `1024`.

Later lessons add annotation, a simple detector, machine learning, species
classification, identity experiments, and finally streaming input.
