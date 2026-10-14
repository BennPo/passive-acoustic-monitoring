import numpy as np
import pytest
import soundfile as sf

from main import AudioProcessor


def test_mono_duration_and_time_axis(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(800), 8000)
    recording = AudioProcessor(path)
    assert recording.channels == 1
    assert recording.original_audio.shape == (800, 1)
    assert recording.audio.shape == (800,)
    assert recording.duration == 0.1
    np.testing.assert_allclose(recording.get_time_axis(), np.arange(800) / 8000)


def test_stereo_preserves_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.column_stack((np.full(800, 0.25), np.full(800, -0.25)))
    sf.write(path, samples, 8000)
    recording = AudioProcessor(path)
    assert recording.channels == 2
    np.testing.assert_allclose(recording.original_audio, samples)
    np.testing.assert_allclose(recording.audio, 0)


def test_missing_recording(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioProcessor(tmp_path / "missing.wav")


def test_unsupported_recording(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text("This is not audio.")
    with pytest.raises(sf.LibsndfileError):
        AudioProcessor(path)
