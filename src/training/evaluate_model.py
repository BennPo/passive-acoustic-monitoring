"""Load saved weights and make a few paired audio comparisons for listening."""

import argparse
from itertools import islice
import math
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch

from .create_mixtures import give_noise
from .dataset import create_loaders
from .model import OceanDenoiser


def load_model(checkpoint_path):
    """Recreate the saved architecture, restore its weights and select eval mode."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = OceanDenoiser(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def denoise_clip(model, mixture):
    """Run one mono clip through the model without calculating gradients."""
    samples = np.asarray(mixture, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Expected a nonempty mono clip")
    inputs = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    estimate = model(inputs)[0, 0].numpy()
    if not np.isfinite(estimate).all():
        raise ValueError("The model produced nonfinite audio values")
    return estimate


def save_comparison(output_dir, mixture, reference, estimate, sample_rate):
    """Save all three versions with a shared gain, preserving relative levels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = {"mixture": mixture, "reference": reference, "estimate": estimate}
    peak = max(float(np.max(np.abs(clip))) for clip in clips.values())
    gain = min(1.0, 0.95 / max(peak, 1e-8))
    for name, clip in clips.items():
        sf.write(output_dir / f"{name}.wav", clip * gain, sample_rate, subtype="FLOAT")
    return gain


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=project_root / "outputs/ocean_denoiser.pt",
    )
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs/evaluation",
    )
    args = parser.parse_args()
    if args.clips < 1:
        parser.error("clips must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, checkpoint = load_model(args.checkpoint)
    sample_rate = int(checkpoint["sample_rate"])
    if sample_rate <= 0:
        parser.error("The checkpoint sample rate must be positive")

    _, validation_loader = create_loaders(batch_size=1)
    validation_dataset = validation_loader.dataset
    # A short source keeps this first listening check quick to load.
    recording = min(validation_dataset.audio_files, key=lambda path: sf.info(path).duration)
    recording_info = sf.info(recording)
    resampled_length = math.ceil(recording_info.frames * sample_rate / recording_info.samplerate)

    # Reuse the same mixture choices on each run with unchanged source files.
    random.seed(42)
    np.random.seed(42)
    print("Paired listening check using saved model weights.")
    print("Completed training epochs:", checkpoint.get("epochs_completed", "unknown"))
    print("Current validation source:", recording)
    print("This checks a few clips, not the full validation collection.")
    print("The current split matches training only if its source file lists are unchanged.")

    pairs = give_noise(
        recording, validation_dataset.background_files, target_sample_rate=sample_rate,
    )
    example_count = 0
    for index, (mixture, reference, _) in enumerate(islice(pairs, args.clips)):
        estimate = denoise_clip(model, mixture)
        # Infer on the padded five-second clip, then exclude padding from scores/files.
        clip_samples = len(reference)
        valid_samples = min(clip_samples, resampled_length - index * clip_samples)
        mixture = mixture[:valid_samples]
        reference = reference[:valid_samples]
        estimate = estimate[:valid_samples]
        input_error = float(np.mean(np.abs(mixture - reference)))
        output_error = float(np.mean(np.abs(estimate - reference)))

        destination = args.output_dir / f"clip_{index + 1:02d}"
        gain = save_comparison(destination, mixture, reference, estimate, sample_rate)
        start = index * clip_samples / sample_rate
        end = start + valid_samples / sample_rate
        print(
            f"Clip {index + 1} ({start:.2f}-{end:.2f}s): "
            f"input MAE = {input_error:.6f}, estimate MAE = {output_error:.6f}, "
            f"shared playback gain = {gain:.3f}", flush=True,
        )
        example_count += 1

    if example_count == 0:
        parser.error("The selected recording produced no clips")
    print(f"Saved {example_count} listening comparisons to {args.output_dir}")
    print("Listen for preserved calls and reduced noise; lower MAE can also reflect attenuation.")


if __name__ == "__main__":
    main()
